//! Persistent tag cache using redb.
//!
//! Strategy: Cache parsed tags per file, keyed by (path, mtime).
//! On cache hit with matching mtime, skip parsing entirely = 100x+ speedup.
//!
//! Cache structure:
//! - Database: .ripmap.cache/tags.redb (redb provides ACID guarantees)
//! - Key: file path (relative to project root)
//! - Value: bincode-serialized (mtime_secs, mtime_nanos, Vec<Tag>)
//!
//! Design decisions:
//! - Bincode for compact binary serialization (faster than JSON, smaller than msgpack)
//! - mtime stored in value for atomic validation (no separate metadata table)
//! - redb for zero-copy reads and write durability without WAL overhead
//! - Cache directory in .ripmap.cache/ to keep project root clean

use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result};
use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
use serde::{Deserialize, Serialize};

use crate::types::Tag;

/// Table definition for tag cache.
/// Key = file path (relative), Value = serialized CacheEntry
const TAGS_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("tags");

/// Cache entry containing mtime validation data + parsed tags.
/// Stored as bincode bytes in redb for compact representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry {
    /// Modification time seconds since UNIX_EPOCH
    mtime_secs: u64,
    /// Modification time nanoseconds component
    mtime_nanos: u32,
    /// Parsed tags for this file
    tags: Vec<Tag>,
}

impl CacheEntry {
    /// Create entry from SystemTime and tags
    fn new(mtime: SystemTime, tags: Vec<Tag>) -> Result<Self> {
        let duration = mtime
            .duration_since(SystemTime::UNIX_EPOCH)
            .context("File mtime is before UNIX_EPOCH")?;

        Ok(Self {
            mtime_secs: duration.as_secs(),
            mtime_nanos: duration.subsec_nanos(),
            tags,
        })
    }

    /// Check if this entry's mtime matches the given mtime
    fn is_valid(&self, mtime: SystemTime) -> bool {
        let Ok(duration) = mtime.duration_since(SystemTime::UNIX_EPOCH) else {
            return false;
        };

        self.mtime_secs == duration.as_secs()
            && self.mtime_nanos == duration.subsec_nanos()
    }

    /// Serialize to bytes using bincode
    fn to_bytes(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).context("Failed to serialize cache entry")
    }

    /// Deserialize from bytes
    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        bincode::deserialize(bytes).context("Failed to deserialize cache entry")
    }
}

/// Persistent tag cache backed by redb.
///
/// Provides mtime-validated caching of parsed tags to avoid re-parsing
/// unchanged files. Cache hits can provide 100x+ speedup on warm runs.
pub struct TagCache {
    /// redb database handle (thread-safe, uses parking_lot internally)
    db: Database,
    /// Path to cache directory (.ripmap.cache/)
    /// Reserved for future features (e.g., cache cleanup, size management)
    #[allow(dead_code)]
    cache_dir: PathBuf,
}

impl TagCache {
    /// Open or create the tag cache database.
    ///
    /// Cache location: `<root>/.ripmap.cache/tags.redb`
    ///
    /// Creates the cache directory if it doesn't exist.
    /// Returns error if directory creation or database opening fails.
    pub fn open(root: &Path) -> Result<Self> {
        let cache_dir = root.join(".ripmap.cache");

        // Ensure cache directory exists
        fs::create_dir_all(&cache_dir)
            .with_context(|| format!("Failed to create cache directory: {}", cache_dir.display()))?;

        let db_path = cache_dir.join("tags.redb");

        // Open or create database
        // redb automatically handles schema migration and corruption recovery
        let db = Database::create(&db_path)
            .with_context(|| format!("Failed to open cache database: {}", db_path.display()))?;

        Ok(Self { db, cache_dir })
    }

    /// Get cached tags for a file if the cache entry is still valid.
    ///
    /// Returns `Some(tags)` if:
    /// - File exists in cache
    /// - Stored mtime matches current mtime
    ///
    /// Returns `None` if:
    /// - File not in cache
    /// - mtime mismatch (file was modified)
    /// - Deserialization error (cache corruption)
    pub fn get(&self, fname: &str, mtime: SystemTime) -> Option<Vec<Tag>> {
        // Read transaction - zero-copy access to cached data
        let read_txn = self.db.begin_read().ok()?;
        let table = read_txn.open_table(TAGS_TABLE).ok()?;

        // Lookup by file path
        let value_guard = table.get(fname).ok()??;
        let bytes = value_guard.value();

        // Deserialize and validate mtime
        let entry = CacheEntry::from_bytes(bytes).ok()?;

        if entry.is_valid(mtime) {
            Some(entry.tags)
        } else {
            // mtime mismatch - file was modified, cache invalid
            None
        }
    }

    /// Store tags for a file with its current mtime.
    ///
    /// Overwrites any existing cache entry for this file.
    /// Returns error if serialization or database write fails.
    pub fn set(&self, fname: &str, mtime: SystemTime, tags: &[Tag]) -> Result<()> {
        let entry = CacheEntry::new(mtime, tags.to_vec())?;
        let bytes = entry.to_bytes()?;

        // Write transaction - ACID guarantees from redb
        let write_txn = self.db.begin_write()
            .context("Failed to begin write transaction")?;

        {
            let mut table = write_txn.open_table(TAGS_TABLE)
                .context("Failed to open tags table")?;

            table.insert(fname, bytes.as_slice())
                .with_context(|| format!("Failed to insert cache entry for {}", fname))?;
        }

        // Commit transaction - durability guaranteed
        write_txn.commit()
            .context("Failed to commit cache write")?;

        Ok(())
    }

    /// Clear all cached data.
    ///
    /// Removes all entries from the cache database.
    /// Does not delete the database file itself.
    pub fn clear(&self) -> Result<()> {
        let write_txn = self.db.begin_write()
            .context("Failed to begin write transaction for clear")?;

        {
            let mut table = write_txn.open_table(TAGS_TABLE)
                .context("Failed to open tags table")?;

            // Drain iterator removes all entries
            let keys: Vec<String> = table.iter()
                .ok()
                .into_iter()
                .flatten()
                .filter_map(|r| r.ok())
                .map(|(k, _)| k.value().to_string())
                .collect();

            for key in keys {
                table.remove(key.as_str())
                    .context("Failed to remove cache entry during clear")?;
            }
        }

        write_txn.commit()
            .context("Failed to commit cache clear")?;

        Ok(())
    }

    /// Get cache statistics for monitoring and debugging.
    ///
    /// Returns number of cached files and approximate database size.
    pub fn stats(&self) -> CacheStats {
        let read_txn = match self.db.begin_read() {
            Ok(txn) => txn,
            Err(_) => return CacheStats::default(),
        };

        let table = match read_txn.open_table(TAGS_TABLE) {
            Ok(t) => t,
            Err(_) => return CacheStats::default(),
        };

        let entries = table.len().unwrap_or(0) as usize;

        // Approximate size by iterating entries and summing value lengths
        let size_bytes = table.iter()
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|r| r.ok())
            .map(|(k, v)| k.value().len() + v.value().len())
            .sum::<usize>() as u64;

        CacheStats { entries, size_bytes }
    }
}

/// Cache statistics for monitoring.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of files in cache
    pub entries: usize,
    /// Approximate total size in bytes (keys + values)
    pub size_bytes: u64,
}

impl CacheStats {
    /// Format size in human-readable form (KB, MB, GB)
    pub fn size_human(&self) -> String {
        const KB: u64 = 1024;
        const MB: u64 = KB * 1024;
        const GB: u64 = MB * 1024;

        if self.size_bytes >= GB {
            format!("{:.2} GB", self.size_bytes as f64 / GB as f64)
        } else if self.size_bytes >= MB {
            format!("{:.2} MB", self.size_bytes as f64 / MB as f64)
        } else if self.size_bytes >= KB {
            format!("{:.2} KB", self.size_bytes as f64 / KB as f64)
        } else {
            format!("{} B", self.size_bytes)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{TagKind, Tag};

    fn make_test_tag(name: &str) -> Tag {
        Tag {
            rel_fname: "test.rs".into(),
            fname: "/tmp/test.rs".into(),
            line: 1,
            name: name.into(),
            kind: TagKind::Def,
            node_type: "function".into(),
            parent_name: None,
            parent_line: None,
            signature: None,
            fields: None,
        metadata: None,
        }
    }

    #[test]
    fn test_cache_entry_mtime_validation() {
        let now = SystemTime::now();
        let tags = vec![make_test_tag("foo")];

        let entry = CacheEntry::new(now, tags.clone()).unwrap();

        // Same mtime should validate
        assert!(entry.is_valid(now));

        // Different mtime should not validate
        let later = now + std::time::Duration::from_secs(1);
        assert!(!entry.is_valid(later));
    }

    #[test]
    fn test_cache_entry_serialization() {
        let now = SystemTime::now();
        let tags = vec![
            make_test_tag("foo"),
            make_test_tag("bar"),
        ];

        let entry = CacheEntry::new(now, tags.clone()).unwrap();
        let bytes = entry.to_bytes().unwrap();
        let decoded = CacheEntry::from_bytes(&bytes).unwrap();

        assert_eq!(entry.mtime_secs, decoded.mtime_secs);
        assert_eq!(entry.mtime_nanos, decoded.mtime_nanos);
        assert_eq!(entry.tags.len(), decoded.tags.len());
    }

    #[test]
    fn test_cache_roundtrip() -> Result<()> {
        let temp_dir = std::env::temp_dir().join("ripmap_test_cache");
        let _ = fs::remove_dir_all(&temp_dir); // Clean up from previous runs
        fs::create_dir_all(&temp_dir)?;

        let cache = TagCache::open(&temp_dir)?;
        let now = SystemTime::now();
        let tags = vec![make_test_tag("test_fn")];

        // Store
        cache.set("test.rs", now, &tags)?;

        // Retrieve with matching mtime
        let retrieved = cache.get("test.rs", now);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().len(), 1);

        // Retrieve with different mtime
        let later = now + std::time::Duration::from_secs(1);
        let retrieved = cache.get("test.rs", later);
        assert!(retrieved.is_none());

        // Clean up
        fs::remove_dir_all(&temp_dir)?;
        Ok(())
    }

    #[test]
    fn test_cache_clear() -> Result<()> {
        let temp_dir = std::env::temp_dir().join("ripmap_test_cache_clear");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir)?;

        let cache = TagCache::open(&temp_dir)?;
        let now = SystemTime::now();

        // Add multiple entries
        cache.set("file1.rs", now, &[make_test_tag("fn1")])?;
        cache.set("file2.rs", now, &[make_test_tag("fn2")])?;

        let stats = cache.stats();
        assert_eq!(stats.entries, 2);

        // Clear
        cache.clear()?;

        let stats = cache.stats();
        assert_eq!(stats.entries, 0);

        // Clean up
        fs::remove_dir_all(&temp_dir)?;
        Ok(())
    }

    #[test]
    fn test_cache_stats() -> Result<()> {
        let temp_dir = std::env::temp_dir().join("ripmap_test_cache_stats");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir)?;

        let cache = TagCache::open(&temp_dir)?;
        let now = SystemTime::now();

        // Empty cache
        let stats = cache.stats();
        assert_eq!(stats.entries, 0);
        assert_eq!(stats.size_bytes, 0);

        // Add entry
        cache.set("test.rs", now, &[make_test_tag("foo")])?;

        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert!(stats.size_bytes > 0);

        // Human-readable size
        let size_str = stats.size_human();
        assert!(size_str.contains("B")); // Should contain "B" for bytes

        // Clean up
        fs::remove_dir_all(&temp_dir)?;
        Ok(())
    }
}
