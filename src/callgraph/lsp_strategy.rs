//! LSP-backed resolution strategy using pre-populated type cache.
//!
//! This strategy resolves method calls by querying an LSP-derived type cache.
//! The cache is populated offline by running LSP hover queries on all call sites,
//! then stored for fast lookup during call graph construction.
//!
//! # Resolution Logic
//!
//! For a call like `user.save()` at line 42:
//! 1. Extract receiver name from call metadata: "user"
//! 2. Look up receiver's type in cache at call site location (file, line, col)
//! 3. If found (e.g., "User"), find all method definitions matching:
//!    - method name = "save"
//!    - parent class = "User"
//! 4. Return candidate with LSP confidence (~0.95)
//!
//! # Cache Structure
//!
//! The LspTypeCache wraps a HashMap:
//! - Key: (file_path, line, column) - 0-based LSP coordinates
//! - Value: TypeInfo { type_str, confidence }
//!
//! The cache is pre-populated by running LSP hover queries offline and
//! serializing the results. This amortizes LSP cost across multiple runs.
//!
//! # Confidence
//!
//! LSP-resolved types have much higher confidence than heuristics:
//! - LSP (default): 0.95 - semantic analysis with full type inference
//! - Type hints: 0.85 - syntactic annotations (may be incomplete)
//! - Same file: 0.9 - structural proximity
//! - Name match: 0.5 - fallback heuristic
//!
//! This confidence premium creates the regime shift: when LSP resolves
//! a call, it dominates all other signals and dramatically improves
//! downstream PageRank accuracy.

use super::graph::FunctionId;
use super::strategies::{Candidate, ResolutionContext, ResolutionStrategy};
use crate::lsp::TypeInfo;
use crate::types::Tag;
use std::collections::HashMap;
use std::sync::Arc;

/// Type cache keyed by call site location.
///
/// The cache is populated offline by running LSP hover queries on all
/// receivers in call expressions, then serialized to disk. During call
/// graph construction, we load the cache and use it for instant lookups.
///
/// # Coordinate System
///
/// Keys use 0-based coordinates (LSP standard):
/// - Editor line 42, col 10 → cache key (file, 41, 9)
///
/// When querying from ripmap tags (1-based), convert before lookup:
/// - tag.line = 42 (1-based) → cache_key.1 = 41 (0-based)
/// - tag.col = 10 (1-based) → cache_key.2 = 9 (0-based)
///
/// # Thread Safety
///
/// The cache is wrapped in Arc for sharing across strategies. The inner
/// HashMap is immutable after construction (read-only during resolution).
#[derive(Debug, Clone)]
pub struct LspTypeCache {
    /// Cache mapping (file, line, col) → TypeInfo
    /// Coordinates are 0-based (LSP standard)
    cache: HashMap<(Arc<str>, u32, u32), TypeInfo>,
}

impl LspTypeCache {
    /// Create empty cache
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Create cache from HashMap
    ///
    /// Expected coordinate system: 0-based (LSP standard)
    pub fn from_map(cache: HashMap<(Arc<str>, u32, u32), TypeInfo>) -> Self {
        Self { cache }
    }

    /// Look up type information at a specific location
    ///
    /// CRITICAL: Input coordinates must be 0-based (LSP standard).
    /// If calling from ripmap tags (1-based), convert before calling:
    ///
    /// ```rust
    /// let type_info = cache.get(
    ///     &tag.rel_fname,
    ///     tag.line.saturating_sub(1),  // Convert to 0-based
    ///     tag.col.saturating_sub(1)    // Convert to 0-based
    /// );
    /// ```
    pub fn get(&self, file: &str, line: u32, col: u32) -> Option<&TypeInfo> {
        // Keys are stored as 0-based coordinates
        self.cache.get(&(Arc::from(file), line, col))
    }

    /// Insert type information (for building cache)
    ///
    /// CRITICAL: Coordinates must be 0-based (LSP standard)
    pub fn insert(&mut self, file: Arc<str>, line: u32, col: u32, type_info: TypeInfo) {
        self.cache.insert((file, line, col), type_info);
    }

    /// Get cache size (for debugging/stats)
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Create cache from LSP batch query results.
    ///
    /// Converts the HashMap returned by TypeResolver::resolve_batch into an LspTypeCache.
    /// This is used in the pipeline's Phase 3 to convert LSP query results into a cache
    /// that LspStrategy can use during final pass resolution.
    ///
    /// # Arguments
    ///
    /// * `results` - HashMap from (file, line, col) to TypeInfo, using 0-based coordinates
    ///
    /// # Returns
    ///
    /// LspTypeCache populated with the query results.
    ///
    /// # Example
    ///
    /// ```rust
    /// let results = resolver.resolve_batch(&queries);
    /// let type_cache = LspTypeCache::from_lsp_results(&results);
    /// ```
    pub fn from_lsp_results(results: &HashMap<(String, u32, u32), TypeInfo>) -> Self {
        let mut cache = Self::new();
        for ((file, line, col), type_info) in results {
            cache.insert(Arc::from(file.as_str()), *line, *col, type_info.clone());
        }
        cache
    }
}

impl Default for LspTypeCache {
    fn default() -> Self {
        Self::new()
    }
}

/// LSP resolution strategy using pre-populated type cache.
///
/// This strategy bridges the gap between heuristics and real-time LSP:
/// 1. Offline: Run LSP hover queries on all receivers, build cache
/// 2. Runtime: Instant lookups from cache (no LSP overhead)
/// 3. Resolution: Same logic as live LSP, but with cached types
///
/// # Confidence Injection
///
/// The base confidence can be customized at runtime:
///
/// ```rust
/// // Default: 0.95 (matching live LSP)
/// let strategy = LspStrategy::new(cache);
///
/// // Custom: higher confidence for high-quality LSP servers
/// let strategy = LspStrategy::with_base_confidence(cache, 0.98);
///
/// // Custom: lower confidence for incomplete caches
/// let strategy = LspStrategy::with_base_confidence(cache, 0.85);
/// ```
///
/// This allows domain-specific tuning without recompiling.
pub struct LspStrategy {
    /// Pre-populated type cache
    cache: Arc<LspTypeCache>,

    /// Base confidence for LSP-resolved types. Default: 0.95
    ///
    /// This is the "regime shift" parameter. LSP confidence is much higher
    /// than heuristics (0.5-0.85), creating a strong signal that dominates
    /// other strategies and improves PageRank accuracy.
    pub base_confidence: f64,
}

impl LspStrategy {
    /// Create new LSP strategy with default confidence (0.95)
    pub fn new(cache: Arc<LspTypeCache>) -> Self {
        Self {
            cache,
            base_confidence: 0.95,
        }
    }

    /// Create LSP strategy with custom base confidence
    ///
    /// Allows runtime tuning of confidence levels without recompilation.
    /// Use cases:
    /// - Higher (0.98): High-quality LSP server with excellent type inference
    /// - Lower (0.85): Incomplete cache or unreliable LSP
    /// - Default (0.95): Standard LSP confidence
    pub fn with_base_confidence(cache: Arc<LspTypeCache>, confidence: f64) -> Self {
        Self {
            cache,
            base_confidence: confidence,
        }
    }
}

impl ResolutionStrategy for LspStrategy {
    fn name(&self) -> &'static str {
        "lsp"
    }

    fn supports_language(&self, lang: &str) -> bool {
        // LSP strategy only works for languages with LSP type inference
        // Currently: Python, TypeScript/TSX
        matches!(lang, "python" | "typescript" | "tsx")
    }

    fn resolve(&self, call: &Tag, context: &ResolutionContext) -> Vec<Candidate> {
        // Only applies to method calls with a receiver
        // e.g., user.save() has receiver="user"
        let receiver = call
            .metadata
            .as_ref()
            .and_then(|m| m.get("receiver"))
            .map(|s| s.as_str());

        let Some(receiver) = receiver else {
            // Not a method call, can't use LSP type resolution
            return vec![];
        };

        // Look up receiver's type in cache using call site location
        // CRITICAL: Tags use 1-based coordinates, cache uses 0-based
        let receiver_type = self.cache.get(
            call.rel_fname.as_ref(),
            call.line.saturating_sub(1), // Convert to 0-based
            // We need column from metadata or estimate
            // For now, assume receiver starts at column 0 of the line
            // TODO: Extract precise column from metadata when available
            0,
        );

        let Some(receiver_type) = receiver_type else {
            // Cache miss - graceful degradation (fallback to other strategies)
            return vec![];
        };

        // Find method definitions matching the resolved type
        let defs = context.find_definitions(&call.name);

        defs.into_iter()
            .filter(|def| {
                // Check if this definition is a method of the receiver's type
                // e.g., receiver_type = "User", def.parent_name = "User"
                def.parent_name
                    .as_ref()
                    .map_or(false, |p| p.as_ref() == receiver_type.type_str)
            })
            .map(|def| Candidate {
                target: FunctionId::new(def.rel_fname.clone(), def.name.clone(), def.line)
                    .with_parent(receiver_type.type_str.as_str()),
                confidence: self.base_confidence,
                type_hint: Some(format!("{}: {}", receiver, receiver_type.type_str)),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TagKind;
    use std::collections::HashMap;

    fn make_def(file: &str, name: &str, line: u32, parent: Option<&str>) -> Tag {
        Tag {
            rel_fname: Arc::from(file),
            fname: Arc::from(file),
            line,
            name: Arc::from(name),
            kind: TagKind::Def,
            node_type: Arc::from("function"),
            parent_name: parent.map(Arc::from),
            parent_line: None,
            signature: None,
            fields: None,
            metadata: None,
        }
    }

    fn make_call(file: &str, name: &str, line: u32, receiver: Option<&str>) -> Tag {
        let metadata = receiver.map(|r| {
            let mut m = HashMap::new();
            m.insert("receiver".to_string(), r.to_string());
            m
        });

        Tag {
            rel_fname: Arc::from(file),
            fname: Arc::from(file),
            line,
            name: Arc::from(name),
            kind: TagKind::Ref,
            node_type: Arc::from("call"),
            parent_name: None,
            parent_line: None,
            signature: None,
            fields: None,
            metadata,
        }
    }

    #[test]
    fn test_cache_hit_returns_candidate() {
        // Build cache with type info at line 42 (1-based in tag, 41 0-based in cache)
        let mut cache = LspTypeCache::new();
        cache.insert(
            Arc::from("test.py"),
            41, // 0-based: corresponds to tag line 42
            0,  // Assuming receiver at start of line
            TypeInfo {
                type_str: "User".to_string(),
                confidence: 0.95,
            },
        );

        let strategy = LspStrategy::new(Arc::new(cache));

        // Set up context with User.save definition
        let tags = vec![
            make_def("models.py", "save", 10, Some("User")),
            make_call("test.py", "save", 42, Some("user")),
        ];
        let context = ResolutionContext::new(&tags);

        let call = &tags[1];
        let candidates = strategy.resolve(call, &context);

        // Should resolve to User.save
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].target.name.as_ref(), "save");
        assert_eq!(
            candidates[0].target.parent.as_ref().map(|s| s.as_ref()),
            Some("User")
        );
        assert_eq!(candidates[0].confidence, 0.95);
        assert_eq!(candidates[0].type_hint.as_deref(), Some("user: User"));
    }

    #[test]
    fn test_cache_miss_returns_empty() {
        // Empty cache
        let cache = LspTypeCache::new();
        let strategy = LspStrategy::new(Arc::new(cache));

        // Set up context
        let tags = vec![
            make_def("models.py", "save", 10, Some("User")),
            make_call("test.py", "save", 42, Some("user")),
        ];
        let context = ResolutionContext::new(&tags);

        let call = &tags[1];
        let candidates = strategy.resolve(call, &context);

        // Cache miss → empty vec (graceful degradation)
        assert_eq!(candidates.len(), 0);
    }

    #[test]
    fn test_custom_confidence() {
        // Build cache
        let mut cache = LspTypeCache::new();
        cache.insert(
            Arc::from("test.py"),
            41, // 0-based
            0,
            TypeInfo {
                type_str: "User".to_string(),
                confidence: 0.95, // Cache stores 0.95, but strategy overrides
            },
        );

        // Custom confidence: 0.88
        let strategy = LspStrategy::with_base_confidence(Arc::new(cache), 0.88);

        let tags = vec![
            make_def("models.py", "save", 10, Some("User")),
            make_call("test.py", "save", 42, Some("user")),
        ];
        let context = ResolutionContext::new(&tags);

        let call = &tags[1];
        let candidates = strategy.resolve(call, &context);

        assert_eq!(candidates.len(), 1);
        // Confidence should be 0.88 (strategy override), not 0.95 (cache value)
        assert_eq!(candidates[0].confidence, 0.88);
    }

    #[test]
    fn test_non_method_call_returns_empty() {
        // Build cache
        let mut cache = LspTypeCache::new();
        cache.insert(
            Arc::from("test.py"),
            41,
            0,
            TypeInfo {
                type_str: "User".to_string(),
                confidence: 0.95,
            },
        );

        let strategy = LspStrategy::new(Arc::new(cache));

        // Regular function call (no receiver)
        let tags = vec![
            make_def("utils.py", "helper", 5, None),
            make_call("test.py", "helper", 42, None), // No receiver
        ];
        let context = ResolutionContext::new(&tags);

        let call = &tags[1];
        let candidates = strategy.resolve(call, &context);

        // No receiver → LSP strategy doesn't apply
        assert_eq!(candidates.len(), 0);
    }

    #[test]
    fn test_supports_language() {
        let cache = LspTypeCache::new();
        let strategy = LspStrategy::new(Arc::new(cache));

        // Supported languages
        assert!(strategy.supports_language("python"));
        assert!(strategy.supports_language("typescript"));
        assert!(strategy.supports_language("tsx"));

        // Unsupported languages
        assert!(!strategy.supports_language("rust"));
        assert!(!strategy.supports_language("go"));
        assert!(!strategy.supports_language("java"));
    }

    #[test]
    fn test_multiple_candidates() {
        // Build cache
        let mut cache = LspTypeCache::new();
        cache.insert(
            Arc::from("test.py"),
            41,
            0,
            TypeInfo {
                type_str: "User".to_string(),
                confidence: 0.95,
            },
        );

        let strategy = LspStrategy::new(Arc::new(cache));

        // Multiple User.save definitions (e.g., across inheritance hierarchy)
        let tags = vec![
            make_def("models.py", "save", 10, Some("User")),
            make_def("models.py", "save", 50, Some("User")), // Overload or duplicate
            make_def("admin.py", "save", 20, Some("Admin")), // Different class
            make_call("test.py", "save", 42, Some("user")),
        ];
        let context = ResolutionContext::new(&tags);

        let call = &tags[3];
        let candidates = strategy.resolve(call, &context);

        // Should find both User.save definitions (lines 10, 50), but not Admin.save
        assert_eq!(candidates.len(), 2);
        for candidate in &candidates {
            assert_eq!(candidate.target.name.as_ref(), "save");
            assert_eq!(
                candidate.target.parent.as_ref().map(|s| s.as_ref()),
                Some("User")
            );
        }
    }

    #[test]
    fn test_cache_api() {
        let mut cache = LspTypeCache::new();

        // Empty cache
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        // Insert
        cache.insert(
            Arc::from("test.py"),
            41,
            9,
            TypeInfo {
                type_str: "User".to_string(),
                confidence: 0.95,
            },
        );

        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);

        // Get
        let result = cache.get("test.py", 41, 9);
        assert!(result.is_some());
        assert_eq!(result.unwrap().type_str, "User");

        // Miss
        let miss = cache.get("test.py", 42, 10);
        assert!(miss.is_none());
    }
}
