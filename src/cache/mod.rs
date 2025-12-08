//! Persistent caching with redb.
//!
//! Caches parsed tags per file, keyed by (path, mtime).
//! Enables 100x+ speedup on warm runs.

mod store;

pub use store::{TagCache, CacheStats};
