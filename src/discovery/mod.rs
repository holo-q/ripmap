//! Git-aware file discovery.
//!
//! Uses the `ignore` crate to respect .gitignore and walk directories
//! efficiently. Parallel traversal with rayon.
//!
//! Supports configuration from pyproject.toml or ripmap.toml for
//! include/exclude patterns.

mod files;

pub use files::{find_source_files, find_source_files_with_config};
