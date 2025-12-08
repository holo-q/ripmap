//! Git-aware file discovery.
//!
//! Uses the `ignore` crate to respect .gitignore and walk directories
//! efficiently. Parallel traversal with rayon.

mod files;

pub use files::find_source_files;
