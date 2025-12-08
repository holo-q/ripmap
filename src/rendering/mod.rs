//! Output rendering - from ranked tags to terminal/text output.
//!
//! Supports multiple rendering modes:
//! - Directory mode: hierarchical symbol overview with badges
//! - Tree mode: code snippets with syntax highlighting
//!
//! Optimizes for token budget via binary search.

mod directory;
mod tree;
mod colors;

pub use directory::DirectoryRenderer;
pub use tree::TreeRenderer;
pub use colors::{colorize, Badge};
