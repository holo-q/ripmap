//! Tag extraction from source code using tree-sitter.
//!
//! This module handles:
//! - Loading tree-sitter parsers and queries
//! - Parsing source files into ASTs
//! - Running .scm queries to extract symbol tags
//! - Walking AST for parent scopes
//! - Extracting function signatures and class fields
//!
//! # Parser Selection
//!
//! The module provides two parsing strategies:
//! - `TreeSitterParser`: Full AST parsing with .scm queries (preferred)
//! - `Parser`: Regex-based fallback for unsupported languages

mod fields;
mod parser;
mod signatures;
mod tags;
mod treesitter;

pub use parser::Parser;
pub use tags::extract_tags;
pub use treesitter::{TreeSitterParser, extension_to_language};
