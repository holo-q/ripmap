//! ripmap - Ultra-fast codebase cartography
//!
//! A Rust rewrite of grepmap, targeting 1000x faster performance.
//! Uses tree-sitter for parsing, PageRank for importance ranking,
//! and rich terminal rendering for output.
//!
//! # Architecture
//!
//! ```text
//! File Discovery → Tag Extraction → Graph Building → PageRank → Boosts → Rendering
//!       ↓              ↓                ↓              ↓          ↓          ↓
//!    ignore        tree-sitter      petgraph      iterative   contextual   ANSI
//!    crate          + .scm          DiGraph        power       signals     colors
//! ```
//!
//! # Performance Strategies
//!
//! - Parallel file parsing via rayon
//! - Memory-mapped I/O for large files
//! - Arena allocators for tag batches
//! - Lock-free graph building with dashmap
//! - String interning for symbol names
//! - Persistent cache with redb

pub mod cache;
pub mod callgraph;
pub mod config;
pub mod discovery;
pub mod extraction;
pub mod lsp;
pub mod mcp;
pub mod ranking;
pub mod rendering;
pub mod training;
pub mod training_outer;
pub mod types;

// Re-export core types
pub use types::{
    DetailLevel, FieldInfo, FilePhase, Intent, RankedTag, RankingConfig, SignatureInfo, SymbolId,
    Tag, TagKind,
};

// Re-export call graph types
pub use callgraph::{
    CallEdge, CallGraph, CallResolver, Candidate, FunctionId, ImportStrategy, NameMatchStrategy,
    ResolutionContext, ResolutionStats, ResolutionStrategy, ResolverBuilder, ResolverConfig,
    SameFileStrategy, TypeHintStrategy,
};

// Re-export LSP types
pub use lsp::{LspClient, LspPolicyCoordinates, PolicyEngine};
