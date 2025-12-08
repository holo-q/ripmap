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

pub mod types;
pub mod extraction;
pub mod discovery;
pub mod ranking;
pub mod rendering;
pub mod cache;
pub mod mcp;
pub mod training;
pub mod callgraph;

// Re-export core types
pub use types::{
    Tag, TagKind, RankedTag, SignatureInfo, FieldInfo,
    DetailLevel, RankingConfig, FilePhase, Intent, SymbolId,
};

// Re-export call graph types
pub use callgraph::{
    CallGraph, CallEdge, FunctionId,
    CallResolver, ResolverBuilder, ResolverConfig, ResolutionStats,
    ResolutionStrategy, ResolutionContext, Candidate,
    SameFileStrategy, NameMatchStrategy, TypeHintStrategy, ImportStrategy,
};
