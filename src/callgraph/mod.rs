//! Call graph construction with pluggable resolution strategies.
//!
//! # Bicameral Architecture
//!
//! The call graph system uses a two-phase pipeline to achieve 80%+ resolution:
//!
//! ```text
//! Tags → [Shadow Pass] → PageRank → [Policy] → LSP → [Final Pass] → CallGraph
//!             │                                           │
//!    ┌────────┴────────┐                      ┌───────────┴───────────┐
//!    │ Shadow Resolver │                      │    Final Resolver     │
//!    │ (Recall-opt)    │                      │    (Precision-opt)    │
//!    │ NameMatch=0.9   │                      │    NameMatch=0.2      │
//!    │ No LSP          │                      │    LSP=1.5            │
//!    └─────────────────┘                      └───────────────────────┘
//! ```
//!
//! The bicameral design separates concerns:
//! - **Shadow Pass**: Aggressive heuristics seed PageRank (recall-optimized)
//! - **Policy Layer**: Uses PageRank to determine which edges need LSP verification
//! - **Final Pass**: Conservative heuristics + LSP for precision (precision-optimized)
//!
//! This achieves 80%+ resolution while minimizing LSP overhead.
//!
//! # Quick Start
//!
//! ```ignore
//! // Full pipeline with LSP (recommended)
//! let coords = PipelineCoordinates::default();
//! let pipeline = Pipeline::new(coords).with_lsp(lsp_client);
//! let (graph, stats) = pipeline.build_graph(&tags);
//!
//! // Heuristic-only baseline (14% resolution)
//! let resolver = CallResolver::new()
//!     .with_strategy(Box::new(SameFileStrategy::new()))
//!     .with_strategy(Box::new(NameMatchStrategy::new()));
//! let graph = resolver.build_graph(&tags);
//! ```
//!
//! # Modular Components
//!
//! - Core graph structure is strategy-agnostic
//! - Resolution strategies are plug-and-play
//! - Each signal (types, imports, names) is independent
//! - Strategies can be combined with confidence weighting
//! - LSP integration is optional but dramatically improves precision

mod coordinates;
mod graph;
mod lsp_strategy;
mod pipeline;
mod resolver;
mod strategies;

pub use coordinates::{PipelineCoordinates, StrategyCoordinates};
pub use graph::{CallEdge, CallGraph, FunctionId};
pub use lsp_strategy::{LspStrategy, LspTypeCache};
pub use pipeline::{Pipeline, PipelineStats};
pub use resolver::{CallResolver, ResolutionStats, ResolverBuilder, ResolverConfig};
pub use strategies::{
    Candidate, ImportStrategy, NameMatchStrategy, ResolutionContext, ResolutionStrategy,
    SameFileStrategy, TypeHintStrategy,
};
