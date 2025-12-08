//! Call graph construction with pluggable resolution strategies.
//!
//! The call graph system is fully modular:
//! - Core graph structure is strategy-agnostic
//! - Resolution strategies are plug-and-play
//! - Each signal (types, imports, names) is independent
//! - Strategies can be combined with confidence weighting
//!
//! # Architecture
//!
//! ```text
//! Tags → [Resolvers] → CallGraph → PageRank
//!          ↓
//!    ┌─────┴─────┐
//!    │ Strategies │
//!    ├───────────┤
//!    │ NameMatch │  ← Always available
//!    │ TypeHints │  ← Python, TypeScript
//!    │ Imports   │  ← Cross-file resolution
//!    │ SameFile  │  ← Highest confidence
//!    └───────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! let mut resolver = CallResolver::new();
//! resolver.add_strategy(Box::new(SameFileStrategy::new()));
//! resolver.add_strategy(Box::new(TypeHintStrategy::new()));
//! resolver.add_strategy(Box::new(NameMatchStrategy::new()));
//!
//! let graph = resolver.build_graph(&tags);
//! ```

mod graph;
mod resolver;
mod strategies;

pub use graph::{CallGraph, CallEdge, FunctionId};
pub use resolver::{CallResolver, ResolverBuilder, ResolverConfig, ResolutionStats};
pub use strategies::{
    ResolutionStrategy,
    ResolutionContext,
    Candidate,
    SameFileStrategy,
    NameMatchStrategy,
    TypeHintStrategy,
    ImportStrategy,
};
