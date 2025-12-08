//! Ranking pipeline - from tags to importance scores.
//!
//! The ranking system combines:
//! - PageRank on the symbol/file reference graph
//! - Contextual boosts (chat files, mentions, temporal coupling)
//! - Git-based weighting (recency, churn)
//! - Focus expansion via graph traversal
//! - Intent-driven recipe selection

mod pagerank;
mod symbols;
mod boosts;
mod focus;
mod git;
mod bridges;
mod intent;
mod coupling;

pub use pagerank::PageRanker;
pub use symbols::SymbolRanker;
pub use boosts::BoostCalculator;
pub use focus::FocusResolver;
pub use git::{GitWeightCalculator, FileStats};
pub use bridges::BridgeDetector;
pub use intent::IntentClassifier;
pub use coupling::TestCouplingDetector;
