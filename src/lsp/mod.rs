//! LSP-Native Type Resolution for ripmap
//!
//! This module implements the **regime shift** from 14% heuristic resolution
//! to 80%+ LSP-powered resolution. It represents a fundamental phase transition
//! in ripmap's architecture: from passive symbol matching to active sensing.
//!
//! # The Vision
//!
//! Instead of guessing at type relationships through name matching and import
//! analysis, we strategically "buy" truth with latency by querying `ty server`
//! (Python LSP type checker). This isn't just an incremental improvement - it's
//! a regime change that requires rethinking all downstream hyperparameters.
//!
//! # Architecture Overview
//!
//! ```text
//! PolicyEngine → Select high-value query sites (wavefront selection)
//!      ↓
//! LspClient → Execute batch queries to ty server (JSON-RPC)
//!      ↓
//! Type Cache → Store results, track coherence (shared receiver counts)
//!      ↓
//! CallGraph → Integrate resolved types into graph building
//! ```
//!
//! # Dissolved Decision Trees Philosophy
//!
//! Every feature enters as **continuous coordinates**, not categorical modes.
//! The codebase becomes a parameterized family of algorithms where training
//! selects which member to instantiate through coordinate values, not code paths.
//!
//! Instead of:
//! ```ignore
//! if strategy == "greedy" { use_greedy(); }
//! else if strategy == "exploratory" { use_exploratory(); }
//! ```
//!
//! We use:
//! ```ignore
//! let score = w_centrality * x + w_uncertainty * y + w_coherence * z;
//! let action = softmax(scores, temperature);
//! // w_*, temperature are trainable coordinates
//! ```
//!
//! # Wavefront Execution Model
//!
//! Queries are executed in 1-3 generational waves, controlled by the
//! `batch_latency_bias` coordinate:
//!
//! - **Gen 1 (The Spine)**: High-centrality, high-causality roots (variables,
//!   imports, definitions). Type cache populates, many downstream edges resolve
//!   automatically.
//!
//! - **Gen 2 (The Frontier)**: Re-score remaining entropy with new type info.
//!   Query newly exposed ambiguity that Gen 1 couldn't resolve.
//!
//! - **Gen 3 (The Fill)**: Surgical strikes on specific high-rank nodes.
//!   Stop when marginal utility floor is reached.
//!
//! # Training Protocol: Oracle Bootstrap
//!
//! The shift from 14% to 80% resolution requires training in the correct regime:
//!
//! 1. **Phase 1 - Oracle Run**: Run `ty` on entire corpus, build Perfect Graph
//!    (100% resolution), train DOWNSTREAM parameters on this graph.
//!
//! 2. **Phase 2 - Policy Distillation**: Freeze those "Golden Weights", train
//!    LSP policy coordinates to minimize KL(Rank_policy || Rank_oracle) subject
//!    to latency constraints.
//!
//! 3. **Phase 3 - Joint Fine-Tuning**: Unfreeze everything, end-to-end polish.
//!
//! # Integration with Existing Training
//!
//! LSP coordinates become part of the L1 hyperparameter grid alongside existing
//! parameters like `pagerank_alpha` and `boost_caller_weight`. L2 meta-learning
//! can discover optimal policy structures by tuning `interaction_mixing` to
//! find whether additive or multiplicative combination works better.
//!
//! # Graceful Degradation
//!
//! When `ty` is unavailable, the system falls back to 14% heuristic-only
//! resolution. The `marginal_utility_floor` naturally handles this: if LSP
//! returns nothing, expected gain is 0, no queries are made.

mod client;
mod coordinates;
mod policy;

pub use client::{LspClient, MockClient, TypeInfo, TypeResolver};
pub use coordinates::LspPolicyCoordinates;
pub use policy::{PolicyEngine, QueryCandidate};
