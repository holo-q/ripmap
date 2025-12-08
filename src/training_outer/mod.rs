//! L2 Outer Loop: Promptgram Optimization
//!
//! The outer loop sits above the inner reasoning-based optimizer (L1) and evolves
//! the *prompts themselves* rather than just the hyperparameters.
//!
//! ## The 4-Level Stack
//!
//! - **L0 - Environment**: The code ranker, 17 params, NDCG metric
//! - **L1 - Inner Optimizer**: LLM that proposes param changes based on failures
//! - **L2 - Promptgram Optimizer**: This module - evolves the prompts controlling L1
//! - **L3 - Self-hosted Recursion**: L2 eventually learns to edit itself
//!
//! ## Core Concepts
//!
//! - **Promptgram**: A structured prompt treated as a program with sections
//!   (Role, Policy, Heuristics, Output schema, etc.)
//! - **OuterEpisode**: One run of L1 under a candidate promptgram
//! - **Meta-levers**: Latent axes describing optimization behavior
//!   (structural_trust, temporal_horizon, exploration_bias, etc.)
//!
//! ## Directory Structure
//!
//! ```text
//! training-outer/
//!   runs/
//!     0001_outer_run/
//!       config.toml
//!       population/          # Candidate promptgrams
//!       eval/                # Inner run results
//!       outer_scratchpad.md  # Meta-strategy notebook
//!   prompts/
//!     inner/                 # L1 promptgrams
//!     outer/                 # L2 meta-promptgrams
//! ```

pub mod schemas;
pub mod promptgram;
pub mod mesa;

pub use schemas::*;
pub use promptgram::*;
pub use mesa::*;
