//! Hyperparameter training infrastructure.
//!
//! This module enables scientific training of ripmap's ranking quality by:
//! 1. Extracting ground truth from git history (retrocausal oracle)
//! 2. Defining quality metrics (NDCG, MRR, precision@k)
//! 3. Exploring parameter space (grid, LHS, Bayesian)
//! 4. Sensitivity analysis (which parameters actually matter?)
//! 5. **Reasoning-based training via Claude as universal function approximator**
//!
//! ## The Retrocausal Insight
//!
//! Git history records developer attention. Every commit is a trace of
//! "these files were cognitively connected in this moment." We use this
//! as ground truth: given file A as focus, files B,C,D from the same
//! commit SHOULD rank high.
//!
//! ## Commit Quality Weighting
//!
//! Not all commits provide equal signal:
//! - **Bugfixes** (2-6 files): GOLD - causal relationship to symptom
//! - **Features** (3-8 files): strong semantic coupling
//! - **Refactors** (10+ files): weaker signal, mechanical changes
//! - **WIP/save**: noise, skip entirely
//!
//! ## Reasoning-Based Training
//!
//! The paradigm shift from classical optimization:
//! - Classical: observe Loss(θ) → infer ∂Loss/∂θ → step θ (WHY is lost?)
//! - Reasoning: observe Failure(θ) → reason about WHY → propose Δθ OR Δstructure
//!
//! Claude acts as a universal function approximator, understanding *why*
//! rankings fail and proposing semantically-informed adjustments. The
//! sidechain scratchpad accumulates insights into operator wisdom.
//!
//! ## Usage
//!
//! ```bash
//! # Classical training
//! ripmap-train --curated --strategy bayesian --budget 500
//!
//! # Reasoning-based training
//! ripmap-train --curated --reason --episodes 20
//!
//! # Distill accumulated wisdom
//! ripmap-train --distill --scratchpad scratchpad.json
//! ```

pub mod git_oracle;
pub mod gridsearch;
pub mod metrics;
pub mod plots;
pub mod reasoning;
pub mod repos;
pub mod sensitivity;

pub use git_oracle::{
    GitCase, WeightedCase, compute_coupling_weights, extract_cases, weight_cases,
};
pub use gridsearch::{
    ParameterGrid, ParameterPoint, SearchStrategy, bayesian_next_sample, sample_points,
};
pub use metrics::{CaseMetrics, EvalMetrics, mean_reciprocal_rank, precision_at_k, weighted_ndcg};
pub use plots::LiveProgress;
pub use reasoning::{
    Agent, RankingFailure, ReasoningEpisode, Scratchpad, apply_changes, call_agent, call_claude,
    call_gemini, distill_scratchpad, print_scratchpad_summary, reason_about_failures,
    update_scratchpad,
};
pub use repos::{CURATED_REPOS, RepoSpec, quick_repos};
pub use sensitivity::{SensitivityAnalysis, ablation_study, full_analysis, print_summary};
