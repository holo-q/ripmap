//! LSP Policy Coordinates - The Continuous Parameter Space
//!
//! This module defines `LspPolicyCoordinates`, the coordinate system for the
//! LSP query policy. Every decision is expressed as a continuous coordinate,
//! not a categorical mode, enabling gradient-based training.
//!
//! # The Coordinate Philosophy
//!
//! Instead of hard-coded strategies ("greedy" vs "exploratory"), we define a
//! smooth manifold of policy behaviors. Training navigates this manifold to
//! find optimal regions. No discrete choices, only continuous coordinates.
//!
//! # Coordinate Categories
//!
//! ## Resource Physics
//!
//! - `marginal_utility_floor`: The Lagrangian multiplier - the "price" of a
//!   query. Stop when expected gain drops below this threshold. Replaces crude
//!   budget ratios with economics-inspired stopping criterion.
//!
//! - `batch_latency_bias`: Controls wavefront count. 0.0 = sequential (smart,
//!   3 waves), 1.0 = batch (fast, 1 wave). Trainable latency/quality trade-off.
//!
//! ## Signal Weights (The "Desire" Vector)
//!
//! These weights define what makes a query site valuable:
//!
//! - `weight_centrality`: PageRank score - structural importance
//! - `weight_uncertainty`: (1.0 - heuristic_confidence) - how uncertain we are
//! - `weight_coherence`: Group gain - shared receiver count (one query → N edges)
//! - `weight_causality`: Root vs Leaf preference (DAG depth in type inference)
//! - `weight_bridge`: Betweenness centrality proxy - graph connectivity impact
//!
//! ## Attention Beam Direction
//!
//! Softmax logits for simplex normalization (sum to 1.0). These prevent
//! "dimensionality fighting" during training by creating unconstrained surface:
//!
//! - `spread_logit_structural`: Follow call edges
//! - `spread_logit_semantic`: Follow type hierarchy
//! - `spread_logit_spatial`: Follow file locality
//!
//! ## Navigation
//!
//! - `focus_temperature`: Entropy of sampling distribution (low = tight beam)
//! - `gated_threshold`: Stochastic dropout for performance
//!
//! ## Structural Plasticity ("Gene Expression")
//!
//! - `interaction_mixing`: [0.0, 1.0] blend between additive (OR-logic) and
//!   multiplicative (AND-logic) combination of signals. Lets L2 discover
//!   optimal combination structure without recompilation.
//!
//! # Key Insights
//!
//! 1. **Simplex Constraint**: `spread_*` params compete for probability mass.
//!    Use softmax logits to create unconstrained training surface.
//!
//! 2. **Marginal Utility > Budget**: Don't cap at "40% of queries". Learn a
//!    "price" and stop when gain < price. The optimizer finds the budget.
//!
//! 3. **Coherence (Group Gain)**: Resolving `user: User` instantly resolves
//!    `user.name`, `user.id`, `user.email`. One query buys N edges.
//!
//! 4. **Causality (DAG Depth)**: Type inference flows from definitions to usages.
//!    Can't resolve `factory.create().process()` until you resolve `factory`.
//!
//! 5. **Gene Expression**: `interaction_mixing` lets L2 discover whether optimal
//!    policy is additive (high centrality OR high uncertainty) vs multiplicative
//!    (high centrality AND high uncertainty). Structure becomes trainable.

use serde::{Deserialize, Serialize};

/// Continuous coordinates for LSP query policy.
///
/// All decisions are trainable - no discrete branches, no categorical modes.
/// The structure follows the "dissolved decision tree" philosophy:
/// every degree of freedom is a continuous parameter that L1/L2 can optimize.
///
/// # Coordinate Spaces
///
/// ## Resource Physics
/// - `marginal_utility_floor`: Lagrangian multiplier (not budget cap)
/// - `batch_latency_bias`: Sequential vs parallel trade-off
///
/// ## Signal Weights (The "Desire" Vector)
/// Weighted scoring of call sites. Each weight is independent - not a simplex.
/// The optimizer learns which signals matter most for ranking query targets.
///
/// ## Attention Beam (Softmax Logits)
/// Direction of information spread. These ARE a simplex (sum to 1.0) so we use
/// logits to avoid fighting during gradient descent.
///
/// ## Navigation
/// Control focus and exploration via temperature and gating.
///
/// ## Gene Expression (Structural Plasticity)
/// The `interaction_mixing` parameter lets L2 discover whether optimal policy
/// combines signals additively (OR-logic) or multiplicatively (AND-logic).
/// Structure itself becomes a coordinate.
///
/// # Example Usage
///
/// ```rust
/// use ripmap::lsp::LspPolicyCoordinates;
///
/// // Default: middle-of-range neutral policy
/// let coords = LspPolicyCoordinates::default();
///
/// // Extract softmax-normalized attention weights
/// let (structural, semantic, spatial) = coords.spread_weights();
/// assert!((structural + semantic + spatial - 1.0).abs() < 1e-6);
///
/// // Custom policy: greedy, multiplicative, tight beam
/// let greedy = LspPolicyCoordinates {
///     weight_centrality: 2.0,
///     weight_uncertainty: 0.1,
///     interaction_mixing: 0.9,  // Heavy AND-logic
///     focus_temperature: 0.1,   // Tight attention
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspPolicyCoordinates {
    // ========================================================================
    // Resource Physics
    // ========================================================================

    /// The "price" of an LSP query. Stop when expected marginal gain < this.
    ///
    /// This is a Lagrangian multiplier, not a crude budget cap. The optimizer
    /// discovers the query budget implicitly by learning the price.
    ///
    /// **Rationale**: Instead of "query 40% of nodes", we ask "what's the
    /// marginal value of the next query?" When it drops below this floor,
    /// we stop. The budget emerges from the value landscape.
    ///
    /// **Range**: [0.001, 0.1]
    /// - Low (0.001): Query aggressively until diminishing returns
    /// - High (0.1): Conservative, only high-value queries
    ///
    /// **Training**: L1 adjusts this to balance NDCG vs latency cost.
    pub marginal_utility_floor: f64,

    /// Latency trade-off: 0.0 = Sequential (smart), 1.0 = Batch (fast).
    ///
    /// Pure sequential: Query roots, update type cache, re-score, query next wave.
    /// Maximizes information gain but slowest (3+ wavefronts).
    ///
    /// Pure batch: Query everything at once. Fast but ignores sequential
    /// information gain (1 wavefront).
    ///
    /// **Compromise**: Generational wavefronts. This coordinate controls the number:
    /// - 0.0 → 3 waves (Spine, Frontier, Fill)
    /// - 0.5 → 2 waves (Spine, Fill)
    /// - 1.0 → 1 wave (Batch)
    ///
    /// **Range**: [0.0, 1.0]
    ///
    /// **Training**: L1 discovers optimal latency/accuracy trade-off for the
    /// specific codebase topology.
    pub batch_latency_bias: f64,

    // ========================================================================
    // Signal Weights (The "Desire" Vector)
    // ========================================================================
    //
    // These are NOT a simplex - they're independent weights that combine to
    // score each call site. The optimizer learns which signals matter most.
    //
    // Scoring function (before gene expression):
    //   score = Σ weight_i * signal_i
    //
    // Then `interaction_mixing` blends this with multiplicative combination.

    /// Weight for PageRank centrality score.
    ///
    /// **Signal**: log(pagerank) - centrality in the call graph.
    /// High-centrality nodes are load-bearing: resolving them clarifies many
    /// downstream call edges.
    ///
    /// **Rationale**: Hubs have outsized impact. One query to `User.__init__`
    /// resolves hundreds of `user.name`, `user.id`, etc.
    ///
    /// **Range**: [0.0, 2.0]
    /// - 0.0: Ignore centrality entirely
    /// - 1.0: Linear preference for hubs
    /// - 2.0: Strong hub bias
    ///
    /// **Training**: L1 balances hub queries vs uniform coverage.
    pub weight_centrality: f64,

    /// Weight for heuristic uncertainty (1.0 - confidence).
    ///
    /// **Signal**: 1.0 - heuristic_confidence
    /// - NameMatch: conf ≈ 0.4 → uncertainty = 0.6
    /// - TypeHint:  conf ≈ 0.8 → uncertainty = 0.2
    /// - LSP:       conf ≈ 0.95 → uncertainty = 0.05
    ///
    /// **Rationale**: Target edges where heuristics are uncertain. LSP provides
    /// maximum information gain where we're currently confused.
    ///
    /// **Range**: [0.0, 2.0]
    /// - 0.0: Don't prioritize uncertain edges
    /// - 1.0: Linear preference for uncertainty
    /// - 2.0: Strong exploration bias
    ///
    /// **Training**: L1 discovers exploration/exploitation balance.
    pub weight_uncertainty: f64,

    /// Weight for coherence - the "group gain" factor.
    ///
    /// **Signal**: log(1 + receiver_count)
    /// How many other call sites share the same receiver type?
    ///
    /// **Rationale**: Resolving `user: User` instantly resolves `user.name`,
    /// `user.id`, `user.email`, etc. One query buys N edge resolutions.
    /// This captures fan-out potential.
    ///
    /// **Range**: [0.0, 2.0]
    /// - 0.0: Ignore coherence
    /// - 1.0: Linear preference for high fan-out
    /// - 2.0: Strong coherence bias
    ///
    /// **Training**: L1 learns whether coherence or individual edge value matters more.
    pub weight_coherence: f64,

    /// Weight for causality - root vs leaf preference in type DAG.
    ///
    /// **Signal**: is_root(tag) - boolean cast to f64
    /// Type inference flows from definitions → usages. You can't resolve
    /// `factory.create().process()` until you resolve `factory`.
    ///
    /// **Rationale**: The type DAG has causal structure. Roots unlock leaves.
    /// Prefer querying:
    /// - Variable definitions
    /// - Import statements
    /// - Class/function boundaries
    ///
    /// Over querying:
    /// - Deep method chains
    /// - Nested expressions
    ///
    /// **Range**: [0.0, 2.0]
    /// - 0.0: No causality preference
    /// - 1.0: Linear root bias
    /// - 2.0: Strong root bias
    ///
    /// **Training**: L1 discovers optimal root/leaf balance for the DAG topology.
    pub weight_causality: f64,

    /// Weight for betweenness centrality (bridge edges).
    ///
    /// **Signal**: betweenness_proxy
    /// Edges whose removal would disconnect clusters. Bridges are critical
    /// connective tissue.
    ///
    /// **Rationale**: Some edges are structural load-bearing. They connect
    /// otherwise-isolated components. Resolving them prevents cluster isolation
    /// in the graph.
    ///
    /// **Range**: [0.0, 2.0]
    /// - 0.0: Ignore bridge structure
    /// - 1.0: Linear bridge preference
    /// - 2.0: Strong bridge bias
    ///
    /// **Training**: L1 learns whether bridges or hubs matter more.
    pub weight_bridge: f64,

    // ========================================================================
    // Attention Beam Direction (Softmax Logits)
    // ========================================================================
    //
    // These control how we SPREAD attention when choosing query targets.
    // They form a simplex (sum to 1.0), so we parameterize as logits to avoid
    // "dimensionality fighting" during training.
    //
    // softmax([logit_structural, logit_semantic, logit_spatial]) → [p_s, p_e, p_sp]
    //
    // Then we sample/blend query strategies with these probabilities.

    /// Logit for structural spread - follow call graph edges.
    ///
    /// **Strategy**: Starting from a seed node, follow outgoing/incoming calls.
    /// "If we query X, also query things that X calls or that call X."
    ///
    /// **Rationale**: The call graph is the primary structure. Following edges
    /// reveals local neighborhoods.
    ///
    /// **Logit Range**: (-∞, +∞)
    /// After softmax normalization, higher logit → higher probability mass.
    pub spread_logit_structural: f64,

    /// Logit for semantic spread - follow type hierarchy.
    ///
    /// **Strategy**: If we query a method, also query other methods of the same
    /// class, or methods of related classes (subclasses, parent classes).
    ///
    /// **Rationale**: Types are semantic clusters. Resolving one method of a
    /// class makes resolving other methods cheaper (shared context).
    ///
    /// **Logit Range**: (-∞, +∞)
    pub spread_logit_semantic: f64,

    /// Logit for spatial spread - follow file locality.
    ///
    /// **Strategy**: If we query something in file F, also query other things
    /// in F or nearby files in the directory tree.
    ///
    /// **Rationale**: Files are units of coherence. Developers group related
    /// code spatially. Locality captures implicit modularity.
    ///
    /// **Logit Range**: (-∞, +∞)
    pub spread_logit_spatial: f64,

    // ========================================================================
    // Navigation
    // ========================================================================

    /// Temperature for focus sampling - entropy of the attention beam.
    ///
    /// **Mechanics**: When selecting from scored candidates, use softmax(scores / T).
    /// - T → 0: Argmax (tight beam, deterministic)
    /// - T → ∞: Uniform (wide beam, random)
    ///
    /// **Rationale**: Low temperature exploits known high-value targets. High
    /// temperature explores the scoring landscape. This is the classic
    /// exploration/exploitation dial.
    ///
    /// **Range**: [0.01, 2.0]
    /// - 0.01: Greedy, tight focus
    /// - 1.0: Balanced
    /// - 2.0: Exploratory, wide beam
    ///
    /// **Training**: L1 discovers optimal exploration for the graph topology.
    pub focus_temperature: f64,

    /// Threshold for stochastic dropout gating.
    ///
    /// **Mechanics**: After scoring, drop candidates with score < threshold.
    /// Acts as a hard floor to avoid wasting queries on low-value targets.
    ///
    /// **Rationale**: Even with temperature sampling, we want to avoid the long
    /// tail of near-zero-value queries. This is computational budgeting.
    ///
    /// **Range**: [0.0, 0.5]
    /// - 0.0: No dropout, consider everything
    /// - 0.2: Drop bottom ~20% of candidates
    /// - 0.5: Aggressive pruning
    ///
    /// **Training**: L1 balances coverage vs efficiency.
    pub gated_threshold: f64,

    // ========================================================================
    // Gene Expression (Structural Plasticity)
    // ========================================================================

    /// Mixing parameter: 0.0 = Additive (OR), 1.0 = Multiplicative (AND).
    ///
    /// **The Problem**: Should we combine signals via addition or multiplication?
    /// - Additive:        score = Σ w_i * x_i         (OR-logic: high centrality OR high uncertainty)
    /// - Multiplicative:  score = Π (1 + w_i * x_i)   (AND-logic: high centrality AND high uncertainty)
    ///
    /// **The Solution**: Don't choose. Blend.
    ///
    /// ```text
    /// additive = Σ weight_i * signal_i
    /// multiplicative = ln(Π (1 + weight_i * signal_i))
    ///                = ln(centrality * uncertainty * coherence * ...)
    ///
    /// final_score = (1 - mixing) * additive + mixing * multiplicative
    /// ```
    ///
    /// **Rationale**: The optimal combination structure is unknown a priori.
    /// By making it continuous, L2 can evolve the scoring function without
    /// code changes. This is "gene expression" - the genome (weights) is fixed,
    /// but how it's expressed (additive vs multiplicative) varies.
    ///
    /// **Range**: [0.0, 1.0]
    /// - 0.0: Pure additive (OR-logic)
    /// - 0.5: 50/50 blend
    /// - 1.0: Pure multiplicative (AND-logic)
    ///
    /// **Training**: L2 discovers emergent combination structure via meta-learning.
    pub interaction_mixing: f64,
}

impl Default for LspPolicyCoordinates {
    /// Default coordinates: middle-of-range neutral policy.
    ///
    /// These are starting values, not optimal values. Training will discover
    /// task-specific optima. Defaults are chosen to be:
    /// - Non-degenerate (not at extremes)
    /// - Balanced across signals
    /// - Conservative on resource usage
    fn default() -> Self {
        Self {
            // Resource Physics - conservative defaults
            marginal_utility_floor: 0.01,  // Middle of log range [0.001, 0.1]
            batch_latency_bias: 0.3,       // Slight preference for sequential (2-3 waves)

            // Signal Weights - uniform starting point
            weight_centrality: 1.0,
            weight_uncertainty: 1.0,
            weight_coherence: 1.0,
            weight_causality: 0.5,         // Slight root bias
            weight_bridge: 0.5,

            // Attention Beam - equal logits → uniform spread
            spread_logit_structural: 0.0,
            spread_logit_semantic: 0.0,
            spread_logit_spatial: 0.0,

            // Navigation - moderate exploration
            focus_temperature: 1.0,        // Balanced sampling
            gated_threshold: 0.1,          // Minimal dropout

            // Gene Expression - start with additive
            interaction_mixing: 0.0,       // Pure additive (easier to reason about initially)
        }
    }
}

impl LspPolicyCoordinates {
    /// Compute softmax-normalized attention spread weights.
    ///
    /// Returns: (structural, semantic, spatial) probabilities that sum to 1.0.
    ///
    /// **Why softmax?** The spread logits are unconstrained (-∞, +∞), which
    /// makes gradient descent easier. But for execution we need probabilities
    /// that sum to 1.0 (simplex constraint). Softmax gives us both.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ripmap::lsp::LspPolicyCoordinates;
    ///
    /// let coords = LspPolicyCoordinates {
    ///     spread_logit_structural: 2.0,
    ///     spread_logit_semantic: 1.0,
    ///     spread_logit_spatial: 0.0,
    ///     ..Default::default()
    /// };
    ///
    /// let (structural, semantic, spatial) = coords.spread_weights();
    /// assert!(structural > semantic);  // Higher logit → higher weight
    /// assert!(semantic > spatial);
    /// assert!((structural + semantic + spatial - 1.0).abs() < 1e-6);
    /// ```
    pub fn spread_weights(&self) -> (f64, f64, f64) {
        // Numerically stable softmax: exp(x_i - max) / Σ exp(x_j - max)
        let logits = [
            self.spread_logit_structural,
            self.spread_logit_semantic,
            self.spread_logit_spatial,
        ];

        let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        // Compute exp(x_i - max)
        let exp_vals: Vec<f64> = logits.iter()
            .map(|&x| (x - max_logit).exp())
            .collect();

        let sum: f64 = exp_vals.iter().sum();

        // Normalize
        let structural = exp_vals[0] / sum;
        let semantic = exp_vals[1] / sum;
        let spatial = exp_vals[2] / sum;

        (structural, semantic, spatial)
    }

    /// Create a "greedy" policy for testing: high centrality, tight beam.
    ///
    /// **Use case**: Baseline comparison. Pure hub-focused strategy.
    #[cfg(test)]
    pub fn greedy() -> Self {
        Self {
            weight_centrality: 2.0,
            weight_uncertainty: 0.1,
            weight_coherence: 0.5,
            weight_causality: 1.0,
            focus_temperature: 0.1,
            interaction_mixing: 0.0,
            ..Default::default()
        }
    }

    /// Create an "exploratory" policy for testing: high uncertainty, wide beam.
    ///
    /// **Use case**: Maximum exploration, for discovering unknown unknowns.
    #[cfg(test)]
    pub fn exploratory() -> Self {
        Self {
            weight_centrality: 0.5,
            weight_uncertainty: 2.0,
            weight_coherence: 0.5,
            weight_causality: 0.2,
            focus_temperature: 1.5,
            interaction_mixing: 0.0,
            ..Default::default()
        }
    }

    /// Create a "coherent" policy for testing: maximize group gain.
    ///
    /// **Use case**: Optimize for query efficiency via type fan-out.
    #[cfg(test)]
    pub fn coherent() -> Self {
        Self {
            weight_centrality: 1.0,
            weight_uncertainty: 0.5,
            weight_coherence: 2.0,
            weight_causality: 1.0,
            focus_temperature: 0.5,
            interaction_mixing: 0.5,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_coordinates() {
        let coords = LspPolicyCoordinates::default();

        // Sanity checks on defaults
        assert!(coords.marginal_utility_floor > 0.0);
        assert!(coords.marginal_utility_floor < 0.1);
        assert!(coords.batch_latency_bias >= 0.0);
        assert!(coords.batch_latency_bias <= 1.0);
        assert!(coords.interaction_mixing >= 0.0);
        assert!(coords.interaction_mixing <= 1.0);
    }

    #[test]
    fn test_spread_weights_sum_to_one() {
        let coords = LspPolicyCoordinates::default();
        let (s, e, sp) = coords.spread_weights();

        // Softmax must produce a valid probability distribution
        assert!((s + e + sp - 1.0).abs() < 1e-10, "Weights must sum to 1.0");
        assert!(s >= 0.0 && s <= 1.0);
        assert!(e >= 0.0 && e <= 1.0);
        assert!(sp >= 0.0 && sp <= 1.0);
    }

    #[test]
    fn test_spread_weights_uniform_logits() {
        let coords = LspPolicyCoordinates {
            spread_logit_structural: 1.0,
            spread_logit_semantic: 1.0,
            spread_logit_spatial: 1.0,
            ..Default::default()
        };

        let (s, e, sp) = coords.spread_weights();

        // Equal logits → equal weights (1/3 each)
        assert!((s - 1.0/3.0).abs() < 1e-10);
        assert!((e - 1.0/3.0).abs() < 1e-10);
        assert!((sp - 1.0/3.0).abs() < 1e-10);
    }

    #[test]
    fn test_spread_weights_skewed_logits() {
        let coords = LspPolicyCoordinates {
            spread_logit_structural: 10.0,  // Very high
            spread_logit_semantic: 0.0,
            spread_logit_spatial: 0.0,
            ..Default::default()
        };

        let (s, e, sp) = coords.spread_weights();

        // Structural should dominate
        assert!(s > 0.99, "High logit should capture most probability mass");
        assert!(e < 0.01);
        assert!(sp < 0.01);
    }

    #[test]
    fn test_spread_weights_negative_logits() {
        let coords = LspPolicyCoordinates {
            spread_logit_structural: -10.0,  // Very low
            spread_logit_semantic: 0.0,
            spread_logit_spatial: 5.0,       // High
            ..Default::default()
        };

        let (s, _e, sp) = coords.spread_weights();

        // Spatial should dominate, structural near-zero
        assert!(sp > 0.99);
        assert!(s < 0.01);
    }

    #[test]
    fn test_preset_policies() {
        // Ensure preset policies are reasonable
        let greedy = LspPolicyCoordinates::greedy();
        assert!(greedy.weight_centrality > greedy.weight_uncertainty);
        assert!(greedy.focus_temperature < 0.5);

        let exploratory = LspPolicyCoordinates::exploratory();
        assert!(exploratory.weight_uncertainty > exploratory.weight_centrality);
        assert!(exploratory.focus_temperature > 1.0);

        let coherent = LspPolicyCoordinates::coherent();
        assert!(coherent.weight_coherence >= coherent.weight_centrality);
        assert!(coherent.interaction_mixing > 0.0);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let coords = LspPolicyCoordinates {
            marginal_utility_floor: 0.042,
            weight_centrality: 1.337,
            spread_logit_semantic: -2.5,
            ..Default::default()
        };

        let json = serde_json::to_string(&coords).unwrap();
        let deserialized: LspPolicyCoordinates = serde_json::from_str(&json).unwrap();

        assert!((coords.marginal_utility_floor - deserialized.marginal_utility_floor).abs() < 1e-10);
        assert!((coords.weight_centrality - deserialized.weight_centrality).abs() < 1e-10);
        assert!((coords.spread_logit_semantic - deserialized.spread_logit_semantic).abs() < 1e-10);
    }
}
