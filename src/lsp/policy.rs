//! Query selection policy for LSP resolution.
//!
//! Implements "dissolved decision trees" - all choices are continuous coordinates,
//! enabling smooth optimization by L1/L2 training loops. No hard branches, only
//! weighted blends and gated stochastic attention.
//!
//! ## Architecture
//!
//! The PolicyEngine scores call sites using a "gene expression" model that blends
//! additive (OR-logic) and multiplicative (AND-logic) combination functions:
//!
//! ```text
//! Additive:       score = w_a*A + w_b*B + w_c*C + ...
//! Multiplicative: score = w_a*ln(A) + w_b*ln(B) + ... = ln(A^w_a * B^w_b * ...)
//! Blended:        score = (1-mix)*additive + mix*multiplicative
//! ```
//!
//! The `interaction_mixing` parameter lets L2 discover whether the optimal policy
//! is "high centrality OR high uncertainty" vs "high centrality AND high uncertainty"
//! without changing code structure.
//!
//! ## Wavefront Execution
//!
//! Queries are executed in 1-3 generational waves:
//! - Gen 1 (Spine): High-centrality roots (definitions, variables)
//! - Gen 2 (Frontier): Re-scored with type cache updates
//! - Gen 3 (Fill): Surgical cleanup of remaining uncertainty
//!
//! The `batch_latency_bias` coordinate controls the trade-off between
//! sequential intelligence (3 waves) and batch throughput (1 wave).
//!
//! ## Attention Beam Direction (Neighbor Selection)
//!
//! The `spread_logit_*` coordinates control HOW we expand wavefronts - which
//! neighbors to consider when generating candidates from resolved nodes:
//!
//! - **Structural** (call graph): Follow caller → callee, callee → caller edges
//! - **Semantic** (type hierarchy): Follow subtype → supertype, method → method edges
//! - **Spatial** (file locality): Prefer same file, same directory
//!
//! These logits are softmax-normalized to create a probability simplex (sum = 1.0).
//! This prevents "dimensionality fighting" during training - L1/L2 can tune logits
//! independently without constraint satisfaction conflicts.
//!
//! The spread weights are used during candidate generation to weight neighbor selection.
//! High structural weight → follow call edges aggressively.
//! High semantic weight → cluster queries by type.
//! High spatial weight → exploit file locality.
//!
//! **Key distinction**: Spread weights control WHICH candidates to consider (neighbor
//! expansion), while signal weights control HOW to score those candidates (ranking).
//! This separation lets the optimizer independently tune exploration strategy vs
//! exploitation preference.

use crate::lsp::coordinates::LspPolicyCoordinates;
use crate::types::{Tag, TagKind};

/// Query candidate with computed signals.
///
/// Represents a call site or symbol reference that could be queried via LSP.
/// Each signal captures a different aspect of "desire to query":
/// - `pagerank`: Graph centrality (hubs vs leaves)
/// - `heuristic_confidence`: How certain we already are (1.0 = certain, 0.0 = totally unknown)
/// - `coherence`: Fan-out potential (how many edges share this receiver)
/// - `is_root`: Definition vs reference (type inference DAG causality)
/// - `bridge_score`: Betweenness centrality proxy (graph connectivity impact)
#[derive(Debug, Clone)]
pub struct QueryCandidate<'a> {
    /// The tag to potentially query
    pub tag: &'a Tag,
    /// PageRank score from the call graph (raw, corpus-specific)
    pub pagerank: f64,
    /// PageRank percentile rank (0.0 = lowest, 1.0 = highest in corpus).
    /// If None, falls back to raw pagerank for normalization.
    /// This enables gradient-stabilized centrality scoring by blending
    /// with raw PR via `centrality_normalization` coordinate.
    pub pagerank_percentile: Option<f64>,
    /// Confidence from heuristic resolution (0.0 = unknown, 1.0 = certain)
    pub heuristic_confidence: f64,
    /// Coherence: number of other call sites sharing the same receiver.
    /// Resolving `user: User` instantly resolves `user.name`, `user.id`, etc.
    /// High coherence = high group gain.
    pub coherence: f64,
    /// True if this is a definition/variable (root in type inference DAG),
    /// false if it's a reference/call (leaf in DAG).
    /// Roots enable downstream inference, leaves depend on upstream.
    pub is_root: bool,
    /// Betweenness centrality proxy: how many shortest paths pass through this node.
    /// High bridge score = critical for graph connectivity.
    /// Resolving a bridge node can unlock entire clusters of otherwise disconnected components.
    pub bridge_score: f64,
}

/// The policy engine for LSP query selection.
///
/// Implements dissolved decision trees: all choices are continuous coordinates,
/// trainable by L1/L2 optimization loops. The engine scores candidates using
/// the gene expression model and selects wavefront batches using gated stochastic
/// attention.
///
/// ## Design Rationale
///
/// Traditional approaches use hard-coded heuristics ("query hubs first", "stop at 40% budget").
/// This creates fragile, untrainable systems. Instead, we parameterize every decision:
/// - How much to prefer centrality vs uncertainty? → weight_centrality, weight_uncertainty
/// - When to stop? → marginal_utility_floor (learned price, not fixed budget)
/// - How many waves? → batch_latency_bias (continuous latency/intelligence trade-off)
///
/// The optimizer discovers the budget, the strategy, and the stopping condition.
pub struct PolicyEngine {
    coords: LspPolicyCoordinates,
}

impl PolicyEngine {
    /// Create a new policy engine with the given coordinates.
    pub fn new(coords: LspPolicyCoordinates) -> Self {
        Self { coords }
    }

    /// Calculate the "energy" (desire to query) of a candidate.
    ///
    /// Uses the gene expression model to blend additive and multiplicative scoring:
    /// - Additive: `w_a*A + w_b*B + ...` (OR-logic: prefer high centrality OR high uncertainty)
    /// - Multiplicative: `w_a*ln(A) + w_b*ln(B) + ...` (AND-logic: require high centrality AND high uncertainty, with trainable weights)
    /// - Blend: `(1-mix)*additive + mix*multiplicative` where mix ∈ [0,1]
    ///
    /// ## Signal Interpretation
    ///
    /// - **Centrality**: PageRank score. Log-transformed to compress dynamic range.
    ///   High centrality = querying this reveals many downstream edges.
    ///
    /// - **Uncertainty**: `1.0 - confidence`. High uncertainty = we don't know what this is.
    ///   Querying reduces entropy the most.
    ///
    /// - **Coherence**: Number of call sites sharing the same receiver (e.g., `user.name`, `user.id`).
    ///   High coherence = one query resolves many edges (group gain).
    ///   Log-transformed because 10→20 edges is less significant than 1→2.
    ///
    /// - **Causality**: Binary signal (1.0 if root, 0.0 if leaf).
    ///   Roots are definitions/variables that enable downstream type inference.
    ///   Leaves are calls/references that depend on upstream resolution.
    ///   Prefer roots to unlock inference cascades.
    ///
    /// - **Bridge**: Betweenness centrality proxy (how many shortest paths pass through this node).
    ///   High bridge score = critical for graph connectivity.
    ///   Resolving a bridge node can unlock entire clusters of otherwise disconnected components.
    ///   Log-transformed (ln(1+x)) like coherence for smooth behavior at zero.
    ///
    /// ## Gene Expression Rationale
    ///
    /// The `interaction_mixing` parameter lets L2 discover optimal combination structure:
    /// - `mixing = 0.0`: Pure additive. "Query if high centrality OR high uncertainty."
    /// - `mixing = 1.0`: Pure multiplicative. "Query only if high centrality AND high uncertainty."
    /// - `mixing = 0.5`: Balanced. L2 can evolve the logic without code changes.
    ///
    /// This makes the **structure** of the scoring function a trainable coordinate.
    pub fn score_site(&self, candidate: &QueryCandidate) -> f64 {
        // === CENTRALITY NORMALIZATION ===
        // Blend raw centrality with normalized percentile to stabilize gradients.
        // centrality_normalization: 0.0 = raw PR, 1.0 = percentile rank
        let raw_centrality = candidate.pagerank;
        let percentile_centrality = candidate.pagerank_percentile.unwrap_or(raw_centrality);

        let norm = self.coords.centrality_normalization.clamp(0.0, 1.0);
        let centrality = (1.0 - norm) * raw_centrality + norm * percentile_centrality;

        // Extract remaining signals
        let uncertainty = 1.0 - candidate.heuristic_confidence;
        let coherence = candidate.coherence;
        let causality = if candidate.is_root { 1.0 } else { 0.0 };
        let bridge_score = candidate.bridge_score; // Worker 2 is adding this field

        // Additive combination (OR-logic)
        // Log-transform centrality to compress dynamic range (PageRank can span orders of magnitude)
        // Log1p for coherence: ln(1+x) is smooth at x=0, and 1→2 is more significant than 10→11
        let additive = self.coords.weight_centrality * centrality.ln().max(-10.0)
            + self.coords.weight_uncertainty * uncertainty
            + self.coords.weight_coherence * coherence.ln_1p()
            + self.coords.weight_causality * causality
            + self.coords.weight_bridge * bridge_score.ln_1p(); // Worker 2 is adding this weight

        // Multiplicative combination (AND-logic)
        // Weighted log-sum: w_a*ln(A) + w_b*ln(B) + ... = ln(A^w_a * B^w_b * ...)
        // This applies trainable weights to each signal in the geometric mean.
        // epsilon guards against ln(0). For binary causality, use multiplicative gate:
        // if is_root: no penalty, if not is_root: negative contribution proportional to weight.
        let epsilon = 1e-10;
        let causality_mult = if candidate.is_root {
            0.0
        } else {
            -self.coords.weight_causality
        };

        let multiplicative = self.coords.weight_centrality * centrality.max(epsilon).ln()
            + self.coords.weight_uncertainty * uncertainty.max(epsilon).ln()
            + self.coords.weight_coherence * (1.0 + coherence).max(epsilon).ln()
            + self.coords.weight_bridge * (1.0 + bridge_score).max(epsilon).ln()
            + causality_mult;

        // Gene expression: blend additive and multiplicative
        // interaction_mixing ∈ [0, 1] controls the structure of the combination
        let mixing = self.coords.interaction_mixing.clamp(0.0, 1.0);
        (1.0 - mixing) * additive + mixing * multiplicative
    }

    /// Select a wavefront batch using gated stochastic attention with exploration floor.
    ///
    /// ## Algorithm
    ///
    /// 1. Score all candidates using `score_site()` (produces log-space scores)
    /// 2. Apply Boltzmann transformation to convert log-space to probability space
    /// 3. Sort by transformed score (descending)
    /// 4. **EXPLORATION FLOOR**: Force minimum fraction through unconditionally (epsilon-greedy)
    /// 5. Apply gated threshold dropout to remaining candidates
    /// 6. Apply marginal utility floor: stop when expected gain drops below price
    ///
    /// ## Boltzmann Transformation (Anti-Matter Trap Fix)
    ///
    /// The `score_site()` method returns log-space scores (which can be negative).
    /// But `gated_threshold` and `marginal_utility_floor` are defined in [0, 1]
    /// probability space. Without transformation, negative scores would always fail
    /// the threshold filter (e.g., -5.0 >= 0.1 is always false).
    ///
    /// We apply: `probability = exp(log_score / temperature)`
    ///
    /// This converts log-space energies to Boltzmann probabilities using
    /// `focus_temperature` as the temperature parameter. The temperature controls
    /// the sharpness of the distribution:
    /// - Low temperature → sharp focus on highest scores
    /// - High temperature → more uniform distribution
    ///
    /// ## Exploration Floor (Economy Paradox Fix)
    ///
    /// **The Problem**: If `gated_threshold` is tuned too high, NO queries are made.
    /// - Cost = 0 (good!)
    /// - NDCG = baseline (bad!)
    /// - Gradient = 0 (optimizer stuck!)
    ///
    /// Random exploration (making queries) initially LOWERS reward because Cost > 0
    /// but NDCG improvement isn't immediate. The optimizer can't discover that
    /// querying is valuable because the gradient is zero.
    ///
    /// **The Fix**: Epsilon-greedy exploration. Force `exploration_floor` fraction
    /// of candidates through UNCONDITIONALLY (by score rank, not threshold).
    /// This ensures:
    /// 1. Non-zero gradient signal even when thresholds are high
    /// 2. Discovery that querying improves NDCG (which eventually outweighs cost)
    /// 3. Escape from "do nothing" local optimum
    ///
    /// The exploration set takes the TOP-scoring candidates (not random), so we're
    /// still exploiting the scoring function, just bypassing the threshold gate.
    ///
    /// ## Gated Threshold (Stochastic Dropout)
    ///
    /// The `gated_threshold` parameter controls attention beam focus:
    /// - Low threshold → wide beam, include more candidates
    /// - High threshold → tight beam, only top candidates
    ///
    /// This is a form of stochastic dropout that prunes the attention distribution.
    /// Now operates on transformed probabilities in [0, 1] space.
    ///
    /// Applied AFTER exploration floor - only affects exploitation candidates.
    ///
    /// ## Marginal Utility Floor (Lagrangian Multiplier)
    ///
    /// Traditional approaches cap at "40% of queries" (arbitrary budget).
    /// Instead, we learn a "price" (`marginal_utility_floor`) and stop when
    /// expected gain drops below this price.
    ///
    /// The optimizer discovers the budget implicitly by tuning the price.
    /// This is the Lagrangian multiplier for the latency constraint.
    /// Now operates on transformed probabilities in [0, 1] space.
    ///
    /// Applied AFTER exploration floor - only affects exploitation candidates.
    ///
    /// ## Returns
    ///
    /// Tags sorted by score, filtered by thresholds. Ready for batch LSP query.
    /// Includes both exploration set (forced through) and exploitation set (threshold-gated).
    pub fn select_wavefront<'a>(&self, candidates: Vec<QueryCandidate<'a>>) -> Vec<&'a Tag> {
        if candidates.is_empty() {
            return vec![];
        }

        // Score all candidates (log-space)
        let scored: Vec<(f64, &Tag)> = candidates
            .iter()
            .map(|c| (self.score_site(c), c.tag))
            .collect();

        // Boltzmann transform: convert log-space scores to probability-like values
        // Using focus_temperature as the temperature parameter
        // Avoid division by zero with a minimum temperature
        let temp = self.coords.focus_temperature.max(0.01);
        let mut transformed: Vec<(f64, &Tag)> = scored
            .into_iter()
            .map(|(score, tag)| ((score / temp).exp(), tag))
            .collect();

        // Sort by transformed score (descending - highest first)
        transformed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // === EXPLORATION FLOOR ===
        // Force minimum fraction through unconditionally to prevent "do nothing" optimum.
        // Take top-scoring candidates (not random) - still exploiting the scoring function.
        let min_explore = (candidates.len() as f64 * self.coords.exploration_floor).ceil() as usize;
        let min_explore = min_explore.max(1).min(candidates.len()); // At least 1, at most all

        // Exploration set: Top min_explore candidates, unconditional
        let mut selected: Vec<&Tag> = transformed
            .iter()
            .take(min_explore)
            .map(|(_, tag)| *tag)
            .collect();

        // === EXPLOITATION (THRESHOLD-GATED) ===
        // Remaining candidates must pass both gated_threshold and marginal_utility_floor
        let exploitation: Vec<&Tag> = transformed
            .iter()
            .skip(min_explore)
            .filter(|(prob, _)| *prob >= self.coords.gated_threshold)
            .take_while(|(prob, _)| *prob >= self.coords.marginal_utility_floor)
            .map(|(_, tag)| *tag)
            .collect();

        selected.extend(exploitation);

        selected
    }

    /// Check if we should run another wavefront generation.
    ///
    /// ## Dissolved Decision Tree Philosophy
    ///
    /// Instead of hard-coded thresholds ("run exactly N waves"), use a continuous
    /// stopping condition based on marginal utility and latency tolerance.
    ///
    /// This replaces the discrete `num_generations()` budget with a Lagrangian
    /// stopping check: "Is the next wave worth the latency cost?"
    ///
    /// ## Algorithm
    ///
    /// 1. **Base check**: Did last wave produce enough value?
    ///    If `last_yield_rate < marginal_utility_floor`, stop immediately.
    ///
    /// 2. **Diminishing returns**: Each successive wave is less valuable.
    ///    Apply exponential decay based on `batch_latency_bias`:
    ///    - High bias (→1.0): Strong decay → fewer waves (batch mode)
    ///    - Low bias (→0.0): Weak decay → more waves (sequential mode)
    ///
    /// 3. **Adjusted threshold**: The effective threshold rises with each wave:
    ///    `threshold_wave_N = marginal_utility_floor / decay^N`
    ///
    ///    This encodes: "Later waves must have higher yield to justify latency."
    ///
    /// ## Parameters
    ///
    /// - `last_yield_rate`: Edges resolved per query in the previous wave.
    ///   Measures actual productivity of the last wavefront.
    ///
    /// - `wave_index`: Which wave we're considering (0-indexed).
    ///   - 0: First wave (Spine)
    ///   - 1: Second wave (Frontier)
    ///   - 2: Third wave (Fill)
    ///   - ...
    ///
    /// ## How This Replaces num_generations()
    ///
    /// The old approach hard-coded: "run exactly 1, 2, or 3 waves based on bias."
    /// This created discontinuities at 0.5 and 0.8 that L1 cannot optimize through.
    ///
    /// The new approach asks: "Did the last wave produce enough value to justify
    /// another?" The answer depends continuously on:
    /// - The actual yield from the last wave (empirical data)
    /// - The marginal utility floor (learned price)
    /// - The batch_latency_bias (decay rate, continuous)
    /// - The wave index (diminishing returns)
    ///
    /// This makes the budget emerge from the value landscape, not from a discrete lookup.
    ///
    /// ## Training Dynamics
    ///
    /// - **Low batch_latency_bias** (→0.0): Weak decay → tolerates 3+ waves
    ///   L1 discovers this is optimal for stable codebases where sequential gain is high.
    ///
    /// - **High batch_latency_bias** (→1.0): Strong decay → stops after 1 wave
    ///   L1 discovers this is optimal for high-churn codebases where speed matters.
    ///
    /// - **Middle bias** (≈0.5): Moderate decay → 2 waves typical
    ///   Balanced latency/intelligence trade-off.
    ///
    /// The optimizer smoothly navigates this trade-off without hitting discontinuities.
    ///
    /// ## Returns
    ///
    /// `true` if we should run another wavefront, `false` if we should stop.
    pub fn should_continue_wavefront(&self, last_yield_rate: f64, wave_index: usize) -> bool {
        // Base check: Did last wave meet minimum utility?
        if last_yield_rate < self.coords.marginal_utility_floor {
            return false;
        }

        // Exponential decay based on batch_latency_bias
        // High bias → strong decay (few waves)
        // Low bias → weak decay (many waves)
        //
        // We map bias ∈ [0, 1] to decay_rate ∈ [0.9, 0.3]:
        // - bias=0.0 → decay=0.9 (weak decay, tolerates many waves)
        // - bias=1.0 → decay=0.3 (strong decay, stops quickly)
        let decay_rate = 0.9 - 0.6 * self.coords.batch_latency_bias.clamp(0.0, 1.0);

        // Apply diminishing returns: each wave must clear a higher bar
        // decay^0 = 1.0 (first wave, no penalty)
        // decay^1 = 0.9 (second wave, slight penalty)
        // decay^2 = 0.81 (third wave, stronger penalty)
        let decay = decay_rate.powi(wave_index as i32);

        // Adjusted threshold rises with each wave
        // Guard against division by very small decay values
        let adjusted_threshold = self.coords.marginal_utility_floor / decay.max(0.01);

        // Continue if yield exceeds the adjusted threshold
        last_yield_rate >= adjusted_threshold
    }

    /// Check if we should continue querying based on marginal utility.
    ///
    /// ## Marginal Utility Check
    ///
    /// After each wavefront, we measure the "yield rate":
    /// `yield = (newly_resolved_edges) / (queries_issued)`
    ///
    /// If yield drops below the marginal utility floor, stop.
    /// This implements the Lagrangian stopping condition.
    ///
    /// ## Why This Works
    ///
    /// Early queries resolve high-coherence receivers (e.g., `user: User`),
    /// instantly resolving many downstream edges. High yield.
    ///
    /// Later queries hit leaf calls with low fan-out. Low yield.
    ///
    /// The marginal utility floor is the learned "price" - the minimum yield
    /// we're willing to pay latency for. When yield drops below price, stop.
    ///
    /// ## Parameters
    ///
    /// - `last_yield_rate`: Yield from the previous wavefront (edges resolved per query)
    ///
    /// ## Returns
    ///
    /// True if we should issue another wavefront, false if we should stop.
    pub fn should_continue(&self, last_yield_rate: f64) -> bool {
        last_yield_rate >= self.coords.marginal_utility_floor
    }

    /// Expand neighbors using spread-weighted attention.
    ///
    /// Given a resolved node, generate candidate neighbors weighted by:
    /// - **Structural**: Follow call edges (caller → callee, callee → caller)
    /// - **Semantic**: Follow type hierarchy (subtype → supertype, etc.)
    /// - **Spatial**: Follow file locality (same file has higher weight)
    ///
    /// The `spread_logit_*` coordinates control the attention beam direction.
    /// Softmax ensures they sum to 1.0 (simplex constraint).
    ///
    /// ## Usage During Wavefront Expansion
    ///
    /// When we resolve a node via LSP and want to expand to neighbors, these weights
    /// determine the mix of neighbor selection strategies:
    /// - High structural weight → aggressively follow call graph edges
    /// - High semantic weight → cluster queries within type hierarchies
    /// - High spatial weight → exploit file locality, prefer nearby symbols
    ///
    /// This is orthogonal to candidate scoring (`score_site`). Spread weights control
    /// WHICH candidates to generate (exploration strategy), while signal weights control
    /// HOW to rank those candidates (exploitation preference).
    ///
    /// ## Returns
    ///
    /// `NeighborWeights` with softmax-normalized probabilities that sum to 1.0.
    pub fn neighbor_weights(&self) -> NeighborWeights {
        let (w_struct, w_sem, w_spatial) = self.coords.spread_weights();
        NeighborWeights {
            structural: w_struct,
            semantic: w_sem,
            spatial: w_spatial,
        }
    }
}

/// Weights for neighbor expansion during wavefront execution.
///
/// These probabilities control the mix of neighbor selection strategies when expanding
/// from a resolved node. They sum to 1.0 (softmax-normalized simplex).
///
/// ## Interpretation
///
/// - `structural`: Probability mass for following call graph edges
///   * High structural → follow callers/callees aggressively
///   * Low structural → ignore call structure during expansion
///
/// - `semantic`: Probability mass for following type hierarchy edges
///   * High semantic → cluster queries by type (methods of same class, subtype relationships)
///   * Low semantic → ignore type structure during expansion
///
/// - `spatial`: Probability mass for following file locality
///   * High spatial → prefer symbols in the same file or nearby files
///   * Low spatial → ignore file proximity during expansion
///
/// ## Training Dynamics
///
/// The optimizer discovers the optimal exploration mix for each codebase:
/// - OOP codebases might favor semantic (type-driven expansion)
/// - Functional codebases might favor structural (call-driven expansion)
/// - Monolithic files might favor spatial (file-locality expansion)
///
/// By making this continuous, L1/L2 can adapt the exploration strategy without code changes.
#[derive(Debug, Clone, Copy)]
pub struct NeighborWeights {
    /// Weight for call graph edges (callers/callees)
    pub structural: f64,
    /// Weight for type hierarchy edges (subtypes/supertypes)
    pub semantic: f64,
    /// Weight for file locality (same file, same directory)
    pub spatial: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lsp::coordinates::LspPolicyCoordinates;
    use crate::types::Tag;

    fn make_test_tag(name: &str, kind: TagKind) -> Tag {
        Tag {
            rel_fname: "test.py".into(),
            fname: "/test.py".into(),
            line: 1,
            name: name.into(),
            kind,
            node_type: "function".into(),
            parent_name: None,
            parent_line: None,
            signature: None,
            fields: None,
            metadata: None,
        }
    }

    #[test]
    fn test_score_site_additive() {
        // Pure additive mode (interaction_mixing = 0.0)
        let coords = LspPolicyCoordinates {
            weight_centrality: 1.0,
            weight_uncertainty: 1.0,
            weight_coherence: 1.0,
            weight_causality: 1.0,
            weight_bridge: 1.0,
            interaction_mixing: 0.0, // Pure additive
            ..Default::default()
        };

        let engine = PolicyEngine::new(coords);
        let tag = make_test_tag("foo", TagKind::Def);

        let candidate = QueryCandidate {
            tag: &tag,
            pagerank: 0.5,
            pagerank_percentile: None,
            heuristic_confidence: 0.2, // High uncertainty (0.8)
            coherence: 5.0,
            is_root: true,
            bridge_score: 2.0, // Worker 2 is adding this field
        };

        let score = engine.score_site(&candidate);

        // Should be additive combination
        // ln(0.5) + 0.8 + ln(6) + 1.0 + ln(3)
        let expected = 0.5_f64.ln() + 0.8 + 6.0_f64.ln() + 1.0 + 3.0_f64.ln();
        assert!(
            (score - expected).abs() < 0.01,
            "score={}, expected={}",
            score,
            expected
        );
    }

    #[test]
    fn test_score_site_multiplicative() {
        // Pure multiplicative mode (interaction_mixing = 1.0)
        let coords = LspPolicyCoordinates {
            weight_centrality: 1.0,
            weight_uncertainty: 1.0,
            weight_coherence: 1.0,
            weight_causality: 1.0,
            weight_bridge: 1.0,
            interaction_mixing: 1.0, // Pure multiplicative
            ..Default::default()
        };

        let engine = PolicyEngine::new(coords);
        let tag = make_test_tag("bar", TagKind::Ref);

        let candidate = QueryCandidate {
            tag: &tag,
            pagerank: 0.5,
            pagerank_percentile: None,
            heuristic_confidence: 0.2, // Uncertainty = 0.8
            coherence: 5.0,
            is_root: false,    // Not a root, so causality_mult = -weight_causality
            bridge_score: 2.0, // Worker 2 is adding this field
        };

        let score = engine.score_site(&candidate);

        // Should be weighted multiplicative:
        // w_c*ln(centrality) + w_u*ln(uncertainty) + w_coh*ln(1+coherence) + w_b*ln(1+bridge) + causality_mult
        // causality_mult = -1.0 for non-roots (gate penalty)
        let expected = 1.0 * 0.5_f64.ln()
            + 1.0 * 0.8_f64.ln()
            + 1.0 * 6.0_f64.ln()
            + 1.0 * 3.0_f64.ln()
            + (-1.0); // Non-root penalty
        assert!(
            (score - expected).abs() < 0.01,
            "score={}, expected={}",
            score,
            expected
        );
    }

    #[test]
    fn test_select_wavefront_gated_threshold() {
        let coords = LspPolicyCoordinates {
            weight_centrality: 1.0,
            weight_uncertainty: 1.0,
            weight_coherence: 0.0,
            weight_causality: 0.0,
            weight_bridge: 0.0,
            interaction_mixing: 0.0,
            gated_threshold: 0.5, // Filter out low scores
            marginal_utility_floor: 0.0,
            ..Default::default()
        };

        let engine = PolicyEngine::new(coords);
        let tag1 = make_test_tag("high", TagKind::Def);
        let tag2 = make_test_tag("low", TagKind::Def);

        let candidates = vec![
            QueryCandidate {
                tag: &tag1,
                pagerank: 0.8,
                pagerank_percentile: None,
                heuristic_confidence: 0.1, // High uncertainty
                coherence: 0.0,
                is_root: true,
                bridge_score: 0.0,
            },
            QueryCandidate {
                tag: &tag2,
                pagerank: 0.1,
                pagerank_percentile: None,
                heuristic_confidence: 0.9, // Low uncertainty
                coherence: 0.0,
                is_root: false,
                bridge_score: 0.0,
            },
        ];

        let selected = engine.select_wavefront(candidates);

        // Only high-scoring candidate should pass gated threshold
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].name.as_ref(), "high");
    }

    #[test]
    fn test_select_wavefront_marginal_utility() {
        let coords = LspPolicyCoordinates {
            weight_centrality: 1.0,
            weight_uncertainty: 1.0,
            weight_coherence: 0.0,
            weight_causality: 0.0,
            weight_bridge: 0.0,
            interaction_mixing: 0.0,
            gated_threshold: 0.0,
            marginal_utility_floor: 1.0, // High floor - only top candidates
            ..Default::default()
        };

        let engine = PolicyEngine::new(coords);
        let tag1 = make_test_tag("top", TagKind::Def);
        let tag2 = make_test_tag("mid", TagKind::Def);

        let candidates = vec![
            QueryCandidate {
                tag: &tag1,
                pagerank: 0.9,
                pagerank_percentile: None,
                heuristic_confidence: 0.1,
                coherence: 0.0,
                is_root: true,
                bridge_score: 0.0,
            },
            QueryCandidate {
                tag: &tag2,
                pagerank: 0.5,
                pagerank_percentile: None,
                heuristic_confidence: 0.5,
                coherence: 0.0,
                is_root: false,
                bridge_score: 0.0,
            },
        ];

        let selected = engine.select_wavefront(candidates);

        // Only candidate exceeding marginal utility floor
        assert!(selected.len() >= 1);
        assert_eq!(selected[0].name.as_ref(), "top");
    }

    #[test]
    fn test_should_continue_wavefront_sequential() {
        // Low batch_latency_bias → weak decay → tolerates many waves
        let coords = LspPolicyCoordinates {
            batch_latency_bias: 0.0, // Fully sequential (decay = 0.9)
            marginal_utility_floor: 0.1,
            ..Default::default()
        };
        let engine = PolicyEngine::new(coords);

        // High yield (0.2) should continue through multiple waves
        assert!(engine.should_continue_wavefront(0.2, 0)); // Wave 0: threshold = 0.1 / 0.9^0 = 0.1
        assert!(engine.should_continue_wavefront(0.2, 1)); // Wave 1: threshold = 0.1 / 0.9^1 ≈ 0.111
        assert!(engine.should_continue_wavefront(0.2, 2)); // Wave 2: threshold = 0.1 / 0.9^2 ≈ 0.123

        // Moderate yield (0.12) should continue for a few waves then stop
        // Wave 2: threshold = 0.1 / 0.9^2 ≈ 0.123 → 0.12 < 0.123 → stops
        assert!(engine.should_continue_wavefront(0.12, 0));  // threshold = 0.1
        assert!(engine.should_continue_wavefront(0.12, 1));  // threshold ≈ 0.111
        assert!(!engine.should_continue_wavefront(0.12, 2)); // threshold ≈ 0.123 → stops
    }

    #[test]
    fn test_should_continue_wavefront_batched() {
        // High batch_latency_bias → strong decay → stops quickly
        let coords = LspPolicyCoordinates {
            batch_latency_bias: 1.0, // Fully batched (decay = 0.3)
            marginal_utility_floor: 0.1,
            ..Default::default()
        };
        let engine = PolicyEngine::new(coords);

        // Even high yield should stop quickly due to strong decay
        assert!(engine.should_continue_wavefront(0.5, 0)); // Wave 0: threshold = 0.1 / 0.3^0 = 0.1
        assert!(engine.should_continue_wavefront(0.5, 1)); // Wave 1: threshold = 0.1 / 0.3^1 ≈ 0.333
        assert!(!engine.should_continue_wavefront(0.5, 2)); // Wave 2: threshold = 0.1 / 0.3^2 ≈ 1.11 (stops)

        // Moderate yield stops immediately at wave 1
        assert!(engine.should_continue_wavefront(0.3, 0));
        assert!(!engine.should_continue_wavefront(0.3, 1));
    }

    #[test]
    fn test_should_continue_wavefront_moderate() {
        // Medium batch_latency_bias → moderate decay → 2-3 waves typical
        let coords = LspPolicyCoordinates {
            batch_latency_bias: 0.5, // Moderate (decay ≈ 0.6)
            marginal_utility_floor: 0.1,
            ..Default::default()
        };
        let engine = PolicyEngine::new(coords);

        // Good yield continues for 2-3 waves
        assert!(engine.should_continue_wavefront(0.3, 0)); // Wave 0: threshold = 0.1 / 0.6^0 = 0.1
        assert!(engine.should_continue_wavefront(0.3, 1)); // Wave 1: threshold = 0.1 / 0.6^1 ≈ 0.167
        assert!(engine.should_continue_wavefront(0.3, 2)); // Wave 2: threshold = 0.1 / 0.6^2 ≈ 0.278

        // Lower yield stops at wave 2
        assert!(engine.should_continue_wavefront(0.2, 0));
        assert!(engine.should_continue_wavefront(0.2, 1));
        assert!(!engine.should_continue_wavefront(0.2, 2));
    }

    #[test]
    fn test_should_continue_wavefront_base_check() {
        // Base check: yield below marginal_utility_floor always stops
        let coords = LspPolicyCoordinates {
            batch_latency_bias: 0.0, // Even with sequential preference
            marginal_utility_floor: 0.5,
            ..Default::default()
        };
        let engine = PolicyEngine::new(coords);

        // Low yield stops immediately regardless of wave index
        assert!(!engine.should_continue_wavefront(0.3, 0));
        assert!(!engine.should_continue_wavefront(0.3, 1));
        assert!(!engine.should_continue_wavefront(0.3, 2));
    }

    #[test]
    fn test_should_continue() {
        let coords = LspPolicyCoordinates {
            marginal_utility_floor: 0.5, // Stop if yield < 0.5
            ..Default::default()
        };

        let engine = PolicyEngine::new(coords);

        // High yield - should continue
        assert!(engine.should_continue(0.8));

        // Yield exactly at floor - should continue
        assert!(engine.should_continue(0.5));

        // Low yield - should stop
        assert!(!engine.should_continue(0.3));
    }

    #[test]
    fn test_root_vs_leaf_causality() {
        let coords = LspPolicyCoordinates {
            weight_centrality: 0.0,
            weight_uncertainty: 0.0,
            weight_coherence: 0.0,
            weight_causality: 10.0, // Only causality matters
            interaction_mixing: 0.0,
            ..Default::default()
        };

        let engine = PolicyEngine::new(coords);
        let tag_root = make_test_tag("root", TagKind::Def);
        let tag_leaf = make_test_tag("leaf", TagKind::Ref);

        let candidate_root = QueryCandidate {
            tag: &tag_root,
            pagerank: 0.5,
            pagerank_percentile: None,
            heuristic_confidence: 0.5,
            coherence: 0.0,
            is_root: true, // Definition
            bridge_score: 0.0,
        };

        let candidate_leaf = QueryCandidate {
            tag: &tag_leaf,
            pagerank: 0.5,
            pagerank_percentile: None,
            heuristic_confidence: 0.5,
            coherence: 0.0,
            is_root: false, // Reference
            bridge_score: 0.0,
        };

        let score_root = engine.score_site(&candidate_root);
        let score_leaf = engine.score_site(&candidate_leaf);

        // Root should score higher due to causality weight
        assert!(score_root > score_leaf);
        assert!((score_root - 10.0).abs() < 0.01); // 1.0 * 10.0
        assert!((score_leaf - 0.0).abs() < 0.01); // 0.0 * 10.0
    }
    #[test]
    fn test_neighbor_weights_sum_to_one() {
        let coords = LspPolicyCoordinates {
            spread_logit_structural: 2.0,
            spread_logit_semantic: 1.0,
            spread_logit_spatial: 0.0,
            ..Default::default()
        };

        let engine = PolicyEngine::new(coords);
        let weights = engine.neighbor_weights();

        // Weights must sum to 1.0
        let sum = weights.structural + weights.semantic + weights.spatial;
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Weights must sum to 1.0, got {}",
            sum
        );

        // All weights must be non-negative and <= 1.0
        assert!(weights.structural >= 0.0 && weights.structural <= 1.0);
        assert!(weights.semantic >= 0.0 && weights.semantic <= 1.0);
        assert!(weights.spatial >= 0.0 && weights.spatial <= 1.0);

        // Higher logit should have higher weight
        assert!(weights.structural > weights.semantic);
        assert!(weights.semantic > weights.spatial);
    }

    #[test]
    fn test_neighbor_weights_uniform() {
        let coords = LspPolicyCoordinates {
            spread_logit_structural: 0.0,
            spread_logit_semantic: 0.0,
            spread_logit_spatial: 0.0,
            ..Default::default()
        };

        let engine = PolicyEngine::new(coords);
        let weights = engine.neighbor_weights();

        // Equal logits should give equal weights (1/3 each)
        assert!((weights.structural - 1.0 / 3.0).abs() < 1e-10);
        assert!((weights.semantic - 1.0 / 3.0).abs() < 1e-10);
        assert!((weights.spatial - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_neighbor_weights_structural_dominant() {
        let coords = LspPolicyCoordinates {
            spread_logit_structural: 10.0, // Very high
            spread_logit_semantic: 0.0,
            spread_logit_spatial: 0.0,
            ..Default::default()
        };

        let engine = PolicyEngine::new(coords);
        let weights = engine.neighbor_weights();

        // Structural should dominate
        assert!(
            weights.structural > 0.99,
            "Structural weight should be > 0.99, got {}",
            weights.structural
        );
        assert!(weights.semantic < 0.01);
        assert!(weights.spatial < 0.01);
    }

    #[test]
    fn test_exploration_floor_forces_queries() {
        // Economy Paradox scenario: gated_threshold is prohibitively high
        let coords = LspPolicyCoordinates {
            gated_threshold: 0.99,        // Almost impossible to pass
            marginal_utility_floor: 0.99, // Also very high
            exploration_floor: 0.1,       // Force 10% through
            weight_centrality: 1.0,
            weight_uncertainty: 1.0,
            weight_coherence: 0.0,
            weight_causality: 0.0,
            weight_bridge: 0.0,
            interaction_mixing: 0.0,
            ..Default::default()
        };

        let engine = PolicyEngine::new(coords);

        // Create 10 low-scoring candidates that would normally be filtered out
        let tags: Vec<Tag> = (0..10)
            .map(|i| make_test_tag(&format!("tag_{}", i), TagKind::Ref))
            .collect();

        let candidates: Vec<QueryCandidate> = tags
            .iter()
            .map(|t| QueryCandidate {
                tag: t,
                pagerank: 0.001,
                pagerank_percentile: None,
                // Very low PR
                heuristic_confidence: 0.9, // High confidence (low uncertainty = 0.1)
                coherence: 0.0,
                bridge_score: 0.0,
                is_root: false,
            })
            .collect();

        let selected = engine.select_wavefront(candidates);

        // Should have at least 1 (ceil(10 * 0.1) = 1) due to exploration floor
        assert!(
            selected.len() >= 1,
            "Exploration floor should force at least 1 query, got {}",
            selected.len()
        );

        // Specifically, should get exactly 1 (the exploration floor minimum)
        // since none would pass the high thresholds
        assert_eq!(
            selected.len(),
            1,
            "With prohibitive thresholds, should get exactly the exploration floor count"
        );
    }

    #[test]
    fn test_exploration_floor_empty_candidates() {
        let coords = LspPolicyCoordinates {
            exploration_floor: 0.1,
            ..Default::default()
        };

        let engine = PolicyEngine::new(coords);
        let selected = engine.select_wavefront(vec![]);

        // Empty input should produce empty output
        assert_eq!(selected.len(), 0);
    }

    #[test]
    fn test_centrality_normalization() {
        // With raw centrality (norm = 0.0)
        let coords_raw = LspPolicyCoordinates {
            centrality_normalization: 0.0,
            weight_centrality: 1.0,
            weight_uncertainty: 0.0,
            weight_coherence: 0.0,
            weight_causality: 0.0,
            weight_bridge: 0.0,
            interaction_mixing: 0.0,
            ..Default::default()
        };

        // With percentile centrality (norm = 1.0)
        let coords_norm = LspPolicyCoordinates {
            centrality_normalization: 1.0,
            weight_centrality: 1.0,
            weight_uncertainty: 0.0,
            weight_coherence: 0.0,
            weight_causality: 0.0,
            weight_bridge: 0.0,
            interaction_mixing: 0.0,
            ..Default::default()
        };

        let engine_raw = PolicyEngine::new(coords_raw);
        let engine_norm = PolicyEngine::new(coords_norm);

        let tag = make_test_tag("test", TagKind::Def);

        // Low raw PR but high percentile
        let candidate = QueryCandidate {
            tag: &tag,
            pagerank: 0.001,  // Very low raw
            pagerank_percentile: Some(0.9), // But high percentile
            heuristic_confidence: 0.5,
            coherence: 0.0,
            bridge_score: 0.0,
            is_root: false,
        };

        let score_raw = engine_raw.score_site(&candidate);
        let score_norm = engine_norm.score_site(&candidate);

        // Normalized should score higher (percentile 0.9 vs raw 0.001)
        // Raw: ln(0.001) ≈ -6.9
        // Norm: ln(0.9) ≈ -0.105
        assert!(
            score_norm > score_raw,
            "Percentile normalization should boost low-raw high-percentile nodes. raw={}, norm={}",
            score_raw,
            score_norm
        );
    }
}
