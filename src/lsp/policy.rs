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
//! Multiplicative: score = ln(A * B * C * ...)
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
#[derive(Debug, Clone)]
pub struct QueryCandidate<'a> {
    /// The tag to potentially query
    pub tag: &'a Tag,
    /// PageRank score from the call graph
    pub pagerank: f64,
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
    /// - Multiplicative: `ln(A*B*C*...)` (AND-logic: require high centrality AND high uncertainty)
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
    /// ## Gene Expression Rationale
    ///
    /// The `interaction_mixing` parameter lets L2 discover optimal combination structure:
    /// - `mixing = 0.0`: Pure additive. "Query if high centrality OR high uncertainty."
    /// - `mixing = 1.0`: Pure multiplicative. "Query only if high centrality AND high uncertainty."
    /// - `mixing = 0.5`: Balanced. L2 can evolve the logic without code changes.
    ///
    /// This makes the **structure** of the scoring function a trainable coordinate.
    pub fn score_site(&self, candidate: &QueryCandidate) -> f64 {
        // Extract signals
        let centrality = candidate.pagerank;
        let uncertainty = 1.0 - candidate.heuristic_confidence;
        let coherence = candidate.coherence;
        let causality = if candidate.is_root { 1.0 } else { 0.0 };

        // Additive combination (OR-logic)
        // Log-transform centrality to compress dynamic range (PageRank can span orders of magnitude)
        // Log1p for coherence: ln(1+x) is smooth at x=0, and 1→2 is more significant than 10→11
        let additive = self.coords.weight_centrality * centrality.ln().max(-10.0)
            + self.coords.weight_uncertainty * uncertainty
            + self.coords.weight_coherence * coherence.ln_1p()
            + self.coords.weight_causality * causality;

        // Multiplicative combination (AND-logic)
        // Combine signals geometrically: (A * B * C).ln() = ln(A) + ln(B) + ln(C)
        // Add small epsilon to avoid ln(0). Coherence gets +1 offset so 0 coherence doesn't kill score.
        let epsilon = 1e-10;
        let multiplicative = (centrality.max(epsilon)
            * uncertainty.max(epsilon)
            * (1.0 + coherence).max(epsilon))
        .ln();

        // Gene expression: blend additive and multiplicative
        // interaction_mixing ∈ [0, 1] controls the structure of the combination
        let mixing = self.coords.interaction_mixing.clamp(0.0, 1.0);
        (1.0 - mixing) * additive + mixing * multiplicative
    }

    /// Select a wavefront batch using gated stochastic attention.
    ///
    /// ## Algorithm
    ///
    /// 1. Score all candidates using `score_site()`
    /// 2. Sort by score (descending)
    /// 3. Apply gated threshold dropout: filter out candidates with score below threshold
    /// 4. Apply marginal utility floor: stop when expected gain drops below price
    ///
    /// ## Gated Threshold (Stochastic Dropout)
    ///
    /// The `gated_threshold` parameter controls attention beam focus:
    /// - Low threshold → wide beam, include more candidates
    /// - High threshold → tight beam, only top candidates
    ///
    /// This is a form of stochastic dropout that prunes the attention distribution.
    ///
    /// ## Marginal Utility Floor (Lagrangian Multiplier)
    ///
    /// Traditional approaches cap at "40% of queries" (arbitrary budget).
    /// Instead, we learn a "price" (`marginal_utility_floor`) and stop when
    /// expected gain drops below this price.
    ///
    /// The optimizer discovers the budget implicitly by tuning the price.
    /// This is the Lagrangian multiplier for the latency constraint.
    ///
    /// ## Returns
    ///
    /// Tags sorted by score, filtered by thresholds. Ready for batch LSP query.
    pub fn select_wavefront<'a>(
        &self,
        candidates: Vec<QueryCandidate<'a>>,
    ) -> Vec<&'a Tag> {
        // Score all candidates
        let mut scored: Vec<(f64, &Tag)> = candidates
            .iter()
            .map(|c| (self.score_site(c), c.tag))
            .collect();

        // Sort by score (descending - highest first)
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Apply gated threshold dropout
        let gated: Vec<(f64, &Tag)> = scored
            .into_iter()
            .filter(|(score, _)| *score >= self.coords.gated_threshold)
            .collect();

        // Apply marginal utility floor
        // Stop when score drops below the learned "price"
        let selected: Vec<&Tag> = gated
            .into_iter()
            .take_while(|(score, _)| *score >= self.coords.marginal_utility_floor)
            .map(|(_, tag)| tag)
            .collect();

        selected
    }

    /// Determine the number of wavefront generations based on batch_latency_bias.
    ///
    /// ## Wavefront Execution Model
    ///
    /// Pure batch (1 wave) is fast but ignores sequential information gain.
    /// Pure sequential (3+ waves) is intelligent but slow.
    /// Compromise: parameterize the trade-off.
    ///
    /// - `batch_latency_bias = 0.0`: 3 waves (sequential, smart)
    ///   * Gen 1 (Spine): Query high-centrality roots
    ///   * Gen 2 (Frontier): Re-score with updated type cache
    ///   * Gen 3 (Fill): Cleanup remaining uncertainty
    ///
    /// - `batch_latency_bias = 1.0`: 1 wave (batch, fast)
    ///   * Query everything in parallel, no re-scoring
    ///
    /// - `batch_latency_bias ∈ (0, 1)`: Interpolate
    ///   * 0.0 - 0.5: 3 waves
    ///   * 0.5 - 0.8: 2 waves
    ///   * 0.8 - 1.0: 1 wave
    ///
    /// ## Design Rationale
    ///
    /// This makes the execution strategy a **continuous coordinate**.
    /// L1/L2 can discover the optimal latency/intelligence trade-off for each corpus.
    ///
    /// High-churn codebases might prefer 1 wave (speed matters, graph changes fast).
    /// Stable codebases might prefer 3 waves (graph is reliable, intelligence pays off).
    pub fn num_generations(&self) -> usize {
        let bias = self.coords.batch_latency_bias.clamp(0.0, 1.0);

        // Map bias to generation count
        // 0.0 → 3 waves (fully sequential)
        // 0.5 → 2 waves (moderate)
        // 1.0 → 1 wave (fully batched)
        if bias < 0.5 {
            3
        } else if bias < 0.8 {
            2
        } else {
            1
        }
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
            interaction_mixing: 0.0, // Pure additive
            ..Default::default()
        };

        let engine = PolicyEngine::new(coords);
        let tag = make_test_tag("foo", TagKind::Def);

        let candidate = QueryCandidate {
            tag: &tag,
            pagerank: 0.5,
            heuristic_confidence: 0.2, // High uncertainty (0.8)
            coherence: 5.0,
            is_root: true,
        };

        let score = engine.score_site(&candidate);

        // Should be additive combination
        // ln(0.5) + 0.8 + ln(6) + 1.0
        let expected = 0.5_f64.ln() + 0.8 + 6.0_f64.ln() + 1.0;
        assert!((score - expected).abs() < 0.01, "score={}, expected={}", score, expected);
    }

    #[test]
    fn test_score_site_multiplicative() {
        // Pure multiplicative mode (interaction_mixing = 1.0)
        let coords = LspPolicyCoordinates {
            weight_centrality: 1.0,
            weight_uncertainty: 1.0,
            weight_coherence: 1.0,
            weight_causality: 1.0,
            interaction_mixing: 1.0, // Pure multiplicative
            ..Default::default()
        };

        let engine = PolicyEngine::new(coords);
        let tag = make_test_tag("bar", TagKind::Ref);

        let candidate = QueryCandidate {
            tag: &tag,
            pagerank: 0.5,
            heuristic_confidence: 0.2, // Uncertainty = 0.8
            coherence: 5.0,
            is_root: false,
        };

        let score = engine.score_site(&candidate);

        // Should be multiplicative: ln(pagerank * uncertainty * (1+coherence))
        let expected = (0.5 * 0.8 * 6.0).ln();
        assert!((score - expected).abs() < 0.01, "score={}, expected={}", score, expected);
    }

    #[test]
    fn test_select_wavefront_gated_threshold() {
        let coords = LspPolicyCoordinates {
            weight_centrality: 1.0,
            weight_uncertainty: 1.0,
            weight_coherence: 0.0,
            weight_causality: 0.0,
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
                heuristic_confidence: 0.1, // High uncertainty
                coherence: 0.0,
                is_root: true,
            },
            QueryCandidate {
                tag: &tag2,
                pagerank: 0.1,
                heuristic_confidence: 0.9, // Low uncertainty
                coherence: 0.0,
                is_root: false,
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
                heuristic_confidence: 0.1,
                coherence: 0.0,
                is_root: true,
            },
            QueryCandidate {
                tag: &tag2,
                pagerank: 0.5,
                heuristic_confidence: 0.5,
                coherence: 0.0,
                is_root: false,
            },
        ];

        let selected = engine.select_wavefront(candidates);

        // Only candidate exceeding marginal utility floor
        assert!(selected.len() >= 1);
        assert_eq!(selected[0].name.as_ref(), "top");
    }

    #[test]
    fn test_num_generations_mapping() {
        let coords_sequential = LspPolicyCoordinates {
            batch_latency_bias: 0.0, // Fully sequential
            ..Default::default()
        };
        let engine_seq = PolicyEngine::new(coords_sequential);
        assert_eq!(engine_seq.num_generations(), 3);

        let coords_moderate = LspPolicyCoordinates {
            batch_latency_bias: 0.6,
            ..Default::default()
        };
        let engine_mod = PolicyEngine::new(coords_moderate);
        assert_eq!(engine_mod.num_generations(), 2);

        let coords_batched = LspPolicyCoordinates {
            batch_latency_bias: 1.0, // Fully batched
            ..Default::default()
        };
        let engine_batch = PolicyEngine::new(coords_batched);
        assert_eq!(engine_batch.num_generations(), 1);
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
            heuristic_confidence: 0.5,
            coherence: 0.0,
            is_root: true, // Definition
        };

        let candidate_leaf = QueryCandidate {
            tag: &tag_leaf,
            pagerank: 0.5,
            heuristic_confidence: 0.5,
            coherence: 0.0,
            is_root: false, // Reference
        };

        let score_root = engine.score_site(&candidate_root);
        let score_leaf = engine.score_site(&candidate_leaf);

        // Root should score higher due to causality weight
        assert!(score_root > score_leaf);
        assert!((score_root - 10.0).abs() < 0.01); // 1.0 * 10.0
        assert!((score_leaf - 0.0).abs() < 0.01);  // 0.0 * 10.0
    }
}
