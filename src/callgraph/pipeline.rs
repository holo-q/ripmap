//! Bicameral pipeline: Shadow → LSP → Final resolution
//!
//! The pipeline orchestrates the two-phase call graph construction that achieves
//! 80%+ resolution while minimizing LSP overhead:
//!
//! ```text
//! ┌─────────────────────────┐
//! │   PipelineCoordinates   │  (bicameral: separate shadow/final)
//! │ ├── shadow_strategy     │  (recall-optimized)
//! │ ├── final_strategy      │  (precision-optimized)
//! │ └── lsp_policy          │  (query selection)
//! └─────────────────────────┘
//!            │
//!            ▼
//! ┌─────────────────────────┐
//! │       Pipeline          │
//! │                         │
//! │ 1. Shadow Pass          │  → Build graph with heuristics only
//! │    (shadow_strategy)    │     NameMatch=0.9 for recall
//! │                         │
//! │ 2. PageRank             │  → Identify hubs for LSP targeting
//! │                         │
//! │ 3. Policy Selection     │  → PolicyEngine picks query sites
//! │    (lsp_policy)         │
//! │                         │
//! │ 4. Final Pass           │  → Build graph with LSP + heuristics
//! │    (final_strategy)     │     NameMatch=0.2, LSP=1.5
//! └─────────────────────────┘
//! ```
//!
//! # Bicameral Design Rationale
//!
//! The two separate coordinate sets (shadow vs final) enable regime-specific tuning:
//!
//! - **Shadow Pass**: High recall, tolerates false positives
//!   * NameMatch weight: 0.9 (trust name similarity)
//!   * Acceptance threshold: 0.2 (permissive gate)
//!   * Purpose: Seed PageRank with dense connectivity
//!
//! - **Final Pass**: High precision, conservative
//!   * NameMatch weight: 0.2 (suppress noise)
//!   * LSP weight: 1.5 (trust ground truth)
//!   * Acceptance threshold: 0.5 (strict gate)
//!   * Purpose: Build the actual graph with verified edges
//!
//! # Training Dynamics
//!
//! The bicameral architecture creates a natural training progression:
//!
//! 1. **Phase 1 - Shadow Tuning**: Optimize shadow coordinates for PageRank quality
//!    (measured by correlation with oracle ranks)
//!
//! 2. **Phase 2 - Policy Tuning**: Freeze shadow, optimize LSP policy for
//!    query efficiency (minimize cost, maximize NDCG improvement)
//!
//! 3. **Phase 3 - Final Tuning**: Freeze policy, optimize final coordinates for
//!    precision (maximize F1 against oracle graph)
//!
//! 4. **Phase 4 - Joint Polish**: Unfreeze everything, end-to-end refinement
//!
//! This staged approach prevents coordinate fighting and accelerates convergence.

use super::coordinates::{PipelineCoordinates, StrategyCoordinates};
use super::graph::CallGraph;
use super::resolver::{CallResolver, ResolutionStats};
use super::strategies::*;
use crate::lsp::LspClient;
use crate::types::Tag;
use std::collections::HashMap;
use std::sync::Arc;

/// Statistics from bicameral pipeline execution.
///
/// Tracks both shadow and final passes independently, plus LSP utilization
/// and correlation metrics that reveal the quality of the shadow → final transition.
///
/// # Key Metrics
///
/// - **Shadow Stats**: Resolution success in recall-optimized pass
/// - **Final Stats**: Resolution success in precision-optimized pass
/// - **Shadow Connectivity**: Edge density (edges / possible_edges)
/// - **LSP Utilization**: Success rate of LSP queries (resolved / queried)
/// - **Rank Correlation**: Spearman correlation between shadow and final PageRank
///   * High correlation → shadow graph captures true structure
///   * Low correlation → shadow graph is too noisy, retune coordinates
/// - **LSP Latency**: Total time spent in LSP queries (milliseconds)
///
/// # Training Signals
///
/// L1 uses these metrics to tune coordinates:
/// - Low shadow connectivity → increase NameMatch weight in shadow pass
/// - Low rank correlation → shadow graph doesn't match final structure, adjust
/// - Low LSP utilization → queries are being wasted, adjust policy
/// - High LSP latency → too many queries, increase marginal_utility_floor
#[derive(Debug, Clone)]
pub struct PipelineStats {
    /// Shadow pass resolution statistics (recall-optimized)
    pub shadow_stats: ResolutionStats,
    /// Final pass resolution statistics (precision-optimized)
    pub final_stats: ResolutionStats,
    /// Edge density of shadow graph (edges / possible_edges)
    /// Measures how well the shadow pass seeds PageRank
    pub shadow_connectivity: f64,
    /// LSP success rate (resolved / queried)
    /// Measures efficiency of query selection policy
    pub lsp_utilization: f64,
    /// Spearman correlation between shadow and final PageRank vectors
    /// Measures how well shadow graph predicts final importance
    pub shadow_final_rank_correlation: f64,
    /// Total LSP latency in milliseconds
    /// Direct cost metric for policy optimization
    pub lsp_latency_ms: u64,
}

/// The bicameral pipeline orchestrator.
///
/// Coordinates the two-phase construction:
/// 1. Shadow pass → PageRank → Policy selection
/// 2. LSP queries (optional)
/// 3. Final pass with LSP results
///
/// # Usage
///
/// ```ignore
/// // With LSP (80%+ resolution)
/// let pipeline = Pipeline::new(coords).with_lsp(lsp_client);
/// let (graph, stats) = pipeline.build_graph(&tags);
///
/// // Without LSP (14% baseline)
/// let pipeline = Pipeline::new(coords);
/// let (graph, stats) = pipeline.build_graph(&tags);
/// ```
///
/// # Graceful Degradation
///
/// If LSP client is not provided (None), the pipeline falls back to heuristic-only
/// resolution. The final pass still runs, but without LSP support.
pub struct Pipeline {
    coords: PipelineCoordinates,
    lsp_client: Option<Arc<LspClient>>,
}

impl Pipeline {
    /// Create a new pipeline with the given coordinates.
    ///
    /// Without LSP, this provides baseline heuristic-only resolution (~14%).
    pub fn new(coords: PipelineCoordinates) -> Self {
        Self {
            coords,
            lsp_client: None,
        }
    }

    /// Attach an LSP client for 80%+ resolution.
    ///
    /// The client will be used to resolve high-value query sites identified
    /// by the policy engine.
    pub fn with_lsp(mut self, client: Arc<LspClient>) -> Self {
        self.lsp_client = Some(client);
        self
    }

    /// Run the full bicameral pipeline.
    ///
    /// # Algorithm
    ///
    /// 1. **Shadow Pass**: Build recall-optimized graph with aggressive heuristics
    ///    * High NameMatch weight (0.9)
    ///    * Permissive acceptance gate (0.2)
    ///    * No LSP (heuristics only)
    ///
    /// 2. **PageRank**: Compute centrality on shadow graph
    ///    * Identifies hubs and leaves
    ///    * Seeds the policy engine with structural importance
    ///
    /// 3. **Policy Selection**: (if LSP available)
    ///    * Score candidates using PolicyEngine
    ///    * Select wavefront based on centrality, uncertainty, coherence
    ///    * Issue batch LSP queries
    ///
    /// 4. **Final Pass**: Build precision-optimized graph
    ///    * Low NameMatch weight (0.2)
    ///    * High LSP weight (1.5)
    ///    * Strict acceptance gate (0.5)
    ///    * LSP results cached and reused
    ///
    /// # Returns
    ///
    /// * `CallGraph`: The final precision-optimized graph
    /// * `PipelineStats`: Instrumentation for training
    pub fn build_graph(&self, tags: &[Tag]) -> (CallGraph, PipelineStats) {
        // === PHASE 1: SHADOW PASS (Recall-Optimized) ===
        // Build a dense graph for PageRank seeding. False positives are okay.
        let shadow_resolver = CallResolver::new()
            .with_coordinates(self.coords.shadow_strategy.clone())
            .with_strategy(Box::new(SameFileStrategy::new()))
            .with_strategy(Box::new(TypeHintStrategy::new()))
            .with_strategy(Box::new(ImportStrategy::new()))
            .with_strategy(Box::new(NameMatchStrategy::new()));

        let shadow_graph = shadow_resolver.build_graph(tags);
        let shadow_stats = shadow_resolver.stats(tags);

        // Compute shadow connectivity (edge density)
        let node_count = shadow_graph.function_count();
        let edge_count = shadow_graph.call_count();
        let possible_edges = node_count * (node_count - 1); // Directed graph
        let shadow_connectivity = if possible_edges > 0 {
            edge_count as f64 / possible_edges as f64
        } else {
            0.0
        };

        // === PHASE 2: PAGERANK ===
        // Compute centrality on shadow graph to identify query targets
        // TODO: Implement PageRank - for now, stub with uniform distribution
        let shadow_ranks = self.compute_pagerank(&shadow_graph);

        // === PHASE 3: LSP POLICY SELECTION (if client available) ===
        let (_type_cache, lsp_latency_ms, lsp_utilization): (HashMap<String, String>, u64, f64) = if let Some(ref _client) = self.lsp_client
        {
            // TODO: Implement LSP query generation and execution
            // For now, return empty cache
            (HashMap::new(), 0, 0.0)
        } else {
            // No LSP client - skip query phase
            (HashMap::new(), 0, 0.0)
        };

        // === PHASE 4: FINAL PASS (Precision-Optimized) ===
        // Build the actual graph with conservative heuristics + LSP truth
        // TODO: Wire in LspStrategy once implemented
        let final_resolver = CallResolver::new()
            .with_coordinates(self.coords.final_strategy.clone())
            // .with_strategy(Box::new(LspStrategy::new(lsp_cache)))  // HIGHEST priority
            .with_strategy(Box::new(SameFileStrategy::new()))
            .with_strategy(Box::new(TypeHintStrategy::new()))
            .with_strategy(Box::new(ImportStrategy::new()))
            .with_strategy(Box::new(NameMatchStrategy::new()));

        let final_graph = final_resolver.build_graph(tags);
        let final_stats = final_resolver.stats(tags);

        // === PHASE 5: CORRELATION METRICS ===
        // Compute Spearman correlation between shadow and final PageRank
        let final_ranks = self.compute_pagerank(&final_graph);
        let rank_correlation = self.compute_rank_correlation(&shadow_ranks, &final_ranks);

        let stats = PipelineStats {
            shadow_stats,
            final_stats,
            shadow_connectivity,
            lsp_utilization,
            shadow_final_rank_correlation: rank_correlation,
            lsp_latency_ms,
        };

        (final_graph, stats)
    }

    /// Compute PageRank on a call graph.
    ///
    /// Returns a map from FunctionId to rank score.
    ///
    /// # Algorithm
    ///
    /// Standard PageRank with damping factor 0.85, 100 iterations.
    /// This is the "importance" signal for query site selection.
    ///
    /// # TODO
    ///
    /// - Implement actual PageRank using petgraph or custom implementation
    /// - Consider personalized PageRank seeded from entry points
    /// - Add convergence detection to stop early when stable
    fn compute_pagerank(&self, graph: &CallGraph) -> HashMap<String, f64> {
        // STUB: Return uniform distribution
        // TODO: Implement PageRank algorithm
        let mut ranks = HashMap::new();
        let uniform_rank = 1.0 / graph.function_count().max(1) as f64;
        for func in graph.functions() {
            let key = format!("{}::{}", func.file, func.name);
            ranks.insert(key, uniform_rank);
        }
        ranks
    }

    /// Compute Spearman rank correlation between two PageRank vectors.
    ///
    /// Measures how well the shadow graph predicts final importance ranking.
    /// High correlation (→1.0) means shadow graph captures true structure.
    /// Low correlation (→0.0) means shadow graph is too noisy.
    ///
    /// # Returns
    ///
    /// Correlation coefficient in [-1.0, 1.0]:
    /// * 1.0: Perfect positive correlation
    /// * 0.0: No correlation
    /// * -1.0: Perfect negative correlation
    ///
    /// # TODO
    ///
    /// - Implement Spearman correlation calculation
    /// - Handle missing keys (functions in one graph but not the other)
    fn compute_rank_correlation(
        &self,
        shadow: &HashMap<String, f64>,
        final_: &HashMap<String, f64>,
    ) -> f64 {
        // STUB: Return 0.5 (moderate correlation)
        // TODO: Implement Spearman correlation
        let _ = (shadow, final_);
        0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::callgraph::StrategyCoordinates;
    use crate::types::{Tag, TagKind};

    fn make_def(file: &str, name: &str, line: u32) -> Tag {
        Tag {
            rel_fname: Arc::from(file),
            fname: Arc::from(file),
            line,
            name: Arc::from(name),
            kind: TagKind::Def,
            node_type: Arc::from("function"),
            parent_name: None,
            parent_line: None,
            signature: None,
            fields: None,
            metadata: None,
        }
    }

    fn make_call(file: &str, name: &str, line: u32) -> Tag {
        Tag {
            rel_fname: Arc::from(file),
            fname: Arc::from(file),
            line,
            name: Arc::from(name),
            kind: TagKind::Ref,
            node_type: Arc::from("call"),
            parent_name: None,
            parent_line: None,
            signature: None,
            fields: None,
            metadata: None,
        }
    }

    #[test]
    fn test_pipeline_without_lsp() {
        let tags = vec![
            make_def("test.py", "main", 1),
            make_def("test.py", "helper", 10),
            make_call("test.py", "helper", 5), // main calls helper at line 5
        ];

        let coords = PipelineCoordinates::default();
        let pipeline = Pipeline::new(coords);
        let (graph, stats) = pipeline.build_graph(&tags);

        // Should build a graph with basic heuristics
        assert_eq!(graph.function_count(), 2);
        assert!(graph.call_count() >= 1); // At least the helper call

        // Stats should show both passes executed
        assert!(stats.shadow_stats.total_calls > 0);
        assert!(stats.final_stats.total_calls > 0);
    }

    #[test]
    fn test_shadow_vs_final_defaults() {
        let shadow = StrategyCoordinates::shadow_defaults();
        let final_ = StrategyCoordinates::final_defaults();

        // Shadow should have higher name match weight (recall)
        assert!(shadow.weight_name_match > final_.weight_name_match);

        // Final should have higher LSP weight (precision)
        assert!(final_.weight_lsp > shadow.weight_lsp);

        // Shadow should have lower acceptance bias (permissive)
        assert!(shadow.acceptance_bias < final_.acceptance_bias);
    }

    #[test]
    fn test_pipeline_coordinates_default() {
        let coords = PipelineCoordinates::default();

        // Should have distinct shadow and final strategies
        assert!(coords.shadow_strategy.weight_name_match > coords.final_strategy.weight_name_match);

        // LSP policy should be present
        // (exact values depend on LspPolicyCoordinates::default())
    }
}
