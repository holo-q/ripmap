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

use super::coordinates::PipelineCoordinates;
use super::graph::{CallGraph, FunctionId};
use super::lsp_strategy::{LspStrategy, LspTypeCache};
use super::resolver::{CallResolver, ResolutionStats};
use super::strategies::*;
use crate::lsp::{LspClient, MockClient, PolicyEngine, QueryCandidate, TypeResolver};
use crate::ranking::PageRanker;
use crate::types::{RankingConfig, Tag};
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
/// If type resolver is not provided (None), the pipeline falls back to heuristic-only
/// resolution. The final pass still runs, but without LSP/oracle support.
///
/// # Oracle Bootstrap Training
///
/// For fast policy training, use `with_mock()` to attach a MockClient loaded from
/// a pre-computed oracle cache. This enables 10^6x faster training iterations:
///
/// ```ignore
/// // Load oracle cache from `ty dump-types` output
/// let oracle = MockClient::from_oracle_file("oracle_cache.json")?;
/// let pipeline = Pipeline::new(coords).with_mock(Arc::new(oracle));
///
/// // Train policy with instant lookups (no subprocess overhead)
/// for episode in 0..10000 {
///     let (graph, stats) = pipeline.build_graph(&tags);
///     // Fast iterations enable rapid policy convergence
/// }
/// ```
pub struct Pipeline {
    coords: PipelineCoordinates,
    /// Type resolver: either LspClient (production) or MockClient (training).
    /// Using trait object enables polymorphism for oracle bootstrap training.
    type_resolver: Option<Arc<dyn TypeResolver + Send + Sync>>,
}

impl Pipeline {
    /// Create a new pipeline with the given coordinates.
    ///
    /// Without a type resolver, this provides baseline heuristic-only resolution (~14%).
    pub fn new(coords: PipelineCoordinates) -> Self {
        Self {
            coords,
            type_resolver: None,
        }
    }

    /// Attach an LSP client for 80%+ resolution in production.
    ///
    /// The client will be used to resolve high-value query sites identified
    /// by the policy engine. This uses real subprocess calls to `ty server`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let lsp = Arc::new(LspClient::new());
    /// let pipeline = Pipeline::new(coords).with_lsp(lsp);
    /// let (graph, stats) = pipeline.build_graph(&tags);
    /// ```
    pub fn with_lsp(mut self, client: Arc<LspClient>) -> Self {
        self.type_resolver = Some(client as Arc<dyn TypeResolver + Send + Sync>);
        self
    }

    /// Attach a MockClient for oracle bootstrap training.
    ///
    /// This enables fast policy training with instant lookups from a pre-cached
    /// oracle file (generated via `ty dump-types`). Training runs are 10^6x faster
    /// than using real LSP, enabling rapid iteration during policy optimization.
    ///
    /// # Oracle Bootstrap Protocol
    ///
    /// 1. **Build Oracle (Offline)**: Run `ty dump-types` on entire corpus → JSON cache
    /// 2. **Train Policy (Fast)**: Load cache into MockClient, train with instant lookups
    /// 3. **Deploy (Production)**: Use trained policy with real LspClient
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Phase 1: Offline - build perfect oracle cache
    /// // $ ty dump-types src/**/*.py > oracle_cache.json
    ///
    /// // Phase 2: Fast training with instant lookups
    /// let oracle = MockClient::from_oracle_file("oracle_cache.json")?;
    /// let pipeline = Pipeline::new(coords).with_mock(Arc::new(oracle));
    ///
    /// for episode in 0..10000 {
    ///     let (graph, stats) = pipeline.build_graph(&tags);
    ///     // Instant lookups, no I/O overhead
    ///     // Policy learns economy: which positions are worth querying
    /// }
    /// ```
    ///
    /// # Training Efficiency
    ///
    /// - Real LSP: 100-500ms per query → 28 hours for 10K episodes
    /// - MockClient: nanoseconds per query → minutes for 10K episodes
    ///
    /// The policy still learns economy because we penalize query count in the
    /// reward function. It doesn't matter that lookups are instant - what matters
    /// is learning *which* positions maximize information gain per query.
    pub fn with_mock(mut self, mock: Arc<MockClient>) -> Self {
        self.type_resolver = Some(mock as Arc<dyn TypeResolver + Send + Sync>);
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
        // Compute centrality on shadow graph to identify query targets.
        // Uses production PageRanker with depth-aware personalization and edge
        // confidence weighting. High-rank functions are hubs; low-rank are leaves.
        let shadow_ranks = self.compute_pagerank(&shadow_graph);

        // === PHASE 3: TYPE RESOLUTION POLICY SELECTION (if resolver available) ===
        // Whether using LspClient (production) or MockClient (training), the policy
        // engine selects high-value query sites and the resolver provides instant answers.
        let (type_cache, lsp_latency_ms, lsp_utilization): (Arc<LspTypeCache>, u64, f64) =
            if let Some(ref resolver) = self.type_resolver {
                use std::time::Instant;

                // Generate QueryCandidates from shadow graph + PageRank
                // This bridges shadow graph analysis with LSP query execution
                let candidates = self.generate_query_candidates(tags, &shadow_graph, &shadow_ranks);

                if candidates.is_empty() {
                    // No candidates - skip LSP phase
                    (Arc::new(LspTypeCache::new()), 0, 0.0)
                } else {
                    // Use PolicyEngine to select wavefront (dissolved decision tree)
                    let policy = PolicyEngine::new(self.coords.lsp_policy.clone());
                    let selected_tags = policy.select_wavefront(candidates);

                    // Convert tags to query format (file, line, col)
                    // LSP uses 0-based coordinates, tags use 1-based, so convert
                    let queries: Vec<(String, u32, u32)> = selected_tags
                        .iter()
                        .map(|tag| {
                            (
                                tag.rel_fname.to_string(),
                                tag.line.saturating_sub(1), // Convert to 0-based
                                0, // Column - TODO: Extract from tag metadata
                            )
                        })
                        .collect();

                    let queries_issued = queries.len();

                    // Issue queries via resolver (measure latency)
                    // This is instant for MockClient (training), slow for LspClient (production)
                    let start = Instant::now();
                    let results = resolver.resolve_batch(&queries);
                    let latency_ms = start.elapsed().as_millis() as u64;

                    // Build type cache from results
                    let type_cache = Arc::new(LspTypeCache::from_lsp_results(&results));

                    // Calculate utilization (successful resolutions / queries issued)
                    // This is a training signal: low utilization means the policy is wasting queries
                    let utilization = if queries_issued > 0 {
                        results.len() as f64 / queries_issued as f64
                    } else {
                        0.0
                    };

                    (type_cache, latency_ms, utilization)
                }
            } else {
                // No type resolver - empty cache, heuristic-only mode
                (Arc::new(LspTypeCache::new()), 0, 0.0)
            };

        // === PHASE 4: FINAL PASS (Precision-Optimized) ===
        // Build the actual graph with conservative heuristics + LSP truth
        // LspStrategy MUST be first - strategies are tried in order and we want LSP
        // to have highest priority. The resolver picks the highest confidence candidate,
        // but being first means LSP gets first crack at resolution.
        let final_resolver = CallResolver::new()
            .with_coordinates(self.coords.final_strategy.clone())
            .with_strategy(Box::new(LspStrategy::new(type_cache.clone()))) // HIGHEST priority
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
    /// Uses the real PageRanker from the ranking module, which implements:
    /// - Standard PageRank with damping factor 0.85
    /// - Power iteration until convergence (epsilon < 1e-8, max 100 iterations)
    /// - Depth-aware personalization (root files get higher base weight)
    /// - Edge confidence weighting for more accurate importance
    ///
    /// This is the "importance" signal for LSP query site selection - high-rank
    /// functions (hubs) are prioritized for LSP queries since resolving them
    /// provides maximum information gain for downstream resolution.
    fn compute_pagerank(&self, graph: &CallGraph) -> HashMap<FunctionId, f64> {
        // Use the production PageRanker from ranking module
        // Default config provides sensible depth-aware personalization:
        // - Root files: weight 1.0
        // - Moderate depth: weight 0.5
        // - Deep files: weight 0.1
        // - Vendor code: weight 0.01
        let config = RankingConfig::default();
        let ranker = PageRanker::new(config);
        ranker.compute_function_ranks(graph)
    }

    /// Compute Spearman rank correlation between two PageRank vectors.
    ///
    /// Measures how well the shadow graph predicts final importance ranking.
    /// High correlation (→1.0) means shadow graph captures true structure.
    /// Low correlation (→0.0) means shadow graph is too noisy.
    ///
    /// # Algorithm
    ///
    /// Spearman's ρ (rho) computes correlation between rank orderings rather than
    /// raw values. This makes it robust to non-linear transformations of the PageRank
    /// scores while still capturing whether shadow and final graphs agree on relative
    /// importance.
    ///
    /// Formula: ρ = 1 - (6 * Σd²) / (n * (n² - 1))
    /// where d = difference in ranks for each function
    ///
    /// Ties are handled by averaging ranks (e.g., if positions 2,3,4 all have same
    /// value, each gets rank 3.0).
    ///
    /// # Returns
    ///
    /// Correlation coefficient in [-1.0, 1.0]:
    /// * 1.0: Perfect positive correlation (shadow perfectly predicts final order)
    /// * 0.0: No correlation (shadow is noise)
    /// * -1.0: Perfect negative correlation (shadow inverts the order)
    ///
    /// Returns 0.0 if fewer than 2 common keys (correlation undefined).
    fn compute_rank_correlation(
        &self,
        shadow: &HashMap<FunctionId, f64>,
        final_: &HashMap<FunctionId, f64>,
    ) -> f64 {
        // Get common keys (functions present in both graphs)
        // Only compare functions that exist in both shadow and final passes
        let common_keys: Vec<&FunctionId> =
            shadow.keys().filter(|k| final_.contains_key(*k)).collect();

        let n = common_keys.len();
        if n < 2 {
            // Spearman correlation undefined for < 2 points
            return 0.0;
        }

        // Extract PageRank values for common keys
        let shadow_values: Vec<f64> = common_keys.iter().map(|k| shadow[*k]).collect();
        let final_values: Vec<f64> = common_keys.iter().map(|k| final_[*k]).collect();

        // Convert values to ranks (handles ties by averaging)
        let shadow_ranks = compute_ranks(&shadow_values);
        let final_ranks = compute_ranks(&final_values);

        // Compute sum of squared differences in ranks
        let d_squared_sum: f64 = shadow_ranks
            .iter()
            .zip(final_ranks.iter())
            .map(|(r1, r2)| (r1 - r2).powi(2))
            .sum();

        // Spearman's rho formula
        1.0 - (6.0 * d_squared_sum) / (n as f64 * ((n * n) as f64 - 1.0))
    }

    /// Generate query candidates from shadow graph for LSP policy selection.
    ///
    /// Converts shadow graph nodes + PageRank scores into `QueryCandidate` structs
    /// that the `PolicyEngine` can score and select for LSP queries.
    ///
    /// This is the critical bridge between the shadow graph (structural analysis)
    /// and the LSP query execution (ground truth resolution). The candidates encode
    /// multiple signals that the policy engine uses to decide which queries are
    /// most valuable:
    ///
    /// - **PageRank**: Structural importance from shadow graph connectivity
    /// - **PageRank Percentile**: Normalized rank for gradient stability
    /// - **Heuristic Confidence**: How certain the shadow resolution is (placeholder: 0.5)
    /// - **Coherence**: Receiver sharing potential (placeholder: 1.0)
    /// - **Is Root**: Whether this is a definition (enables type inference) vs reference
    /// - **Bridge Score**: Betweenness centrality proxy (placeholder: 0.0)
    ///
    /// # Algorithm
    ///
    /// 1. Filter tags to call references (things we might query LSP for)
    /// 2. Compute PageRank percentiles for normalization
    /// 3. Match tags to shadow graph nodes via FunctionId
    /// 4. Extract PageRank scores (default 0.0 if not in shadow graph)
    /// 5. Package into QueryCandidate structs with placeholder signals
    ///
    /// # Future Work
    ///
    /// Currently some signals use placeholder values:
    /// - `heuristic_confidence`: Should come from shadow resolution stats
    /// - `coherence`: Should compute receiver sharing from call graph
    /// - `bridge_score`: Should compute betweenness centrality
    ///
    /// These will be implemented as the pipeline matures and we measure their
    /// impact on policy quality.
    ///
    /// # Parameters
    ///
    /// * `tags`: All tags in the codebase
    /// * `shadow_graph`: The recall-optimized graph built in Phase 1
    /// * `shadow_ranks`: PageRank scores keyed by FunctionId
    ///
    /// # Returns
    ///
    /// Vector of `QueryCandidate` structs ready for policy scoring.
    fn generate_query_candidates<'a>(
        &self,
        tags: &'a [Tag],
        _shadow_graph: &CallGraph,
        shadow_ranks: &HashMap<FunctionId, f64>,
    ) -> Vec<QueryCandidate<'a>> {
        // Filter to call references (things we might query LSP about)
        // We want Ref tags since those are the call sites that need resolution
        let call_refs: Vec<&Tag> = tags.iter().filter(|t| t.kind.is_reference()).collect();

        if call_refs.is_empty() {
            return vec![];
        }

        // Compute percentile ranks for normalization
        // This enables the policy engine to blend raw PageRank with normalized percentiles
        // for gradient stability (centrality_normalization coordinate)
        let mut rank_values: Vec<f64> = shadow_ranks.values().cloned().collect();
        rank_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let percentile = |v: f64| -> f64 {
            if rank_values.is_empty() {
                return 0.5;
            }
            let pos = rank_values
                .iter()
                .position(|&x| x >= v)
                .unwrap_or(rank_values.len());
            pos as f64 / rank_values.len() as f64
        };

        // Convert tags to candidates
        call_refs
            .into_iter()
            .map(|tag| {
                // Construct FunctionId to match PageRank map keys
                // Use tag's location info to build the ID
                let func_id = FunctionId::new(tag.rel_fname.clone(), tag.name.clone(), tag.line);

                // Look up PageRank score for this function
                let pagerank = shadow_ranks.get(&func_id).cloned().unwrap_or(0.0);

                QueryCandidate {
                    tag,
                    pagerank,
                    pagerank_percentile: Some(percentile(pagerank)),
                    // TODO: Get heuristic_confidence from shadow resolution stats
                    // This should measure how certain the shadow strategies were about
                    // this call site's resolution. For now, 0.5 = maximum uncertainty.
                    heuristic_confidence: 0.5,
                    // TODO: Compute coherence (receiver sharing)
                    // This measures how many call sites share the same receiver.
                    // Resolving `user: User` instantly resolves `user.name`, `user.id`, etc.
                    // For now, 1.0 = assume no sharing (each site independent).
                    coherence: 1.0,
                    // is_root: Definitions enable downstream type inference (causality signal)
                    // References depend on upstream resolution
                    is_root: tag.kind.is_definition(),
                    // TODO: Compute bridge_score (betweenness centrality)
                    // This measures how many shortest paths pass through this node.
                    // High bridge score = critical for graph connectivity.
                    // For now, 0.0 = assume no bridge importance.
                    bridge_score: 0.0,
                }
            })
            .collect()
    }
}

/// Convert values to fractional ranks (1-based, averaged for ties).
///
/// # Algorithm
///
/// 1. Create indexed pairs (original_index, value)
/// 2. Sort by value
/// 3. Scan for tie groups (consecutive equal values)
/// 4. Assign average rank to each tie group
/// 5. Return ranks in original order
///
/// # Example
///
/// ```text
/// Values: [10, 30, 20, 30]
/// Sorted: [(0,10), (2,20), (1,30), (3,30)]
/// Ranks:  [1.0,   2.0,    3.5,    3.5]
/// Output: [1.0, 3.5, 2.0, 3.5]  (in original order)
/// ```
///
/// # Tie Handling
///
/// When multiple values are equal, they receive the average of the ranks they
/// would occupy. For example, if positions 3 and 4 both have value 30, they
/// each get rank (3+4)/2 = 3.5.
fn compute_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();

    // Create (original_index, value) pairs and sort by value
    let mut indexed: Vec<(usize, f64)> = values.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Assign ranks, averaging ties
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        // Find the extent of the current tie group
        let mut j = i;
        while j < n - 1 && (indexed[j].1 - indexed[j + 1].1).abs() < 1e-10 {
            j += 1;
        }

        // Compute average rank for this tie group
        // Ranks are 1-based: positions [i..=j] get ranks [(i+1)..(j+1)]
        // Average rank = (sum of ranks) / count = ((i+1) + (j+1)) / 2
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;

        // Assign average rank to all members of tie group
        for k in i..=j {
            ranks[indexed[k].0] = avg_rank;
        }

        i = j + 1;
    }

    ranks
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

    #[test]
    fn test_pipeline_with_mock_client() {
        // Test oracle bootstrap training workflow:
        // Pipeline should accept MockClient for instant type resolution during training
        let tags = vec![
            make_def("test.py", "main", 1),
            make_def("test.py", "helper", 10),
            make_call("test.py", "helper", 5), // main calls helper at line 5
        ];

        let coords = PipelineCoordinates::default();
        let mock = Arc::new(MockClient::empty());
        let pipeline = Pipeline::new(coords).with_mock(mock);

        // Should work without panicking - MockClient provides instant (empty) lookups
        let (graph, _stats) = pipeline.build_graph(&tags);

        // Graph should build successfully even with empty oracle cache
        assert_eq!(graph.function_count(), 2);

        // This validates the oracle bootstrap training path:
        // 1. Create empty MockClient (or load from oracle_cache.json)
        // 2. Attach to Pipeline via with_mock()
        // 3. Run build_graph() → instant type resolutions (no subprocess overhead)
        // 4. Train policy with 10^6x faster iterations
    }

    #[test]
    fn test_spearman_correlation() {
        // Test Spearman correlation calculation with perfect positive correlation
        let coords = PipelineCoordinates::default();
        let pipeline = Pipeline::new(coords);

        let mut shadow = HashMap::new();
        let mut final_ = HashMap::new();

        // Perfect correlation: same relative ordering
        for i in 1..=5 {
            let id = FunctionId::new(Arc::from("test.py"), Arc::from(format!("f{}", i)), i as u32);
            shadow.insert(id.clone(), i as f64 * 0.1);
            final_.insert(id, i as f64 * 0.2); // Same ordering, different scale
        }

        let rho = pipeline.compute_rank_correlation(&shadow, &final_);
        assert!((rho - 1.0).abs() < 0.01, "Expected rho ≈ 1.0, got {}", rho);
    }
}
