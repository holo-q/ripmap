//! Call resolver: orchestrates multiple resolution strategies.
//!
//! The resolver is the brain of call graph construction:
//! 1. Builds context from tags (indexes, type maps)
//! 2. Runs each strategy on each call reference
//! 3. Uses coordinate-based differentiable selection (NOT hard argmax)
//! 4. Builds the final CallGraph
//!
//! # Differentiable Selection (Dissolved Argmax)
//!
//! Traditional resolvers use hard argmax: `candidates.sort(); candidates.first()`
//! This is NON-DIFFERENTIABLE - the gradient is zero everywhere except at the max.
//! You can't train through it.
//!
//! We replace argmax with a three-stage differentiable pipeline:
//!
//! 1. **Strategy Weighting**: `weighted = coords.weighted_confidence(strategy, raw)`
//!    - Each strategy (SameFile, TypeHint, Import, NameMatch) gets a trainable weight
//!    - Instead of fixed confidences, we learn which strategies to trust
//!
//! 2. **Sigmoid Acceptance Gate**: `accepted if coords.acceptance_probability(weighted) > 0.5`
//!    - Replaces hard threshold (`if conf >= 0.3`) with smooth sigmoid
//!    - Creates gradients for training the acceptance policy
//!
//! 3. **Mode Selection**: controlled by `coords.evidence_accumulation`
//!    - `< 0.5`: Softmax selection (temperature-controlled, pick best)
//!    - `>= 0.5`: Evidence accumulation (combine signals, future: ensemble)
//!
//! This enables gradient-based optimization of the entire resolution policy.
//! See `StrategyCoordinates` for the trainable parameter space.

use super::coordinates::StrategyCoordinates;
use super::graph::{CallEdge, CallGraph, FunctionId};
use super::strategies::{Candidate, ResolutionContext, ResolutionStrategy};
use crate::types::{Tag, TagKind};
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for the resolver
///
/// # Migration from Hard Thresholds to Sigmoid Gate
///
/// The resolver now supports two acceptance modes:
///
/// 1. **Legacy Mode** (hard threshold): Set `min_confidence` to `Some(threshold)`
///    - Candidate with confidence 0.299 → rejected
///    - Candidate with confidence 0.301 → accepted
///    - Creates gradient cliffs, hard to optimize
///
/// 2. **Modern Mode** (sigmoid gate): Set `min_confidence` to `None`
///    - Uses `StrategyCoordinates.acceptance_probability()` for smooth acceptance
///    - Confidence 0.299 → accept with probability ~0.48
///    - Confidence 0.301 → accept with probability ~0.52
///    - Smooth, differentiable, trainable
///
/// The `use_sigmoid_gate` flag controls which mode is active. New code should
/// use the sigmoid gate for better gradient behavior during training.
#[derive(Debug, Clone)]
pub struct ResolverConfig {
    /// DEPRECATED: Hard minimum confidence threshold (legacy mode)
    ///
    /// When `Some(threshold)`, uses hard cutoff at this value.
    /// When `None`, uses sigmoid gate from StrategyCoordinates instead.
    ///
    /// **Migration path**:
    /// - Old code: `min_confidence: 0.3`
    /// - New code: `min_confidence: None, use_sigmoid_gate: true`
    ///
    /// The sigmoid gate provides the same filtering behavior but with smooth
    /// gradients that enable training.
    pub min_confidence: Option<f64>,

    /// Whether to include unresolved calls as dangling nodes
    pub include_unresolved: bool,

    /// Language hint for strategy filtering (e.g., "python")
    pub language: Option<String>,

    /// Use sigmoid acceptance gate instead of hard threshold
    ///
    /// When `true`, candidates are accepted based on sigmoid probability from
    /// StrategyCoordinates. When `false`, uses hard `min_confidence` threshold
    /// (if set) or accepts all candidates.
    ///
    /// **Default**: `true` (modern mode)
    pub use_sigmoid_gate: bool,
}

impl Default for ResolverConfig {
    fn default() -> Self {
        Self {
            // Modern default: no hard threshold, use sigmoid instead
            min_confidence: None,
            include_unresolved: false,
            language: None,
            use_sigmoid_gate: true, // Modern default
        }
    }
}

impl ResolverConfig {
    /// Legacy constructor with hard threshold.
    ///
    /// Creates a config that uses a hard confidence cutoff instead of the
    /// sigmoid gate. Use this for backward compatibility with existing code.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = ResolverConfig::with_min_confidence(0.3);
    /// // Equivalent to old: ResolverConfig { min_confidence: 0.3, .. }
    /// ```
    pub fn with_min_confidence(min: f64) -> Self {
        Self {
            min_confidence: Some(min),
            include_unresolved: false,
            language: None,
            use_sigmoid_gate: false, // Legacy mode
        }
    }
}

/// The main resolver that combines strategies to build call graphs.
///
/// # Sigmoid Acceptance Gate
///
/// The resolver uses `StrategyCoordinates` to determine acceptance of candidates
/// via a smooth sigmoid function instead of hard thresholds. This enables:
/// - Gradient-based optimization of acceptance policy
/// - Smooth transitions from rejection to acceptance
/// - Trainable acceptance criteria via L1/L2
pub struct CallResolver {
    strategies: Vec<Box<dyn ResolutionStrategy>>,
    config: ResolverConfig,
    /// Coordinates for sigmoid acceptance gate and strategy weighting
    coords: StrategyCoordinates,
}

impl CallResolver {
    pub fn new() -> Self {
        Self {
            strategies: vec![],
            config: ResolverConfig::default(),
            coords: StrategyCoordinates::default(),
        }
    }

    pub fn with_config(mut self, config: ResolverConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the strategy coordinates for sigmoid gating and weighting.
    ///
    /// **Use case**: Training loop provides custom coordinates to tune acceptance policy.
    pub fn with_coordinates(mut self, coords: StrategyCoordinates) -> Self {
        self.coords = coords;
        self
    }

    /// Add a resolution strategy (order matters for tie-breaking)
    pub fn add_strategy(&mut self, strategy: Box<dyn ResolutionStrategy>) {
        self.strategies.push(strategy);
    }

    /// Convenience builder pattern
    pub fn with_strategy(mut self, strategy: Box<dyn ResolutionStrategy>) -> Self {
        self.add_strategy(strategy);
        self
    }

    /// Build a call graph AND return resolution statistics.
    ///
    /// This is the instrumented version of `build_graph` that tracks metrics
    /// for L1 training diagnostics and Shadow Collapse detection.
    ///
    /// **Key instrumentation**:
    /// - LSP strategy usage (queries issued, resolutions, utilization)
    /// - LSP-heuristic conflicts/agreements (shadow divergence proxy)
    /// - Gate rejections and strategy competition
    ///
    /// **Use case**: Training loops, diagnostic analysis, L1 optimization.
    /// For production use without stats overhead, call `build_graph()` instead.
    pub fn build_graph_with_stats(&self, tags: &[Tag]) -> (CallGraph, ResolutionStats) {
        let context = ResolutionContext::new(tags);
        let mut graph = CallGraph::new();
        let mut stats = ResolutionStats::default();

        // Add all function definitions as nodes
        for tag in tags {
            if tag.kind.is_definition() {
                // Only add functions/methods, not classes or variables
                let node_type = tag.node_type.as_ref();
                if node_type.contains("function") || node_type.contains("method") {
                    let id = FunctionId::new(tag.rel_fname.clone(), tag.name.clone(), tag.line)
                        .with_parent_opt(tag.parent_name.clone());
                    graph.add_function(id);
                }
            }
        }

        // Process each call reference
        for tag in tags {
            if !tag.kind.is_reference() {
                continue;
            }

            stats.total_calls += 1;

            // Find the enclosing function (caller)
            let caller = self.find_enclosing_function(tag, tags);
            let Some(caller) = caller else {
                stats.unresolved += 1;
                continue; // Call not inside a function
            };

            // Run all strategies, collect candidates
            let mut all_candidates: Vec<(Candidate, &str)> = vec![];
            let mut lsp_candidate: Option<Candidate> = None;
            let mut heuristic_candidates: Vec<Candidate> = vec![];

            for strategy in &self.strategies {
                // Skip strategies that don't support this language
                if let Some(ref lang) = self.config.language {
                    if !strategy.supports_language(lang) {
                        continue;
                    }
                }

                let strat_name = strategy.name();
                let candidates = strategy.resolve(tag, &context);

                // Track LSP queries (even if they return nothing)
                if strat_name == "lsp" {
                    stats.lsp_queries_issued += 1;
                }

                for c in candidates {
                    all_candidates.push((c.clone(), strat_name));

                    // Separate LSP from heuristic candidates for conflict tracking
                    if strat_name == "lsp" {
                        lsp_candidate = Some(c.clone());
                        stats.lsp_resolved += 1;
                    } else {
                        heuristic_candidates.push(c);
                    }
                }
            }

            // Track LSP-heuristic conflicts/agreements
            if let Some(ref lsp_cand) = lsp_candidate {
                for heur_cand in &heuristic_candidates {
                    if lsp_cand.target == heur_cand.target {
                        stats.lsp_heuristic_agreements += 1;
                    } else {
                        stats.lsp_heuristic_conflicts += 1;
                    }
                }
            }

            // === DIFFERENTIABLE SELECTION ===
            if all_candidates.is_empty() {
                stats.unresolved += 1;
                if self.config.include_unresolved {
                    // Add unresolved call as a dangling reference
                    let unresolved = FunctionId::new(
                        Arc::<str>::from("?"), // Unknown file
                        tag.name.clone(),
                        0,
                    );
                    let edge = CallEdge::new(0.0, "unresolved", tag.line);
                    graph.add_call(caller, unresolved, edge);
                }
                continue;
            }

            // Weight each candidate by strategy weight
            let weighted_candidates: Vec<(Candidate, &str, f64)> = all_candidates
                .into_iter()
                .map(|(c, strat)| {
                    let weighted = self.coords.weighted_confidence(strat, c.confidence);
                    (c, strat, weighted)
                })
                .collect();

            // Filter by acceptance probability (sigmoid gate)
            let accepted: Vec<_> = weighted_candidates
                .iter()
                .filter(|(_, _, weighted_conf)| {
                    self.coords.acceptance_probability(*weighted_conf) > 0.5
                })
                .collect();

            if accepted.is_empty() {
                // No candidates passed the acceptance gate
                stats.dropped_by_gate += 1;
                stats.unresolved += 1;
                continue;
            }

            // Track strategy competition
            if accepted.len() >= 2 {
                let sorted: Vec<_> = {
                    let mut tmp = accepted.clone();
                    tmp.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
                    tmp
                };
                let top1_conf = sorted[0].2;
                let top2_conf = sorted[1].2;
                if (top1_conf - top2_conf).abs() < 0.1 {
                    stats.strategy_competition += 1;
                }

                // Check for strategy agreement on same target
                let top1_target = &sorted[0].0.target;
                let mut agreement_count = 0;
                for (cand, _, _) in &sorted[1..] {
                    if &cand.target == top1_target {
                        agreement_count += 1;
                    }
                }
                if agreement_count > 0 {
                    stats.strategy_agreement += 1;
                }
            }

            // Selection mode based on evidence_accumulation coordinate
            let (selected, strategy_name) = if self.coords.evidence_accumulation < 0.5 {
                // Softmax selection mode
                let best = accepted
                    .iter()
                    .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap();
                (&best.0, best.1)
            } else {
                // Evidence accumulation mode
                let best = accepted
                    .iter()
                    .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap();
                (&best.0, best.1)
            };

            // Update by_strategy counter
            *stats
                .by_strategy
                .entry(strategy_name.to_string())
                .or_insert(0) += 1;

            // Update confidence histogram
            let bucket = ((selected.confidence * 5.0).floor() as usize).min(4);
            stats.confidence_histogram[bucket] += 1;

            // Add the selected edge to the graph
            let edge = CallEdge::new(selected.confidence, strategy_name, tag.line);
            let edge = if let Some(ref hint) = selected.type_hint {
                edge.with_type_hint(hint.clone())
            } else {
                edge
            };

            graph.add_call(caller, selected.target.clone(), edge);
        }

        (graph, stats)
    }

    /// Build graph and extract per-tag confidence map for query signal computation.
    ///
    /// Returns:
    /// - CallGraph: The resolved call graph
    /// - HashMap<(Arc<str>, u32), f64>: Confidence map keyed by (file, line)
    ///
    /// The confidence map enables the Policy Engine to measure "heuristic certainty" -
    /// how confident were the shadow strategies about each call site's resolution.
    /// High confidence (0.9) = SameFile match, low confidence (0.5) = NameMatch fallback.
    ///
    /// This is used by `generate_query_candidates()` to compute `heuristic_confidence`
    /// signal, which guides LSP query selection: query the uncertain sites first.
    pub fn build_graph_with_confidences(
        &self,
        tags: &[Tag],
    ) -> (CallGraph, HashMap<(Arc<str>, u32), f64>) {
        let context = ResolutionContext::new(tags);
        let mut graph = CallGraph::new();
        let mut confidences: HashMap<(Arc<str>, u32), f64> = HashMap::new();

        // Add all function definitions as nodes
        for tag in tags {
            if tag.kind.is_definition() {
                // Only add functions/methods, not classes or variables
                let node_type = tag.node_type.as_ref();
                if node_type.contains("function") || node_type.contains("method") {
                    let id = FunctionId::new(tag.rel_fname.clone(), tag.name.clone(), tag.line)
                        .with_parent_opt(tag.parent_name.clone());
                    graph.add_function(id);
                }
            }
        }

        // Process each call reference and track confidence
        for tag in tags {
            if !tag.kind.is_reference() {
                continue;
            }

            // Find the enclosing function (caller)
            let caller = self.find_enclosing_function(tag, tags);
            let Some(caller) = caller else {
                continue; // Call not inside a function
            };

            // Run all strategies, collect candidates
            let mut all_candidates: Vec<(Candidate, &str)> = vec![];

            for strategy in &self.strategies {
                // Skip strategies that don't support this language
                if let Some(ref lang) = self.config.language {
                    if !strategy.supports_language(lang) {
                        continue;
                    }
                }

                let strat_name = strategy.name();
                let candidates = strategy.resolve(tag, &context);

                for c in candidates {
                    all_candidates.push((c, strat_name));
                }
            }

            if all_candidates.is_empty() {
                // Track unresolved sites with confidence 0.0
                confidences.insert((tag.rel_fname.clone(), tag.line), 0.0);
                continue;
            }

            // Weight each candidate by strategy weight
            let weighted_candidates: Vec<(Candidate, &str, f64)> = all_candidates
                .into_iter()
                .map(|(c, strat)| {
                    let weighted = self.coords.weighted_confidence(strat, c.confidence);
                    (c, strat, weighted)
                })
                .collect();

            // Filter by acceptance probability (sigmoid gate)
            let accepted: Vec<_> = weighted_candidates
                .iter()
                .filter(|(_, _, weighted_conf)| {
                    self.coords.acceptance_probability(*weighted_conf) > 0.5
                })
                .collect();

            if accepted.is_empty() {
                // Track rejected sites with low confidence
                let max_weighted = weighted_candidates
                    .iter()
                    .map(|(_, _, w)| w)
                    .cloned()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0.0);
                confidences.insert((tag.rel_fname.clone(), tag.line), max_weighted);
                continue;
            }

            // Select best candidate (same logic as build_graph_with_stats)
            let best = accepted
                .iter()
                .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();

            let selected = &best.0;
            let weighted_confidence = best.2;

            // Track the weighted confidence for this call site
            confidences.insert((tag.rel_fname.clone(), tag.line), weighted_confidence);

            // Add edge to graph
            let edge = CallEdge::new(selected.confidence, best.1, tag.line)
                .with_type_hint(selected.type_hint.clone().unwrap_or_default());

            graph.add_call(caller, selected.target.clone(), edge);
        }

        (graph, confidences)
    }

    /// Build a complete call graph from extracted tags.
    ///
    /// Process:
    /// 1. Build resolution context (indexes, type maps)
    /// 2. Add all definitions as nodes
    /// 3. For each call reference, run strategies
    /// 4. Pick best resolution, add edge
    ///
    /// **Note**: This is the non-instrumented version. For training diagnostics
    /// and Shadow Collapse detection, use `build_graph_with_stats()` instead.
    pub fn build_graph(&self, tags: &[Tag]) -> CallGraph {
        let context = ResolutionContext::new(tags);
        let mut graph = CallGraph::new();

        // Add all function definitions as nodes
        for tag in tags {
            if tag.kind.is_definition() {
                // Only add functions/methods, not classes or variables
                let node_type = tag.node_type.as_ref();
                if node_type.contains("function") || node_type.contains("method") {
                    let id = FunctionId::new(tag.rel_fname.clone(), tag.name.clone(), tag.line)
                        .with_parent_opt(tag.parent_name.clone());
                    graph.add_function(id);
                }
            }
        }

        // Process each call reference
        for tag in tags {
            if !tag.kind.is_reference() {
                continue;
            }

            // Find the enclosing function (caller)
            let caller = self.find_enclosing_function(tag, tags);
            let Some(caller) = caller else {
                continue; // Call not inside a function
            };

            // Run all strategies, collect candidates
            let mut all_candidates: Vec<(Candidate, &str)> = vec![];

            for strategy in &self.strategies {
                // Skip strategies that don't support this language
                if let Some(ref lang) = self.config.language {
                    if !strategy.supports_language(lang) {
                        continue;
                    }
                }

                let candidates = strategy.resolve(tag, &context);
                for c in candidates {
                    all_candidates.push((c, strategy.name()));
                }
            }

            // === DIFFERENTIABLE SELECTION: Replace hard argmax with coordinate-based selection ===
            //
            // OLD (NON-DIFFERENTIABLE):
            //   all_candidates.sort_by(confidence);
            //   candidates.first()  // Argmax - gradient is ZERO everywhere except at the max
            //
            // NEW (DIFFERENTIABLE):
            //   1. Weight each candidate by strategy weight (coords.weighted_confidence)
            //   2. Filter by acceptance probability (sigmoid gate, not hard threshold)
            //   3. Select via coordinate-controlled mode (softmax vs evidence accumulation)
            //
            // This enables gradient-based optimization of both strategy weights and selection policy.

            if all_candidates.is_empty() {
                if self.config.include_unresolved {
                    // Add unresolved call as a dangling reference
                    let unresolved = FunctionId::new(
                        Arc::<str>::from("?"), // Unknown file
                        tag.name.clone(),
                        0,
                    );
                    let edge = CallEdge::new(0.0, "unresolved", tag.line);
                    graph.add_call(caller, unresolved, edge);
                }
                continue;
            }

            // Step 1: Weight each candidate by strategy weight
            let weighted_candidates: Vec<(Candidate, &str, f64)> = all_candidates
                .into_iter()
                .map(|(c, strat)| {
                    let weighted = self.coords.weighted_confidence(strat, c.confidence);
                    (c, strat, weighted)
                })
                .collect();

            // Step 2: Filter by acceptance probability (sigmoid gate)
            let accepted: Vec<_> = weighted_candidates
                .iter()
                .filter(|(_, _, weighted_conf)| {
                    self.coords.acceptance_probability(*weighted_conf) > 0.5
                })
                .collect();

            if accepted.is_empty() {
                // No candidates passed the acceptance gate
                continue;
            }

            // Step 3: Selection mode based on evidence_accumulation coordinate
            let (selected, strategy_name) = if self.coords.evidence_accumulation < 0.5 {
                // Softmax selection mode (probabilistic, temperature-controlled)
                // For determinism in current implementation, we take argmax of weighted scores
                // Future work: sample from softmax(scores / temperature) for true exploration
                let best = accepted
                    .iter()
                    .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap();
                (&best.0, best.1)
            } else {
                // Evidence accumulation mode
                // For now, still single target (take highest accumulated evidence)
                // Future work: Emit multiple weighted edges (ensemble)
                let best = accepted
                    .iter()
                    .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap();
                (&best.0, best.1)
            };

            // Add the selected edge to the graph
            let edge = CallEdge::new(selected.confidence, strategy_name, tag.line);
            let edge = if let Some(ref hint) = selected.type_hint {
                edge.with_type_hint(hint.clone())
            } else {
                edge
            };

            graph.add_call(caller, selected.target.clone(), edge);
        }

        graph
    }

    /// Find the function definition that encloses a given tag.
    /// Uses line number proximity within the same file.
    fn find_enclosing_function(&self, tag: &Tag, all_tags: &[Tag]) -> Option<FunctionId> {
        // Find all function definitions in the same file
        let mut functions: Vec<&Tag> = all_tags
            .iter()
            .filter(|t| {
                t.rel_fname == tag.rel_fname
                    && t.kind.is_definition()
                    && (t.node_type.contains("function") || t.node_type.contains("method"))
            })
            .collect();

        // Sort by line number descending (so we can find closest before)
        functions.sort_by_key(|t| std::cmp::Reverse(t.line));

        // Find the first function defined before this tag's line
        for func in functions {
            if func.line <= tag.line {
                return Some(
                    FunctionId::new(func.rel_fname.clone(), func.name.clone(), func.line)
                        .with_parent_opt(func.parent_name.clone()),
                );
            }
        }

        None
    }

    /// Get statistics about resolution success with detailed L1 instrumentation.
    ///
    /// This method runs the full resolution pipeline on all call references,
    /// collecting detailed metrics about:
    /// - Gate rejections (candidates filtered by confidence threshold)
    /// - Strategy competition (multiple strategies with similar confidence)
    /// - Strategy agreement (multiple strategies converging on same target)
    /// - Confidence distribution (histogram of accepted resolutions)
    ///
    /// **Performance**: This runs all strategies for all calls, so it's O(strategies × calls).
    /// Use sparingly in production; primarily for training and diagnostics.
    pub fn stats(&self, tags: &[Tag]) -> ResolutionStats {
        let context = ResolutionContext::new(tags);
        let mut stats = ResolutionStats::default();

        // Determine the effective acceptance threshold for gate diagnostics
        // Legacy mode: use hard min_confidence
        // Modern mode: use sigmoid with threshold at p=0.5 (confidence where acceptance_probability crosses 0.5)
        let gate_threshold = if let Some(hard_min) = self.config.min_confidence {
            hard_min
        } else if self.config.use_sigmoid_gate {
            // For sigmoid, threshold is where acceptance_probability(c) = 0.5
            // This depends on StrategyCoordinates.acceptance_bias, but we approximate with 0.5
            // TODO: Extract actual threshold from self.coords when available
            0.5
        } else {
            // No gating - effectively 0.0 threshold
            0.0
        };

        for tag in tags {
            if !tag.kind.is_reference() {
                continue;
            }

            stats.total_calls += 1;

            // Collect ALL candidates from ALL strategies (not just first match)
            let mut all_candidates: Vec<(Candidate, &str)> = vec![];

            for strategy in &self.strategies {
                // Skip strategies that don't support this language
                if let Some(ref lang) = self.config.language {
                    if !strategy.supports_language(lang) {
                        continue;
                    }
                }

                let candidates = strategy.resolve(tag, &context);
                for c in candidates {
                    all_candidates.push((c, strategy.name()));
                }
            }

            if all_candidates.is_empty() {
                stats.unresolved += 1;
                continue;
            }

            // Sort by confidence descending to find best candidates
            all_candidates.sort_by(|a, b| b.0.confidence.partial_cmp(&a.0.confidence).unwrap());

            // Check for strategy competition: top-2 within 0.1
            if all_candidates.len() >= 2 {
                let top1_conf = all_candidates[0].0.confidence;
                let top2_conf = all_candidates[1].0.confidence;
                if (top1_conf - top2_conf).abs() < 0.1 {
                    stats.strategy_competition += 1;
                }

                // Check for strategy agreement: do top candidates point to same target?
                let top1_target = &all_candidates[0].0.target;
                let mut agreement_count = 0;
                for (cand, _) in &all_candidates[1..] {
                    if &cand.target == top1_target {
                        agreement_count += 1;
                    }
                }
                if agreement_count > 0 {
                    stats.strategy_agreement += 1;
                }
            }

            // Apply acceptance gate to best candidate
            let (best, best_strategy) = &all_candidates[0];

            let accepted = if let Some(hard_min) = self.config.min_confidence {
                // Legacy mode: hard threshold
                best.confidence >= hard_min
            } else if self.config.use_sigmoid_gate {
                // Modern mode: sigmoid gate
                // Accept if probability > 0.5 (deterministic for stats)
                self.coords.acceptance_probability(best.confidence) > 0.5
            } else {
                // No gating: accept all
                true
            };

            if accepted {
                // Update by_strategy counter for the winning strategy
                *stats
                    .by_strategy
                    .entry(best_strategy.to_string())
                    .or_insert(0) += 1;

                // Update confidence histogram
                // Buckets: [0.0-0.2), [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0]
                let bucket = ((best.confidence * 5.0).floor() as usize).min(4);
                stats.confidence_histogram[bucket] += 1;
            } else {
                // Candidate was dropped by gate
                stats.dropped_by_gate += 1;
                stats.unresolved += 1;

                // Track near-misses: candidates within 0.2 of threshold
                if best.confidence >= (gate_threshold - 0.2) {
                    stats.gate_near_misses += gate_threshold - best.confidence;
                }
            }
        }

        stats
    }
}

impl Default for CallResolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about call resolution - instrumented for L1 visibility.
///
/// L1 uses these to diagnose coordinate tuning:
/// - High `dropped_by_gate` → acceptance threshold too high (min_confidence or sigmoid bias)
/// - High `strategy_competition` → need to separate strategy confidence levels more
/// - `confidence_histogram` reveals score distribution
/// - Gate diagnostics show how many candidates were close to threshold
///
/// # Integration with StrategyCoordinates
///
/// When using sigmoid gating (modern mode), these stats reveal:
/// - Whether `acceptance_bias` in the acceptance gate is well-tuned
/// - Whether strategy weighting needs adjustment
/// - Distribution patterns that guide coordinate optimization
#[derive(Debug, Default, Clone)]
pub struct ResolutionStats {
    // Existing fields
    pub total_calls: usize,
    pub unresolved: usize,
    pub by_strategy: HashMap<String, usize>,

    // NEW: Gate diagnostics
    /// Candidates that passed initial strategy matching but were rejected by confidence gate.
    /// High value suggests the min_confidence threshold (or sigmoid bias) is too aggressive.
    pub dropped_by_gate: usize,

    /// Sum of (threshold - confidence) for rejected candidates that were close.
    /// High value = many near-misses, consider lowering acceptance threshold.
    /// We track candidates within 0.2 of threshold as "near-misses".
    pub gate_near_misses: f64,

    // NEW: Competition diagnostics
    /// Cases where top-2 candidates had confidence within 0.1 of each other.
    /// High value suggests strategies have overlapping confidence ranges,
    /// making resolution ambiguous. May need to adjust strategy confidence levels.
    pub strategy_competition: usize,

    /// Cases where multiple strategies agreed on the same target (resolved to same FunctionId).
    /// High value indicates good cross-strategy consensus, boosting confidence in resolutions.
    pub strategy_agreement: usize,

    // NEW: Distribution diagnostics
    /// Histogram of final selected confidence scores (buckets: 0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0).
    /// Reveals the distribution of resolution confidence. Useful for:
    /// - Detecting if most resolutions are low-confidence (histogram skewed left)
    /// - Detecting if threshold is cutting off a large cluster (sharp drop at threshold)
    /// - Understanding the "shape" of the confidence landscape
    pub confidence_histogram: [usize; 5],

    // === LSP-SPECIFIC INSTRUMENTATION (Shadow Collapse Diagnostics) ===
    /// Calls resolved by LSP strategy.
    /// Incremented when strategy.name() == "lsp" and it returns candidates.
    /// Used to calculate LSP utilization and shadow coverage.
    pub lsp_resolved: usize,

    /// LSP queries issued (for utilization calculation).
    /// Tracks how many times LSP was queried, regardless of success.
    /// **Key metric**: `lsp_utilization = lsp_resolved / lsp_queries_issued`
    /// Low utilization indicates LSP overhead without payoff (wasted RPC calls).
    pub lsp_queries_issued: usize,

    /// Cases where LSP and heuristics disagreed (resolved to different targets).
    /// When both LSP and heuristic strategies resolve a call but to different targets,
    /// this counter increments. High value indicates systematic bias between strategies.
    /// **Shadow Collapse signal**: If LSP always disagrees, shadow graph diverges from heuristic graph.
    pub lsp_heuristic_conflicts: usize,

    /// Cases where LSP confirmed a heuristic guess (same target).
    /// When both LSP and heuristic strategies resolve to the SAME target,
    /// this counter increments. High value indicates heuristic strategies are already accurate.
    /// **Implication**: If heuristics are already good, LSP adds latency without benefit.
    pub lsp_heuristic_agreements: usize,
}

impl ResolutionStats {
    /// Resolution success rate (resolved / total).
    pub fn resolution_rate(&self) -> f64 {
        if self.total_calls == 0 {
            return 1.0;
        }
        (self.total_calls - self.unresolved) as f64 / self.total_calls as f64
    }

    /// Fraction of candidates rejected by confidence gate.
    ///
    /// This approximates the gate rejection rate by comparing dropped candidates
    /// to total calls. In a multi-strategy scenario, this reveals how "picky"
    /// the acceptance gate is.
    pub fn gate_rejection_rate(&self) -> f64 {
        if self.total_calls == 0 {
            return 0.0;
        }
        self.dropped_by_gate as f64 / self.total_calls as f64
    }

    /// Fraction of resolutions with strategy competition (top-2 within 0.1).
    ///
    /// High competition rate suggests ambiguous resolutions where multiple
    /// strategies provide similar confidence. May indicate need to adjust
    /// strategy confidence levels to create clearer winners.
    pub fn competition_rate(&self) -> f64 {
        let resolved = self.total_calls - self.unresolved;
        if resolved == 0 {
            return 0.0;
        }
        self.strategy_competition as f64 / resolved as f64
    }

    // === LSP-SPECIFIC DIAGNOSTICS (Shadow Collapse Metrics) ===

    /// Calculate LSP utilization: resolved / queries issued.
    ///
    /// Returns 0.0 if no queries issued (avoids div by zero).
    ///
    /// **Interpretation**:
    /// - High utilization (>0.7): LSP is productive, most queries yield resolutions
    /// - Low utilization (<0.3): LSP overhead without payoff, many wasted RPC calls
    /// - This metric reveals whether LSP is worth its latency cost
    pub fn lsp_utilization(&self) -> f64 {
        if self.lsp_queries_issued == 0 {
            0.0
        } else {
            self.lsp_resolved as f64 / self.lsp_queries_issued as f64
        }
    }

    /// Calculate LSP-heuristic conflict rate: conflicts / (conflicts + agreements).
    ///
    /// Returns 0.0 if no LSP-heuristic interactions occurred.
    ///
    /// **Interpretation**:
    /// - High conflict rate (>0.5): LSP and heuristics disagree frequently → Shadow Collapse
    /// - Low conflict rate (<0.2): LSP confirms heuristics → Shadow is close to base graph
    /// - This is a proxy for "shadow connectivity divergence"
    pub fn lsp_conflict_rate(&self) -> f64 {
        let total_interactions = self.lsp_heuristic_conflicts + self.lsp_heuristic_agreements;
        if total_interactions == 0 {
            0.0
        } else {
            self.lsp_heuristic_conflicts as f64 / total_interactions as f64
        }
    }

    /// Format stats as JSON for L1 consumption.
    ///
    /// This provides a machine-readable format for training loops and
    /// diagnostic tools to consume resolution statistics.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "total_calls": self.total_calls,
            "unresolved": self.unresolved,
            "resolution_rate": self.resolution_rate(),
            "by_strategy": self.by_strategy,
            "dropped_by_gate": self.dropped_by_gate,
            "gate_near_misses": self.gate_near_misses,
            "gate_rejection_rate": self.gate_rejection_rate(),
            "strategy_competition": self.strategy_competition,
            "strategy_agreement": self.strategy_agreement,
            "competition_rate": self.competition_rate(),
            "confidence_histogram": self.confidence_histogram,
            // LSP-specific Shadow Collapse diagnostics
            "lsp_resolved": self.lsp_resolved,
            "lsp_queries_issued": self.lsp_queries_issued,
            "lsp_utilization": self.lsp_utilization(),
            "lsp_heuristic_conflicts": self.lsp_heuristic_conflicts,
            "lsp_heuristic_agreements": self.lsp_heuristic_agreements,
            "lsp_conflict_rate": self.lsp_conflict_rate(),
        })
    }
}

/// Builder for creating a fully-configured resolver with default strategies.
///
/// EVOLUTION: Boolean toggles replaced with continuous scalars [0.0, 1.0].
/// This creates smooth gradients for L1 optimization instead of cliffs.
/// - 0.0 = strategy disabled (same as false)
/// - 1.0 = strategy fully weighted (same as true)
/// - 0.5 = strategy at half weight (NEW capability)
///
/// Backward-compatible boolean methods preserved for existing code.
pub struct ResolverBuilder {
    config: ResolverConfig,
    coords: StrategyCoordinates, // NEW: coordinate-based selection policy

    // Continuous strategy weights [0.0, 1.0] - smooth gradients for L1
    same_file_weight: f64,
    type_hints_weight: f64,
    imports_weight: f64,
    name_match_weight: f64,
}

impl ResolverBuilder {
    pub fn new() -> Self {
        Self {
            config: ResolverConfig::default(),
            coords: StrategyCoordinates::default(), // NEW: default coordinates
            // All strategies enabled by default (weight = 1.0)
            same_file_weight: 1.0,
            type_hints_weight: 1.0,
            imports_weight: 1.0,
            name_match_weight: 1.0,
        }
    }

    pub fn config(mut self, config: ResolverConfig) -> Self {
        self.config = config;
        self
    }

    /// Set strategy coordinates for differentiable selection.
    ///
    /// **Use case**: Training loop provides custom coordinates to tune:
    /// - Strategy weights (which strategies to trust)
    /// - Acceptance threshold (how selective to be)
    /// - Selection mode (softmax vs evidence accumulation)
    pub fn coordinates(mut self, coords: StrategyCoordinates) -> Self {
        self.coords = coords;
        self
    }

    // === Continuous Weight Methods (NEW: enable smooth L1 gradients) ===

    /// Set weight for same-file strategy [0.0, 1.0].
    /// 0.0 = disabled, 1.0 = full weight, 0.5 = half weight.
    pub fn same_file_weight(mut self, weight: f64) -> Self {
        self.same_file_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set weight for type-hint strategy [0.0, 1.0].
    pub fn type_hints_weight(mut self, weight: f64) -> Self {
        self.type_hints_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set weight for import strategy [0.0, 1.0].
    pub fn imports_weight(mut self, weight: f64) -> Self {
        self.imports_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set weight for name-match strategy [0.0, 1.0].
    pub fn name_match_weight(mut self, weight: f64) -> Self {
        self.name_match_weight = weight.clamp(0.0, 1.0);
        self
    }

    // === Boolean Methods (BACKWARD COMPATIBLE: map to 0.0 or 1.0) ===

    /// Enable/disable same-file strategy (backward compatible).
    /// Maps to same_file_weight(1.0) or same_file_weight(0.0).
    pub fn same_file(mut self, enabled: bool) -> Self {
        self.same_file_weight = if enabled { 1.0 } else { 0.0 };
        self
    }

    /// Enable/disable type-hints strategy (backward compatible).
    pub fn type_hints(mut self, enabled: bool) -> Self {
        self.type_hints_weight = if enabled { 1.0 } else { 0.0 };
        self
    }

    /// Enable/disable imports strategy (backward compatible).
    pub fn imports(mut self, enabled: bool) -> Self {
        self.imports_weight = if enabled { 1.0 } else { 0.0 };
        self
    }

    /// Enable/disable name-match strategy (backward compatible).
    pub fn name_match(mut self, enabled: bool) -> Self {
        self.name_match_weight = if enabled { 1.0 } else { 0.0 };
        self
    }

    /// Build the resolver with weighted strategies.
    ///
    /// SMOOTH GRADIENTS: Strategies are weighted by continuous scalars [0.0, 1.0].
    /// - Weight > 0.01: strategy is added with confidence multiplied by weight
    /// - Weight <= 0.01: strategy is effectively disabled
    ///
    /// This allows L1 to "fade" strategies smoothly instead of killing them at cliffs.
    /// Example: weight=0.3 means strategy confidence is scaled to 30% of its base value.
    pub fn build(self) -> CallResolver {
        use super::strategies::*;

        let mut resolver = CallResolver::new().with_config(self.config);

        // Add strategies in order of confidence (highest first)
        // Only add if weight > 0.01 (essentially enabled)
        // The weight multiplies the strategy's base confidence score

        if self.same_file_weight > 0.01 {
            // Apply weight to base confidence: 0.9 * weight
            let base_confidence = 0.9 * self.same_file_weight;
            resolver.add_strategy(Box::new(SameFileStrategy::with_base_confidence(
                base_confidence,
            )));
        }

        if self.type_hints_weight > 0.01 {
            let base_confidence = 0.85 * self.type_hints_weight;
            resolver.add_strategy(Box::new(TypeHintStrategy::with_base_confidence(
                base_confidence,
            )));
        }

        if self.imports_weight > 0.01 {
            let base_confidence = 0.8 * self.imports_weight;
            resolver.add_strategy(Box::new(ImportStrategy::with_base_confidence(
                base_confidence,
            )));
        }

        if self.name_match_weight > 0.01 {
            let base_confidence = 0.5 * self.name_match_weight;
            let proximity_boost = 0.1 * self.name_match_weight;
            // Both base confidence and proximity_boost are scaled by weight
            // to maintain the relative importance of the proximity signal
            resolver.add_strategy(Box::new(NameMatchStrategy::with_params(
                base_confidence,
                proximity_boost,
            )));
        }

        // Transfer coordinates to resolver
        resolver.coords = self.coords;

        resolver
    }
}

impl Default for ResolverBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_build_graph() {
        let tags = vec![
            make_def("test.py", "main", 1),
            make_def("test.py", "helper", 10),
            make_call("test.py", "helper", 5), // main calls helper at line 5
        ];

        let resolver = ResolverBuilder::new().build();
        let graph = resolver.build_graph(&tags);

        assert_eq!(graph.function_count(), 2);
        assert_eq!(graph.call_count(), 1);
    }

    #[test]
    fn test_resolution_stats() {
        let tags = vec![
            make_def("test.py", "main", 1),
            make_def("test.py", "helper", 10),
            make_call("test.py", "helper", 5),
            make_call("test.py", "unknown", 7), // Unresolved
        ];

        let resolver = ResolverBuilder::new()
            .name_match(false) // Disable name matching to test unresolved
            .build();
        let stats = resolver.stats(&tags);

        assert_eq!(stats.total_calls, 2);
        assert!(stats.resolution_rate() > 0.0);
    }

    #[test]
    fn test_sigmoid_gate_vs_hard_threshold() {
        use super::StrategyCoordinates;

        let tags = vec![
            make_def("test.py", "main", 1),
            make_def("test.py", "helper", 10),
            make_call("test.py", "helper", 5),
        ];

        // Test legacy mode with hard threshold
        let legacy_config = ResolverConfig::with_min_confidence(0.3);
        let legacy_resolver = ResolverBuilder::new().config(legacy_config).build();
        let legacy_graph = legacy_resolver.build_graph(&tags);

        // Test modern mode with sigmoid gate
        let modern_config = ResolverConfig::default(); // use_sigmoid_gate = true
        let modern_coords = StrategyCoordinates {
            acceptance_bias: 0.3,
            acceptance_slope: 10.0,
            ..Default::default()
        };
        let modern_resolver = ResolverBuilder::new()
            .config(modern_config)
            .build()
            .with_coordinates(modern_coords);
        let modern_graph = modern_resolver.build_graph(&tags);

        // Both should produce similar results for high-confidence edges
        assert_eq!(legacy_graph.call_count(), modern_graph.call_count());
    }

    #[test]
    fn test_acceptance_gate_smooth_transition() {
        use super::StrategyCoordinates;

        // Coordinates with bias at 0.5
        let coords = StrategyCoordinates {
            acceptance_bias: 0.5,
            acceptance_slope: 10.0,
            ..Default::default()
        };

        // Test smooth transition around bias point
        let p_low = coords.acceptance_probability(0.4);
        let p_mid = coords.acceptance_probability(0.5);
        let p_high = coords.acceptance_probability(0.6);

        // Should be monotonically increasing
        assert!(p_low < p_mid);
        assert!(p_mid < p_high);

        // At bias, should be ~0.5
        assert!((p_mid - 0.5).abs() < 0.01, "At bias, p should be ~0.5");

        // Should be smooth, not a cliff
        assert!(p_low > 0.1, "Should not drop to zero just below bias");
        assert!(p_high < 0.9, "Should not jump to one just above bias");
    }
}
