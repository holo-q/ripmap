//! Core schemas for the L2 outer loop.
//!
//! These types define the interface between inner runs (L1) and the outer
//! promptgram optimizer (L2).

use serde::{Deserialize, Deserializer, Serialize};

use crate::training::ParameterPoint;

/// Custom deserializer that handles both null and missing string fields.
/// Gemini sometimes outputs `"target": null` for append operations.
fn deserialize_nullable_string<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    // Deserialize as Option<String>, which naturally handles null → None
    Option::<String>::deserialize(deserializer)
}

/// Meta-levers: latent axes describing optimization behavior.
///
/// These are high-level concepts that summarize HOW the optimizer is behaving,
/// not just what parameters it's tuning. They let the outer loop reason in
/// a human-interpretable space.
///
/// Values are normalized to [0.0, 1.0]:
/// - 0.0 = fully one pole
/// - 1.0 = fully the opposite pole
/// - 0.5 = balanced
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetaLevers {
    /// Structural vs Contextual Trust
    /// Low = trust git/temporal signals, High = trust graph structure
    pub structural_trust: f64,

    /// Temporal Horizon
    /// Low = focus on recent changes, High = weight historical patterns equally
    pub temporal_horizon: f64,

    /// Exploration vs Exploitation Bias
    /// Low = conservative incremental, High = bold experimental
    pub exploration_bias: f64,

    /// Depth Sensitivity
    /// Low = flat (all depths equal), High = steep penalty for deep files
    pub depth_sensitivity: f64,

    /// Hub Damping
    /// Low = let popular files dominate, High = penalize high-degree nodes
    pub hub_damping: f64,

    /// Focus Locality
    /// Low = broad graph traversal, High = tight local neighborhood
    pub focus_locality: f64,
}

impl MetaLevers {
    /// Estimate meta-levers from current parameters.
    ///
    /// This is a heuristic projection from θ (17D params) to ℓ (6D meta-levers).
    /// Eventually this could be learned, but we start with hand-coded mapping.
    pub fn from_params(params: &crate::training::gridsearch::ParameterPoint) -> Self {
        // structural_trust: high pagerank_alpha = more structural
        let structural_trust = params.pagerank_alpha.clamp(0.0, 1.0);

        // temporal_horizon: inverse of recency decay (fast decay = short horizon)
        let temporal_horizon =
            1.0 - (30.0 / params.git_recency_decay_days.max(1.0)).clamp(0.0, 1.0);

        // exploration_bias: estimated from confidence if available, otherwise from param variance
        // For now, assume moderate exploration
        let exploration_bias = 0.5;

        // depth_sensitivity: ratio of root to deep weights
        let depth_ratio = params.depth_weight_root / params.depth_weight_deep.max(0.01);
        let depth_sensitivity = (depth_ratio / 100.0).clamp(0.0, 1.0);

        // hub_damping: inverse of chat multiplier (high multiplier = low damping)
        let hub_damping = 1.0 - (params.pagerank_chat_multiplier / 200.0).clamp(0.0, 1.0);

        // focus_locality: inverse of max_hops and decay
        let focus_locality = (1.0 - params.focus_decay) * (1.0 - params.focus_max_hops / 5.0);
        let focus_locality = focus_locality.clamp(0.0, 1.0);

        MetaLevers {
            structural_trust,
            temporal_horizon,
            exploration_bias,
            depth_sensitivity,
            hub_damping,
            focus_locality,
        }
    }

    /// Format as a compact string for display.
    pub fn summary(&self) -> String {
        format!(
            "struct={:.2} temp={:.2} explore={:.2} depth={:.2} hub={:.2} focus={:.2}",
            self.structural_trust,
            self.temporal_horizon,
            self.exploration_bias,
            self.depth_sensitivity,
            self.hub_damping,
            self.focus_locality,
        )
    }
}

/// Summary of an inner run for L2 consumption.
///
/// This is the canonical interface between L1 and L2. The outer loop doesn't
/// read raw traces - it reads these summaries which compress the essential
/// information about what happened and why.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OuterEpisodeSummary {
    /// Outer step number
    pub outer_step: usize,

    /// ID of the promptgram used for this inner run
    pub prompt_id: String,

    /// Metrics at start of inner run
    pub baseline_metrics: RunMetrics,

    /// Metrics at end of inner run
    pub final_metrics: RunMetrics,

    /// Delta from baseline to final
    pub delta: RunMetrics,

    /// Stability metrics (variance, collapse events)
    pub stability: StabilityMetrics,

    /// Estimated meta-lever position at end of run
    pub meta_levers_estimate: MetaLevers,

    /// Strategy capsules from inner episodes (the "why" of changes)
    /// These encode the intent/reasoning behind parameter moves.
    pub strategy_capsules: Vec<String>,

    /// Notable failure patterns observed
    pub notable_failures: Vec<String>,

    /// Structural insights discovered (beyond parameter tuning)
    pub structural_insights: Vec<String>,

    /// Number of inner episodes run
    pub inner_episodes: usize,

    /// Total duration of inner run (seconds)
    pub duration_secs: f64,

    /// Timestamp when this outer episode completed
    pub timestamp: i64,

    /// L2 proposal that was made (if any) - tracks what L2 decided
    #[serde(default)]
    pub proposal: Option<OuterProposal>,

    /// Selection mode that chose this promptgram (explore/exploit/best)
    #[serde(default)]
    pub selection_mode: String,

    /// Step from which we warm-started params (None = started from defaults)
    /// Critical diagnostic: if always None, we're not accumulating learning
    #[serde(default)]
    pub warm_started_from_step: Option<usize>,

    /// Path to final trained params from this run
    /// Used for tracking parameter trajectory across outer steps
    #[serde(default)]
    pub final_params_path: Option<String>,

    /// Actual final parameter values from this run.
    /// Stored inline for rich diagnostics - allows analyzing what parameters
    /// the inner loop settled on without loading external files.
    #[serde(default)]
    pub final_params: Option<ParameterPoint>,
}

/// Core metrics for a run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunMetrics {
    /// NDCG@10 score
    pub ndcg: f64,

    /// Number of ranking failures
    pub failures: usize,

    /// Mean confidence of inner optimizer
    pub mean_confidence: f64,
}

impl std::ops::Sub for RunMetrics {
    type Output = RunMetrics;

    fn sub(self, rhs: Self) -> Self::Output {
        RunMetrics {
            ndcg: self.ndcg - rhs.ndcg,
            failures: self.failures.saturating_sub(rhs.failures),
            mean_confidence: self.mean_confidence - rhs.mean_confidence,
        }
    }
}

/// Stability metrics for a run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StabilityMetrics {
    /// Number of collapse events (NDCG drops > 5% in one episode)
    pub collapse_events: usize,

    /// NDCG variance across episodes
    pub ndcg_variance: f64,

    /// Did the run converge (stabilize at high NDCG)?
    pub converged: bool,

    /// Number of oscillations (direction changes in NDCG trend)
    pub oscillations: usize,
}

/// A proposed edit to a promptgram section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptEdit {
    /// Which section to edit (Policy, Heuristics, Style, etc.)
    pub section: String,

    /// Type of edit: "append", "replace", "delete"
    pub edit_type: String,

    /// For replace/delete: target text or rule to modify.
    /// None or empty for append, or to replace entire section.
    #[serde(default, deserialize_with = "deserialize_nullable_string")]
    pub target: Option<String>,

    /// New content (for append/replace)
    pub content: String,

    /// Rationale for this edit
    pub rationale: String,
}

/// Output from the outer optimizer (L2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OuterProposal {
    /// Mode of operation: "explore", "exploit", "consolidate"
    pub mode: String,

    /// Justification for the chosen mode
    pub mode_justification: String,

    /// Confidence in this proposal (0.0 - 1.0)
    pub confidence: f64,

    /// Proposed edits to the promptgram
    pub edits: Vec<PromptEdit>,

    /// Expected effects of these edits
    pub expected_effects: Vec<String>,

    /// What we hope to learn from this experiment
    pub hypothesis: String,

    /// Risk assessment
    pub risk_level: String,
}

/// Accumulated state for the outer loop.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OuterScratchpad {
    /// All outer episode summaries
    pub episodes: Vec<OuterEpisodeSummary>,

    /// Current best promptgram ID
    pub best_prompt_id: String,

    /// Best NDCG achieved
    pub best_ndcg: f64,

    /// Hall of fame: top N promptgrams by final NDCG
    pub hall_of_fame: Vec<(String, f64)>,

    /// Discovered prompt patterns that work
    pub success_patterns: Vec<String>,

    /// Prompt patterns that failed
    pub failure_patterns: Vec<String>,

    /// Meta-insights about promptgram optimization
    pub meta_insights: Vec<String>,

    /// Path to best parameters found so far (for continuation/warm start).
    /// This enables L2 to build on learned params rather than starting from scratch.
    /// Diagnosis: Without this, each inner run starts from default params, and
    /// the outer loop shows no mean improvement (σ=0.0018 baseline vs 0.0032 final,
    /// mean unchanged at 0.8269).
    #[serde(default)]
    pub best_params_path: Option<String>,

    /// Step at which best params were found
    #[serde(default)]
    pub best_params_step: usize,
}

impl OuterScratchpad {
    /// Get the last N episode summaries.
    pub fn recent_episodes(&self, n: usize) -> Vec<&OuterEpisodeSummary> {
        self.episodes.iter().rev().take(n).collect()
    }

    /// Check if we're in a plateau (no improvement in last N episodes).
    pub fn is_plateau(&self, n: usize, threshold: f64) -> bool {
        let recent: Vec<_> = self.episodes.iter().rev().take(n).collect();
        if recent.len() < n {
            return false;
        }

        let first_ndcg = recent.last().map(|e| e.final_metrics.ndcg).unwrap_or(0.0);
        let last_ndcg = recent.first().map(|e| e.final_metrics.ndcg).unwrap_or(0.0);

        (last_ndcg - first_ndcg).abs() < threshold
    }

    /// Check if we're in collapse (NDCG degrading).
    pub fn is_collapse(&self, n: usize, threshold: f64) -> bool {
        let recent: Vec<_> = self.episodes.iter().rev().take(n).collect();
        if recent.len() < n {
            return false;
        }

        let first_ndcg = recent.last().map(|e| e.final_metrics.ndcg).unwrap_or(0.0);
        let last_ndcg = recent.first().map(|e| e.final_metrics.ndcg).unwrap_or(0.0);

        last_ndcg < first_ndcg - threshold
    }

    /// Search for failure patterns across all episodes.
    ///
    /// Returns episodes where the pattern appears in notable_failures or strategy_capsules.
    /// L2 can use this to scavenge its own history for recurring issues.
    pub fn search_failures(&self, pattern: &str) -> Vec<(usize, &OuterEpisodeSummary)> {
        let pattern_lower = pattern.to_lowercase();
        self.episodes
            .iter()
            .enumerate()
            .filter(|(_, ep)| {
                // Check notable failures
                ep.notable_failures.iter().any(|f| f.to_lowercase().contains(&pattern_lower))
                    // Check strategy capsules (sometimes failures appear here)
                    || ep.strategy_capsules.iter().any(|c| c.to_lowercase().contains(&pattern_lower))
                    // Check structural insights
                    || ep.structural_insights.iter().any(|i| i.to_lowercase().contains(&pattern_lower))
            })
            .collect()
    }

    /// Get episodes that used a specific promptgram.
    pub fn episodes_with_prompt(&self, prompt_id: &str) -> Vec<&OuterEpisodeSummary> {
        self.episodes
            .iter()
            .filter(|ep| ep.prompt_id == prompt_id)
            .collect()
    }

    /// Compute mean NDCG for a promptgram across all its runs.
    pub fn promptgram_stats(&self, prompt_id: &str) -> Option<PromptgramStats> {
        let episodes: Vec<_> = self.episodes_with_prompt(prompt_id);
        if episodes.is_empty() {
            return None;
        }

        let ndcgs: Vec<f64> = episodes.iter().map(|e| e.final_metrics.ndcg).collect();
        let mean_ndcg = ndcgs.iter().sum::<f64>() / ndcgs.len() as f64;
        let best_ndcg = ndcgs.iter().cloned().fold(0.0, f64::max);
        let worst_ndcg = ndcgs.iter().cloned().fold(1.0, f64::min);

        Some(PromptgramStats {
            prompt_id: prompt_id.to_string(),
            run_count: episodes.len(),
            mean_ndcg,
            best_ndcg,
            worst_ndcg,
            first_step: episodes.first().map(|e| e.outer_step).unwrap_or(0),
            last_step: episodes.last().map(|e| e.outer_step).unwrap_or(0),
        })
    }

    /// Get all unique promptgram IDs used so far.
    pub fn unique_promptgrams(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.episodes.iter().map(|e| e.prompt_id.clone()).collect();
        ids.sort();
        ids.dedup();
        ids
    }

    /// Compute parameter deltas between consecutive steps.
    ///
    /// Returns Vec of (step, param_name, old_value, new_value, delta).
    /// Critical diagnostic: shows what actually changed between runs.
    pub fn parameter_deltas(&self) -> Vec<(usize, String, f64, f64, f64)> {
        let mut deltas = Vec::new();

        for window in self.episodes.windows(2) {
            let prev = &window[0];
            let curr = &window[1];

            if let (Some(p1), Some(p2)) = (&prev.final_params, &curr.final_params) {
                let fields = [
                    ("pagerank_alpha", p1.pagerank_alpha, p2.pagerank_alpha),
                    (
                        "pagerank_chat_multiplier",
                        p1.pagerank_chat_multiplier,
                        p2.pagerank_chat_multiplier,
                    ),
                    (
                        "depth_weight_root",
                        p1.depth_weight_root,
                        p2.depth_weight_root,
                    ),
                    (
                        "depth_weight_moderate",
                        p1.depth_weight_moderate,
                        p2.depth_weight_moderate,
                    ),
                    (
                        "depth_weight_deep",
                        p1.depth_weight_deep,
                        p2.depth_weight_deep,
                    ),
                    (
                        "depth_weight_vendor",
                        p1.depth_weight_vendor,
                        p2.depth_weight_vendor,
                    ),
                    (
                        "boost_mentioned_ident",
                        p1.boost_mentioned_ident,
                        p2.boost_mentioned_ident,
                    ),
                    (
                        "boost_mentioned_file",
                        p1.boost_mentioned_file,
                        p2.boost_mentioned_file,
                    ),
                    ("boost_chat_file", p1.boost_chat_file, p2.boost_chat_file),
                    (
                        "boost_temporal_coupling",
                        p1.boost_temporal_coupling,
                        p2.boost_temporal_coupling,
                    ),
                    (
                        "boost_focus_expansion",
                        p1.boost_focus_expansion,
                        p2.boost_focus_expansion,
                    ),
                    (
                        "git_recency_decay_days",
                        p1.git_recency_decay_days,
                        p2.git_recency_decay_days,
                    ),
                    (
                        "git_recency_max_boost",
                        p1.git_recency_max_boost,
                        p2.git_recency_max_boost,
                    ),
                    (
                        "git_churn_threshold",
                        p1.git_churn_threshold,
                        p2.git_churn_threshold,
                    ),
                    (
                        "git_churn_max_boost",
                        p1.git_churn_max_boost,
                        p2.git_churn_max_boost,
                    ),
                    ("focus_decay", p1.focus_decay, p2.focus_decay),
                    ("focus_max_hops", p1.focus_max_hops, p2.focus_max_hops),
                ];

                for (name, v1, v2) in fields {
                    let delta = v2 - v1;
                    if delta.abs() > 1e-6 {
                        deltas.push((curr.outer_step, name.to_string(), v1, v2, delta));
                    }
                }
            }
        }

        deltas
    }

    /// Generate a statistical summary of the entire L2 run.
    ///
    /// Prints human-readable diagnostics for understanding learning trajectory:
    /// - NDCG trajectory (start/end/best/variance)
    /// - Parameter evolution (which params changed most)
    /// - Warm-start diagnostics (how many steps leveraged previous learning)
    /// - Convergence analysis
    pub fn statistical_summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str("\n╔══════════════════════════════════════════════════════════════╗\n");
        summary.push_str("║                    L2 RUN STATISTICAL SUMMARY                ║\n");
        summary.push_str("╚══════════════════════════════════════════════════════════════╝\n\n");

        if self.episodes.is_empty() {
            summary.push_str("No episodes recorded.\n");
            return summary;
        }

        // NDCG trajectory
        let ndcgs: Vec<f64> = self.episodes.iter().map(|e| e.final_metrics.ndcg).collect();
        let first_ndcg = ndcgs.first().copied().unwrap_or(0.0);
        let last_ndcg = ndcgs.last().copied().unwrap_or(0.0);
        let best_ndcg = ndcgs.iter().cloned().fold(0.0_f64, f64::max);
        let worst_ndcg = ndcgs.iter().cloned().fold(1.0_f64, f64::min);
        let mean_ndcg = ndcgs.iter().sum::<f64>() / ndcgs.len() as f64;
        let variance =
            ndcgs.iter().map(|v| (v - mean_ndcg).powi(2)).sum::<f64>() / ndcgs.len() as f64;
        let std_dev = variance.sqrt();

        summary.push_str("═══ NDCG TRAJECTORY ═══\n");
        summary.push_str(&format!("  Episodes:     {}\n", self.episodes.len()));
        summary.push_str(&format!("  First:        {:.4}\n", first_ndcg));
        summary.push_str(&format!("  Last:         {:.4}\n", last_ndcg));
        summary.push_str(&format!(
            "  Best:         {:.4} (step {})\n",
            best_ndcg, self.best_params_step
        ));
        summary.push_str(&format!("  Worst:        {:.4}\n", worst_ndcg));
        summary.push_str(&format!("  Mean:         {:.4}\n", mean_ndcg));
        summary.push_str(&format!("  Std Dev:      {:.4}\n", std_dev));
        summary.push_str(&format!(
            "  Total Δ:      {:+.4} ({:+.1}%)\n",
            last_ndcg - first_ndcg,
            (last_ndcg - first_ndcg) / first_ndcg * 100.0
        ));
        summary.push_str("\n");

        // Learning diagnosis
        summary.push_str("═══ LEARNING DIAGNOSIS ═══\n");
        let warm_started: usize = self
            .episodes
            .iter()
            .filter(|e| e.warm_started_from_step.is_some())
            .count();
        summary.push_str(&format!(
            "  Warm-started:  {} / {} ({:.0}%)\n",
            warm_started,
            self.episodes.len(),
            warm_started as f64 / self.episodes.len() as f64 * 100.0
        ));

        if warm_started == 0 && self.episodes.len() > 1 {
            summary.push_str("  ⚠️  NO WARM STARTS - each step started from defaults!\n");
            summary.push_str("      Learning may not accumulate across steps.\n");
        }

        // Check for actual improvement
        if (last_ndcg - first_ndcg).abs() < std_dev * 0.5 {
            summary.push_str("  ⚠️  PLATEAU - final NDCG within 0.5σ of initial\n");
        } else if last_ndcg > first_ndcg {
            summary.push_str(&format!(
                "  ✓  IMPROVED by {:.4} ({:.1}σ)\n",
                last_ndcg - first_ndcg,
                (last_ndcg - first_ndcg) / std_dev
            ));
        } else {
            summary.push_str(&format!(
                "  ✗  REGRESSED by {:.4} ({:.1}σ)\n",
                first_ndcg - last_ndcg,
                (first_ndcg - last_ndcg) / std_dev
            ));
        }
        summary.push_str("\n");

        // Parameter evolution
        let deltas = self.parameter_deltas();
        if !deltas.is_empty() {
            summary.push_str("═══ PARAMETER EVOLUTION ═══\n");

            // Aggregate deltas per parameter
            use std::collections::HashMap;
            let mut param_totals: HashMap<&str, (f64, usize)> = HashMap::new();
            for (_, name, _, _, delta) in &deltas {
                let entry = param_totals.entry(name.as_str()).or_insert((0.0, 0));
                entry.0 += delta.abs();
                entry.1 += 1;
            }

            // Sort by total movement
            let mut sorted: Vec<_> = param_totals.into_iter().collect();
            sorted.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());

            summary.push_str("  Most changed params (cumulative absolute Δ):\n");
            for (name, (total, count)) in sorted.iter().take(5) {
                summary.push_str(&format!(
                    "    {}: Δ{:.2} ({} changes)\n",
                    name, total, count
                ));
            }
            summary.push_str("\n");
        }

        // Stability
        let collapse_total: usize = self
            .episodes
            .iter()
            .map(|e| e.stability.collapse_events)
            .sum();
        let oscillation_total: usize = self.episodes.iter().map(|e| e.stability.oscillations).sum();
        let converged_count = self
            .episodes
            .iter()
            .filter(|e| e.stability.converged)
            .count();

        summary.push_str("═══ STABILITY ═══\n");
        summary.push_str(&format!("  Collapses:    {} total\n", collapse_total));
        summary.push_str(&format!("  Oscillations: {} total\n", oscillation_total));
        summary.push_str(&format!(
            "  Converged:    {} / {} runs\n",
            converged_count,
            self.episodes.len()
        ));
        summary.push_str("\n");

        // Best config location
        summary.push_str("═══ BEST CONFIGURATION ═══\n");
        summary.push_str(&format!("  NDCG:   {:.4}\n", self.best_ndcg));
        summary.push_str(&format!("  Step:   {}\n", self.best_params_step));
        if let Some(ref path) = self.best_params_path {
            summary.push_str(&format!("  Path:   {}\n", path));
        }
        summary.push_str("\n");

        summary
    }

    /// Find episodes where a specific meta-lever was extreme (>0.8 or <0.2).
    ///
    /// Useful for L2 to understand when certain strategies were tried.
    pub fn extreme_lever_episodes(&self, lever: &str) -> Vec<(usize, f64, &OuterEpisodeSummary)> {
        self.episodes
            .iter()
            .enumerate()
            .filter_map(|(i, ep)| {
                let ml = &ep.meta_levers_estimate;
                let value = match lever {
                    "structural_trust" => ml.structural_trust,
                    "temporal_horizon" => ml.temporal_horizon,
                    "exploration_bias" => ml.exploration_bias,
                    "depth_sensitivity" => ml.depth_sensitivity,
                    "hub_damping" => ml.hub_damping,
                    "focus_locality" => ml.focus_locality,
                    _ => return None,
                };
                if value > 0.8 || value < 0.2 {
                    Some((i, value, ep))
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Statistics for a promptgram's performance across runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptgramStats {
    pub prompt_id: String,
    pub run_count: usize,
    pub mean_ndcg: f64,
    pub best_ndcg: f64,
    pub worst_ndcg: f64,
    pub first_step: usize,
    pub last_step: usize,
}

/// Configuration for an outer loop run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OuterConfig {
    /// Number of inner episodes per outer step
    pub inner_episodes: usize,

    /// Which agent to use for L1
    pub inner_agent: String,

    /// Which agent to use for L2
    pub outer_agent: String,

    /// Maximum outer steps
    pub max_outer_steps: usize,

    /// Exploration quota: minimum fraction of steps in explore mode
    pub exploration_quota: f64,

    /// Corpus to use for inner runs
    pub corpus: String,

    /// Path to inner prompt template (contains placeholders like {current_ndcg:.4})
    /// If not set, the promptgram will be rendered and saved to the output directory.
    pub prompt_template_path: Option<String>,

    /// Enable L2 prompt editing. Without this, outer loop just records (Stage 0).
    pub edit_prompts: bool,

    /// Sections of the promptgram that can be edited
    pub editable_sections: Vec<String>,

    /// Sections that are immutable (safety/contract)
    pub immutable_sections: Vec<String>,
}

impl Default for OuterConfig {
    fn default() -> Self {
        OuterConfig {
            inner_episodes: 20,
            inner_agent: "claude".to_string(),
            outer_agent: "codex".to_string(),
            max_outer_steps: 50,
            exploration_quota: 0.2,
            corpus: "curated".to_string(),
            prompt_template_path: Some("training-outer/prompts/inner/v001.md".to_string()),
            edit_prompts: false,
            editable_sections: vec![
                "Policy".to_string(),
                "Heuristics".to_string(),
                "Style".to_string(),
            ],
            immutable_sections: vec![
                "Role".to_string(),
                "API_contract".to_string(),
                "Output_schema".to_string(),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_levers_summary() {
        let levers = MetaLevers::default();
        let s = levers.summary();
        assert!(s.contains("struct="));
        assert!(s.contains("explore="));
    }

    #[test]
    fn test_run_metrics_sub() {
        let a = RunMetrics {
            ndcg: 0.85,
            failures: 10,
            mean_confidence: 0.7,
        };
        let b = RunMetrics {
            ndcg: 0.80,
            failures: 15,
            mean_confidence: 0.6,
        };
        let delta = a - b;
        assert!((delta.ndcg - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_outer_scratchpad_plateau() {
        let mut pad = OuterScratchpad::default();

        // Add 5 episodes with same NDCG
        for i in 0..5 {
            pad.episodes.push(OuterEpisodeSummary {
                outer_step: i,
                prompt_id: "test".to_string(),
                baseline_metrics: RunMetrics::default(),
                final_metrics: RunMetrics {
                    ndcg: 0.85,
                    failures: 5,
                    mean_confidence: 0.7,
                },
                delta: RunMetrics::default(),
                stability: StabilityMetrics::default(),
                meta_levers_estimate: MetaLevers::default(),
                strategy_capsules: vec![],
                notable_failures: vec![],
                structural_insights: vec![],
                inner_episodes: 10,
                duration_secs: 60.0,
                timestamp: 0,
                proposal: None,
                selection_mode: "best".to_string(),
                warm_started_from_step: None,
                final_params_path: None,
                final_params: None,
            });
        }

        assert!(pad.is_plateau(3, 0.01));
        assert!(!pad.is_collapse(3, 0.01));
    }

    #[test]
    fn test_search_failures() {
        let mut pad = OuterScratchpad::default();

        pad.episodes.push(OuterEpisodeSummary {
            outer_step: 1,
            prompt_id: "v001".to_string(),
            baseline_metrics: RunMetrics::default(),
            final_metrics: RunMetrics::default(),
            delta: RunMetrics::default(),
            stability: StabilityMetrics::default(),
            meta_levers_estimate: MetaLevers::default(),
            strategy_capsules: vec!["boost was too high".to_string()],
            notable_failures: vec!["depth penalty issue".to_string()],
            structural_insights: vec![],
            inner_episodes: 10,
            duration_secs: 60.0,
            timestamp: 0,
            proposal: None,
            selection_mode: "explore".to_string(),
            warm_started_from_step: None,
            final_params_path: None,
            final_params: None,
        });

        // Search should find matches
        assert_eq!(pad.search_failures("depth").len(), 1);
        assert_eq!(pad.search_failures("boost").len(), 1);
        assert_eq!(pad.search_failures("nonexistent").len(), 0);
    }
}
