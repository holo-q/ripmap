//! Core schemas for the L2 outer loop.
//!
//! These types define the interface between inner runs (L1) and the outer
//! promptgram optimizer (L2).

use serde::{Deserialize, Serialize};

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
        let temporal_horizon = 1.0 - (30.0 / params.git_recency_decay_days.max(1.0)).clamp(0.0, 1.0);

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

    /// For replace/delete: target text or rule to modify
    #[serde(default)]
    pub target: String,

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
        let mut ids: Vec<String> = self.episodes
            .iter()
            .map(|e| e.prompt_id.clone())
            .collect();
        ids.sort();
        ids.dedup();
        ids
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
        let a = RunMetrics { ndcg: 0.85, failures: 10, mean_confidence: 0.7 };
        let b = RunMetrics { ndcg: 0.80, failures: 15, mean_confidence: 0.6 };
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
                final_metrics: RunMetrics { ndcg: 0.85, failures: 5, mean_confidence: 0.7 },
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
        });

        // Search should find matches
        assert_eq!(pad.search_failures("depth").len(), 1);
        assert_eq!(pad.search_failures("boost").len(), 1);
        assert_eq!(pad.search_failures("nonexistent").len(), 0);
    }
}
