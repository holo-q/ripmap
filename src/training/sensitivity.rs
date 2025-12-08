//! Sensitivity analysis for understanding parameter importance.
//!
//! ## Ablation Studies
//!
//! Measure the impact of each parameter by:
//! 1. Setting it to its "neutral" value (1.0 for multipliers, midpoint for others)
//! 2. Measuring degradation in NDCG
//! 3. Ranking parameters by impact
//!
//! Parameters with high ablation impact are "load-bearing" - the ranking
//! quality depends heavily on them being tuned correctly.
//!
//! ## One-at-a-Time Sensitivity
//!
//! Sweep each parameter while holding others fixed at baseline:
//! - Generates a curve of metric vs parameter value
//! - Reveals non-linear effects, optimal ranges, and plateaus
//!
//! ## Interaction Effects
//!
//! Some parameters interact - their combined effect is non-additive.
//! We detect this by comparing:
//! - Effect of A alone + Effect of B alone
//! - Effect of A and B together
//!
//! Strong interactions suggest these parameters should be tuned jointly.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::gridsearch::{ParameterGrid, ParameterPoint};

/// Results of sensitivity analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityAnalysis {
    /// Parameter importance scores (higher = more impactful)
    /// Computed via ablation: how much does NDCG drop when we neutralize this param?
    pub importance: HashMap<String, f64>,

    /// Rank of each parameter by importance (1 = most important)
    pub importance_rank: HashMap<String, usize>,

    /// One-at-a-time sensitivity curves: param -> [(value, ndcg)]
    pub oat_curves: HashMap<String, Vec<(f64, f64)>>,

    /// Detected interactions: "param_a|param_b" -> interaction strength
    /// Positive = synergistic, Negative = antagonistic
    /// Key format is "param_a|param_b" for JSON compatibility
    pub interactions: HashMap<String, f64>,
}

/// Perform ablation study to measure parameter importance.
///
/// For each parameter, we set it to a "neutral" value and measure
/// how much the metric degrades. Larger degradation = more important.
///
/// # Arguments
///
/// * `baseline` - The baseline parameter configuration (e.g., best found)
/// * `baseline_score` - NDCG of baseline configuration
/// * `evaluator` - Function that evaluates a parameter point and returns NDCG
///
/// # Returns
///
/// Map from parameter name to importance score (baseline_ndcg - ablated_ndcg)
pub fn ablation_study<F>(
    baseline: &ParameterPoint,
    baseline_score: f64,
    evaluator: F,
) -> HashMap<String, f64>
where
    F: Fn(&ParameterPoint) -> f64,
{
    let grid = ParameterGrid::default();
    let mut importance = HashMap::new();

    for name in grid.param_names() {
        let ablated = ablate_param(baseline, &name);
        let ablated_score = evaluator(&ablated);
        let impact = baseline_score - ablated_score;
        importance.insert(name, impact.max(0.0)); // Clamp to non-negative
    }

    importance
}

/// Create a copy of the parameter point with one parameter set to neutral.
///
/// Neutral values:
/// - Multipliers/boosts: 1.0 (no effect)
/// - Decay factors: 0.5 (middle)
/// - Thresholds: midpoint of range
fn ablate_param(baseline: &ParameterPoint, param: &str) -> ParameterPoint {
    let mut ablated = baseline.clone();

    match param {
        // Boosts -> 1.0 means no boost
        "boost_mentioned_ident" => ablated.boost_mentioned_ident = 1.0,
        "boost_mentioned_file" => ablated.boost_mentioned_file = 1.0,
        "boost_chat_file" => ablated.boost_chat_file = 1.0,
        "boost_temporal_coupling" => ablated.boost_temporal_coupling = 1.0,
        "boost_focus_expansion" => ablated.boost_focus_expansion = 1.0,

        // PageRank multiplier
        "pagerank_chat_multiplier" => ablated.pagerank_chat_multiplier = 1.0,

        // PageRank alpha -> middle of range
        "pagerank_alpha" => ablated.pagerank_alpha = 0.85,

        // Depth weights -> all equal (1.0)
        "depth_weight_root" => ablated.depth_weight_root = 1.0,
        "depth_weight_moderate" => ablated.depth_weight_moderate = 1.0,
        "depth_weight_deep" => ablated.depth_weight_deep = 1.0,
        "depth_weight_vendor" => ablated.depth_weight_vendor = 1.0,

        // Git -> neutral values
        "git_recency_decay_days" => ablated.git_recency_decay_days = 30.0,
        "git_recency_max_boost" => ablated.git_recency_max_boost = 1.0,
        "git_churn_threshold" => ablated.git_churn_threshold = 10.0,
        "git_churn_max_boost" => ablated.git_churn_max_boost = 1.0,

        // Focus expansion -> conservative
        "focus_decay" => ablated.focus_decay = 0.5,
        "focus_max_hops" => ablated.focus_max_hops = 1.0,

        _ => {} // Unknown param, leave unchanged
    }

    ablated
}

/// One-at-a-time sensitivity analysis.
///
/// For each parameter, sweep it across its range while holding others
/// at baseline values. Returns curves showing metric vs parameter value.
///
/// # Arguments
///
/// * `baseline` - Baseline parameter configuration
/// * `evaluator` - Function that evaluates a parameter point
/// * `n_points` - Number of points per parameter (default 11)
pub fn oat_sensitivity<F>(
    baseline: &ParameterPoint,
    evaluator: F,
    n_points: usize,
) -> HashMap<String, Vec<(f64, f64)>>
where
    F: Fn(&ParameterPoint) -> f64,
{
    let grid = ParameterGrid::default();
    let mut curves = HashMap::new();

    for name in grid.param_names() {
        let range = &grid.ranges[&name];
        let mut curve = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let t = i as f64 / (n_points - 1) as f64;
            let value = range.decode(t);

            let point = set_param(baseline, &name, value);
            let score = evaluator(&point);

            curve.push((value, score));
        }

        curves.insert(name, curve);
    }

    curves
}

/// Set a single parameter value in a ParameterPoint.
fn set_param(baseline: &ParameterPoint, param: &str, value: f64) -> ParameterPoint {
    let mut point = baseline.clone();

    match param {
        "pagerank_alpha" => point.pagerank_alpha = value,
        "pagerank_chat_multiplier" => point.pagerank_chat_multiplier = value,
        "depth_weight_root" => point.depth_weight_root = value,
        "depth_weight_moderate" => point.depth_weight_moderate = value,
        "depth_weight_deep" => point.depth_weight_deep = value,
        "depth_weight_vendor" => point.depth_weight_vendor = value,
        "boost_mentioned_ident" => point.boost_mentioned_ident = value,
        "boost_mentioned_file" => point.boost_mentioned_file = value,
        "boost_chat_file" => point.boost_chat_file = value,
        "boost_temporal_coupling" => point.boost_temporal_coupling = value,
        "boost_focus_expansion" => point.boost_focus_expansion = value,
        "git_recency_decay_days" => point.git_recency_decay_days = value,
        "git_recency_max_boost" => point.git_recency_max_boost = value,
        "git_churn_threshold" => point.git_churn_threshold = value,
        "git_churn_max_boost" => point.git_churn_max_boost = value,
        "focus_decay" => point.focus_decay = value,
        "focus_max_hops" => point.focus_max_hops = value,
        _ => {}
    }

    point
}

/// Detect parameter interactions (non-additive effects).
///
/// For each pair of parameters, compare:
/// - Effect of A alone + Effect of B alone
/// - Effect of A and B together
///
/// Large difference indicates interaction.
/// Returns HashMap with string keys "param_a|param_b" for JSON compatibility.
pub fn detect_interactions<F>(
    baseline: &ParameterPoint,
    baseline_score: f64,
    evaluator: F,
    ablation: &HashMap<String, f64>,
) -> HashMap<String, f64>
where
    F: Fn(&ParameterPoint) -> f64,
{
    let grid = ParameterGrid::default();
    let params = grid.param_names();
    let mut interactions = HashMap::new();

    // Only check top parameters (interaction detection is expensive)
    let mut sorted_params: Vec<_> = params
        .iter()
        .map(|p| (p.clone(), ablation.get(p).copied().unwrap_or(0.0)))
        .collect();
    sorted_params.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_params: Vec<_> = sorted_params
        .into_iter()
        .take(6) // Top 6 most important
        .map(|(p, _)| p)
        .collect();

    for (i, param_a) in top_params.iter().enumerate() {
        for param_b in top_params.iter().skip(i + 1) {
            // Effect of A alone
            let effect_a = ablation.get(param_a).copied().unwrap_or(0.0);

            // Effect of B alone
            let effect_b = ablation.get(param_b).copied().unwrap_or(0.0);

            // Effect of both together
            let ablated_both = ablate_param(&ablate_param(baseline, param_a), param_b);
            let score_both = evaluator(&ablated_both);
            let effect_both = baseline_score - score_both;

            // Interaction = difference from additivity
            // Positive = synergistic (together is worse than sum)
            // Negative = antagonistic (together is better than sum)
            let expected_additive = effect_a + effect_b;
            let interaction = effect_both - expected_additive;

            if interaction.abs() > 0.01 {
                // Only record significant interactions
                // Key format: "param_a|param_b" for JSON compatibility
                let key = format!("{}|{}", param_a, param_b);
                interactions.insert(key, interaction);
            }
        }
    }

    interactions
}

/// Compute full sensitivity analysis.
pub fn full_analysis<F>(baseline: &ParameterPoint, evaluator: F) -> SensitivityAnalysis
where
    F: Fn(&ParameterPoint) -> f64 + Copy,
{
    let baseline_score = evaluator(baseline);

    // Ablation importance
    let importance = ablation_study(baseline, baseline_score, evaluator);

    // Rank by importance
    let mut sorted: Vec<_> = importance.iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

    let importance_rank: HashMap<_, _> = sorted
        .iter()
        .enumerate()
        .map(|(rank, (name, _))| ((*name).clone(), rank + 1))
        .collect();

    // OAT curves (expensive - could skip for quick analysis)
    let oat_curves = oat_sensitivity(baseline, evaluator, 7);

    // Interactions (most expensive)
    let interactions = detect_interactions(baseline, baseline_score, evaluator, &importance);

    SensitivityAnalysis {
        importance,
        importance_rank,
        oat_curves,
        interactions,
    }
}

/// Print a summary of sensitivity analysis results.
pub fn print_summary(analysis: &SensitivityAnalysis) {
    println!("\n=== Parameter Importance (Ablation) ===\n");

    // Sort by importance
    let mut sorted: Vec<_> = analysis.importance.iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (name, importance) in sorted.iter().take(10) {
        let rank = analysis.importance_rank.get(*name).unwrap_or(&0);
        let bar_len = (*importance * 50.0).round() as usize;
        let bar = "█".repeat(bar_len.min(50));
        println!("{:>30}: {:>6.4} [{}] {}", name, importance, rank, bar);
    }

    if !analysis.interactions.is_empty() {
        println!("\n=== Significant Interactions ===\n");

        let mut sorted_interactions: Vec<_> = analysis.interactions.iter().collect();
        sorted_interactions.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (key, strength) in sorted_interactions.iter().take(5) {
            // Parse "param_a|param_b" format
            let parts: Vec<&str> = key.split('|').collect();
            let (a, b) = if parts.len() == 2 {
                (parts[0], parts[1])
            } else {
                (key.as_str(), "?")
            };
            let direction = if **strength > 0.0 {
                "synergistic"
            } else {
                "antagonistic"
            };
            println!("  {} × {} = {:+.4} ({})", a, b, strength, direction);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ablate_boost() {
        let baseline = ParameterPoint::default();
        let ablated = ablate_param(&baseline, "boost_mentioned_ident");
        assert!((ablated.boost_mentioned_ident - 1.0).abs() < 1e-6);
        // Other params should be unchanged
        assert!((ablated.pagerank_alpha - baseline.pagerank_alpha).abs() < 1e-6);
    }

    #[test]
    fn test_set_param() {
        let baseline = ParameterPoint::default();
        let modified = set_param(&baseline, "pagerank_alpha", 0.75);
        assert!((modified.pagerank_alpha - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_ablation_study() {
        let baseline = ParameterPoint::default();

        // Mock evaluator: score = sum of boost values (simplistic)
        let evaluator =
            |p: &ParameterPoint| p.boost_mentioned_ident * 0.1 + p.boost_chat_file * 0.05;

        let importance = ablation_study(&baseline, evaluator(&baseline), evaluator);

        // boost_mentioned_ident should have importance > 0
        assert!(importance["boost_mentioned_ident"] > 0.0);
    }

    #[test]
    fn test_oat_sensitivity() {
        let baseline = ParameterPoint::default();

        // Simple evaluator
        let evaluator = |p: &ParameterPoint| p.pagerank_alpha;

        let curves = oat_sensitivity(&baseline, evaluator, 5);

        // Should have curves for all params
        assert!(curves.contains_key("pagerank_alpha"));

        // Alpha curve should be monotonically increasing (since evaluator = alpha)
        let alpha_curve = &curves["pagerank_alpha"];
        assert_eq!(alpha_curve.len(), 5);

        for i in 1..alpha_curve.len() {
            assert!(alpha_curve[i].1 >= alpha_curve[i - 1].1);
        }
    }
}
