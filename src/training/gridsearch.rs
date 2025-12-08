//! Parameter space exploration for hyperparameter optimization.
//!
//! ## Search Strategies
//!
//! | Strategy         | When to use                                          |
//! |------------------|------------------------------------------------------|
//! | Grid             | Small parameter space (<1000 combinations)           |
//! | LatinHypercube   | Medium space, want uniform coverage                  |
//! | Random           | Large space, baseline comparison                     |
//! | Bayesian         | Expensive evaluations, want smart sampling           |
//!
//! ## Latin Hypercube Sampling
//!
//! LHS ensures uniform coverage of each dimension independently while
//! maintaining good space-filling properties. For N samples in D dimensions,
//! each dimension is divided into N equal strata, and exactly one sample
//! is placed in each stratum per dimension.
//!
//! ## Bayesian Optimization
//!
//! Uses a Gaussian Process surrogate model to predict metric values for
//! unsampled points, then samples where Expected Improvement is highest.
//! Good for expensive evaluations where we want to minimize samples needed.
//!
//! ## Parameter Encoding
//!
//! All parameters are normalized to [0, 1] internally, then decoded to
//! their actual ranges for evaluation. This simplifies search algorithms.

use std::collections::HashMap;

use rand::prelude::*;
use serde::{Deserialize, Serialize};

use crate::types::RankingConfig;

/// A single point in parameter space.
///
/// All parameters are f64 to enable continuous optimization.
/// Integer parameters (like max_hops) are rounded at decode time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterPoint {
    // === PageRank ===
    /// Damping factor (typically 0.70-0.95)
    pub pagerank_alpha: f64,
    /// Chat file multiplier (typically 10-200)
    pub pagerank_chat_multiplier: f64,

    // === Depth weights ===
    /// Weight for root/shallow files (typically 0.5-2.0)
    pub depth_weight_root: f64,
    /// Weight for moderate depth (typically 0.2-1.0)
    pub depth_weight_moderate: f64,
    /// Weight for deep files (typically 0.05-0.5)
    pub depth_weight_deep: f64,
    /// Weight for vendor code (typically 0.001-0.1)
    pub depth_weight_vendor: f64,

    // === Boosts ===
    /// Mentioned identifier boost (typically 2-50)
    pub boost_mentioned_ident: f64,
    /// Mentioned file boost (typically 2-20)
    pub boost_mentioned_file: f64,
    /// Chat file boost (typically 5-100)
    pub boost_chat_file: f64,
    /// Temporal coupling boost (typically 1-10)
    pub boost_temporal_coupling: f64,
    /// Focus expansion boost (typically 1-20)
    pub boost_focus_expansion: f64,

    // === Git ===
    /// Recency decay half-life in days (typically 7-90)
    pub git_recency_decay_days: f64,
    /// Max recency boost (typically 2-20)
    pub git_recency_max_boost: f64,
    /// Churn threshold for "high churn" (typically 3-15)
    pub git_churn_threshold: f64,
    /// Max churn boost (typically 2-15)
    pub git_churn_max_boost: f64,

    // === Focus expansion ===
    /// Graph traversal decay per hop (typically 0.2-0.8)
    pub focus_decay: f64,
    /// Max hops for expansion (typically 1-3)
    pub focus_max_hops: f64,
}

impl ParameterPoint {
    /// Convert to a RankingConfig for actual evaluation.
    pub fn to_ranking_config(&self) -> RankingConfig {
        let mut config = RankingConfig::default();

        config.pagerank_alpha = self.pagerank_alpha;
        config.pagerank_chat_multiplier = self.pagerank_chat_multiplier;

        config.depth_weight_root = self.depth_weight_root;
        config.depth_weight_moderate = self.depth_weight_moderate;
        config.depth_weight_deep = self.depth_weight_deep;
        config.depth_weight_vendor = self.depth_weight_vendor;

        config.boost_mentioned_ident = self.boost_mentioned_ident;
        config.boost_mentioned_file = self.boost_mentioned_file;
        config.boost_chat_file = self.boost_chat_file;
        config.boost_temporal_coupling = self.boost_temporal_coupling;
        config.boost_focus_expansion = self.boost_focus_expansion;

        config.git_recency_decay_days = self.git_recency_decay_days;
        config.git_recency_max_boost = self.git_recency_max_boost;
        config.git_churn_threshold = self.git_churn_threshold.round() as usize;
        config.git_churn_max_boost = self.git_churn_max_boost;

        config
    }

    /// Get focus expansion parameters (not in RankingConfig).
    pub fn focus_params(&self) -> (f64, usize) {
        (self.focus_decay, self.focus_max_hops.round() as usize)
    }
}

impl Default for ParameterPoint {
    fn default() -> Self {
        let config = RankingConfig::default();
        Self {
            pagerank_alpha: config.pagerank_alpha,
            pagerank_chat_multiplier: config.pagerank_chat_multiplier,
            depth_weight_root: config.depth_weight_root,
            depth_weight_moderate: config.depth_weight_moderate,
            depth_weight_deep: config.depth_weight_deep,
            depth_weight_vendor: config.depth_weight_vendor,
            boost_mentioned_ident: config.boost_mentioned_ident,
            boost_mentioned_file: config.boost_mentioned_file,
            boost_chat_file: config.boost_chat_file,
            boost_temporal_coupling: config.boost_temporal_coupling,
            boost_focus_expansion: config.boost_focus_expansion,
            git_recency_decay_days: config.git_recency_decay_days,
            git_recency_max_boost: config.git_recency_max_boost,
            git_churn_threshold: config.git_churn_threshold as f64,
            git_churn_max_boost: config.git_churn_max_boost,
            focus_decay: 0.5,
            focus_max_hops: 2.0,
        }
    }
}

/// Range specification for a parameter.
#[derive(Debug, Clone)]
pub struct ParamRange {
    pub min: f64,
    pub max: f64,
    /// If true, sample in log space (good for multipliers/boosts)
    pub log_scale: bool,
}

impl ParamRange {
    pub fn linear(min: f64, max: f64) -> Self {
        Self {
            min,
            max,
            log_scale: false,
        }
    }

    pub fn log(min: f64, max: f64) -> Self {
        Self {
            min,
            max,
            log_scale: true,
        }
    }

    /// Convert normalized [0, 1] value to actual parameter value.
    pub fn decode(&self, normalized: f64) -> f64 {
        let t = normalized.clamp(0.0, 1.0);
        if self.log_scale {
            // Log-linear interpolation
            let log_min = self.min.ln();
            let log_max = self.max.ln();
            (log_min + t * (log_max - log_min)).exp()
        } else {
            self.min + t * (self.max - self.min)
        }
    }

    /// Convert actual value to normalized [0, 1].
    pub fn encode(&self, value: f64) -> f64 {
        if self.log_scale {
            let log_min = self.min.ln();
            let log_max = self.max.ln();
            let log_val = value.clamp(self.min, self.max).ln();
            (log_val - log_min) / (log_max - log_min)
        } else {
            (value - self.min) / (self.max - self.min)
        }
    }
}

/// Full parameter space definition with ranges for each parameter.
#[derive(Debug, Clone)]
pub struct ParameterGrid {
    pub ranges: HashMap<String, ParamRange>,
}

impl Default for ParameterGrid {
    fn default() -> Self {
        let mut ranges = HashMap::new();

        // PageRank
        ranges.insert("pagerank_alpha".into(), ParamRange::linear(0.70, 0.95));
        ranges.insert(
            "pagerank_chat_multiplier".into(),
            ParamRange::log(10.0, 200.0),
        );

        // Depth weights (log scale - multiplicative effect)
        ranges.insert("depth_weight_root".into(), ParamRange::linear(0.5, 2.0));
        ranges.insert("depth_weight_moderate".into(), ParamRange::linear(0.2, 1.0));
        ranges.insert("depth_weight_deep".into(), ParamRange::linear(0.05, 0.5));
        ranges.insert("depth_weight_vendor".into(), ParamRange::log(0.001, 0.1));

        // Boosts (log scale)
        ranges.insert("boost_mentioned_ident".into(), ParamRange::log(2.0, 50.0));
        ranges.insert("boost_mentioned_file".into(), ParamRange::log(2.0, 20.0));
        ranges.insert("boost_chat_file".into(), ParamRange::log(5.0, 100.0));
        ranges.insert("boost_temporal_coupling".into(), ParamRange::log(1.0, 10.0));
        ranges.insert("boost_focus_expansion".into(), ParamRange::log(1.0, 20.0));

        // Git
        ranges.insert(
            "git_recency_decay_days".into(),
            ParamRange::linear(7.0, 90.0),
        );
        ranges.insert("git_recency_max_boost".into(), ParamRange::log(2.0, 20.0));
        ranges.insert("git_churn_threshold".into(), ParamRange::linear(3.0, 15.0));
        ranges.insert("git_churn_max_boost".into(), ParamRange::log(2.0, 15.0));

        // Focus expansion
        ranges.insert("focus_decay".into(), ParamRange::linear(0.2, 0.8));
        ranges.insert("focus_max_hops".into(), ParamRange::linear(1.0, 3.0));

        Self { ranges }
    }
}

impl ParameterGrid {
    /// Decode a normalized vector [0, 1]^D into a ParameterPoint.
    pub fn decode(&self, normalized: &[f64]) -> ParameterPoint {
        let names = self.param_names();
        assert_eq!(normalized.len(), names.len(), "Dimension mismatch");

        let values: HashMap<_, _> = names
            .iter()
            .zip(normalized.iter())
            .map(|(name, &n)| {
                let range = &self.ranges[name];
                (name.as_str(), range.decode(n))
            })
            .collect();

        ParameterPoint {
            pagerank_alpha: values["pagerank_alpha"],
            pagerank_chat_multiplier: values["pagerank_chat_multiplier"],
            depth_weight_root: values["depth_weight_root"],
            depth_weight_moderate: values["depth_weight_moderate"],
            depth_weight_deep: values["depth_weight_deep"],
            depth_weight_vendor: values["depth_weight_vendor"],
            boost_mentioned_ident: values["boost_mentioned_ident"],
            boost_mentioned_file: values["boost_mentioned_file"],
            boost_chat_file: values["boost_chat_file"],
            boost_temporal_coupling: values["boost_temporal_coupling"],
            boost_focus_expansion: values["boost_focus_expansion"],
            git_recency_decay_days: values["git_recency_decay_days"],
            git_recency_max_boost: values["git_recency_max_boost"],
            git_churn_threshold: values["git_churn_threshold"],
            git_churn_max_boost: values["git_churn_max_boost"],
            focus_decay: values["focus_decay"],
            focus_max_hops: values["focus_max_hops"],
        }
    }

    /// Sorted list of parameter names (for consistent ordering).
    pub fn param_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.ranges.keys().cloned().collect();
        names.sort();
        names
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.ranges.len()
    }
}

/// Search strategy for parameter exploration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SearchStrategy {
    /// Full grid search (cartesian product of discrete values)
    Grid { points_per_dim: usize },
    /// Latin Hypercube Sampling (uniform space-filling)
    LatinHypercube,
    /// Random uniform sampling
    Random,
    /// Bayesian optimization with Expected Improvement
    Bayesian,
}

/// Sample points from parameter space using specified strategy.
pub fn sample_points(
    grid: &ParameterGrid,
    strategy: SearchStrategy,
    n_samples: usize,
    seed: u64,
) -> Vec<ParameterPoint> {
    let mut rng = StdRng::seed_from_u64(seed);

    match strategy {
        SearchStrategy::Grid { points_per_dim } => sample_grid(grid, points_per_dim),
        SearchStrategy::LatinHypercube => sample_lhs(grid, n_samples, &mut rng),
        SearchStrategy::Random => sample_random(grid, n_samples, &mut rng),
        SearchStrategy::Bayesian => {
            // Bayesian needs iterative sampling - start with LHS
            // The actual BO loop is in the training runner
            sample_lhs(grid, n_samples.min(20), &mut rng)
        }
    }
}

/// Full grid search: cartesian product of discrete values per dimension.
fn sample_grid(grid: &ParameterGrid, points_per_dim: usize) -> Vec<ParameterPoint> {
    let names = grid.param_names();
    let ndim = names.len();

    // Total combinations
    let total = points_per_dim.pow(ndim as u32);

    (0..total)
        .map(|idx| {
            // Convert linear index to multi-index
            let mut normalized = Vec::with_capacity(ndim);
            let mut remaining = idx;

            for _ in 0..ndim {
                let dim_idx = remaining % points_per_dim;
                remaining /= points_per_dim;

                // Map to [0, 1]
                let t = if points_per_dim > 1 {
                    dim_idx as f64 / (points_per_dim - 1) as f64
                } else {
                    0.5
                };
                normalized.push(t);
            }

            grid.decode(&normalized)
        })
        .collect()
}

/// Latin Hypercube Sampling for uniform space-filling coverage.
///
/// Each dimension is divided into N equal strata, and exactly one
/// sample is placed in each stratum per dimension.
fn sample_lhs<R: Rng>(grid: &ParameterGrid, n_samples: usize, rng: &mut R) -> Vec<ParameterPoint> {
    let ndim = grid.ndim();

    // For each dimension, create a random permutation of strata
    let mut strata: Vec<Vec<usize>> = (0..ndim)
        .map(|_| {
            let mut perm: Vec<usize> = (0..n_samples).collect();
            perm.shuffle(rng);
            perm
        })
        .collect();

    // Generate samples
    (0..n_samples)
        .map(|i| {
            let normalized: Vec<f64> = (0..ndim)
                .map(|d| {
                    let stratum = strata[d][i];
                    // Random point within stratum
                    let lower = stratum as f64 / n_samples as f64;
                    let upper = (stratum + 1) as f64 / n_samples as f64;
                    lower + rng.r#gen::<f64>() * (upper - lower)
                })
                .collect();

            grid.decode(&normalized)
        })
        .collect()
}

/// Random uniform sampling.
fn sample_random<R: Rng>(
    grid: &ParameterGrid,
    n_samples: usize,
    rng: &mut R,
) -> Vec<ParameterPoint> {
    let ndim = grid.ndim();

    (0..n_samples)
        .map(|_| {
            let normalized: Vec<f64> = (0..ndim).map(|_| rng.r#gen()).collect();
            grid.decode(&normalized)
        })
        .collect()
}

/// Bayesian optimization: sample next point based on history.
///
/// Uses a simple Expected Improvement acquisition function.
/// For a production implementation, use a proper GP library.
pub fn bayesian_next_sample<R: Rng>(
    grid: &ParameterGrid,
    history: &[(ParameterPoint, f64)], // (params, score)
    rng: &mut R,
) -> ParameterPoint {
    if history.is_empty() {
        // No history, return random sample
        let normalized: Vec<f64> = (0..grid.ndim()).map(|_| rng.r#gen()).collect();
        return grid.decode(&normalized);
    }

    // Find best score so far
    let best_score = history
        .iter()
        .map(|(_, s)| *s)
        .fold(f64::NEG_INFINITY, f64::max);

    // Simple acquisition: generate candidates and pick highest EI
    // (This is a placeholder - real BO would use a GP)
    let n_candidates = 1000;
    let candidates: Vec<ParameterPoint> = sample_random(grid, n_candidates, rng);

    // For now, use a heuristic: prefer points far from explored regions
    // with slight bias toward high-scoring regions
    candidates
        .into_iter()
        .max_by(|a, b| {
            let dist_a = min_distance_to_history(a, history);
            let dist_b = min_distance_to_history(b, history);

            // Balance exploration (distance) and exploitation (near good points)
            let score_a = dist_a + 0.3 * similarity_to_best(a, history, best_score);
            let score_b = dist_b + 0.3 * similarity_to_best(b, history, best_score);

            score_a
                .partial_cmp(&score_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or_else(|| grid.decode(&vec![0.5; grid.ndim()]))
}

/// Minimum normalized distance from point to any point in history.
fn min_distance_to_history(point: &ParameterPoint, history: &[(ParameterPoint, f64)]) -> f64 {
    history
        .iter()
        .map(|(h, _)| normalized_distance(point, h))
        .fold(f64::INFINITY, f64::min)
}

/// Similarity to high-scoring points in history.
fn similarity_to_best(
    point: &ParameterPoint,
    history: &[(ParameterPoint, f64)],
    best_score: f64,
) -> f64 {
    // Weight by how good each historical point was
    let mut total = 0.0;
    let mut weight_sum = 0.0;

    for (h, score) in history {
        let weight = (*score / best_score).max(0.0);
        let sim = 1.0 / (1.0 + normalized_distance(point, h));
        total += weight * sim;
        weight_sum += weight;
    }

    if weight_sum > 0.0 {
        total / weight_sum
    } else {
        0.0
    }
}

/// Normalized L2 distance between parameter points.
fn normalized_distance(a: &ParameterPoint, b: &ParameterPoint) -> f64 {
    let grid = ParameterGrid::default();
    let names = grid.param_names();

    let a_vals = point_to_vec(a, &grid);
    let b_vals = point_to_vec(b, &grid);

    let sum_sq: f64 = a_vals
        .iter()
        .zip(b_vals.iter())
        .map(|(av, bv)| (av - bv).powi(2))
        .sum();

    (sum_sq / names.len() as f64).sqrt()
}

/// Convert ParameterPoint to normalized vector.
fn point_to_vec(point: &ParameterPoint, grid: &ParameterGrid) -> Vec<f64> {
    let names = grid.param_names();
    names
        .iter()
        .map(|name| {
            let range = &grid.ranges[name];
            let value = match name.as_str() {
                "pagerank_alpha" => point.pagerank_alpha,
                "pagerank_chat_multiplier" => point.pagerank_chat_multiplier,
                "depth_weight_root" => point.depth_weight_root,
                "depth_weight_moderate" => point.depth_weight_moderate,
                "depth_weight_deep" => point.depth_weight_deep,
                "depth_weight_vendor" => point.depth_weight_vendor,
                "boost_mentioned_ident" => point.boost_mentioned_ident,
                "boost_mentioned_file" => point.boost_mentioned_file,
                "boost_chat_file" => point.boost_chat_file,
                "boost_temporal_coupling" => point.boost_temporal_coupling,
                "boost_focus_expansion" => point.boost_focus_expansion,
                "git_recency_decay_days" => point.git_recency_decay_days,
                "git_recency_max_boost" => point.git_recency_max_boost,
                "git_churn_threshold" => point.git_churn_threshold,
                "git_churn_max_boost" => point.git_churn_max_boost,
                "focus_decay" => point.focus_decay,
                "focus_max_hops" => point.focus_max_hops,
                _ => 0.5,
            };
            range.encode(value)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_range_linear() {
        let range = ParamRange::linear(0.0, 10.0);
        assert!((range.decode(0.0) - 0.0).abs() < 1e-6);
        assert!((range.decode(0.5) - 5.0).abs() < 1e-6);
        assert!((range.decode(1.0) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_param_range_log() {
        let range = ParamRange::log(1.0, 100.0);
        assert!((range.decode(0.0) - 1.0).abs() < 1e-6);
        assert!((range.decode(1.0) - 100.0).abs() < 1e-6);
        // Mid-point in log space: sqrt(1*100) = 10
        assert!((range.decode(0.5) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_param_range_roundtrip() {
        let range = ParamRange::log(2.0, 50.0);
        for v in [2.0, 10.0, 25.0, 50.0] {
            let encoded = range.encode(v);
            let decoded = range.decode(encoded);
            assert!((decoded - v).abs() < 1e-6, "Roundtrip failed for {}", v);
        }
    }

    #[test]
    fn test_grid_decode() {
        let grid = ParameterGrid::default();
        let ndim = grid.ndim();

        // All zeros should give min values
        let min_point = grid.decode(&vec![0.0; ndim]);
        assert!(min_point.pagerank_alpha >= 0.69);

        // All ones should give max values
        let max_point = grid.decode(&vec![1.0; ndim]);
        assert!(max_point.pagerank_alpha <= 0.96);
    }

    #[test]
    fn test_lhs_coverage() {
        let grid = ParameterGrid::default();
        let mut rng = StdRng::seed_from_u64(42);
        let samples = sample_lhs(&grid, 10, &mut rng);

        assert_eq!(samples.len(), 10);

        // Check that alpha values span the range reasonably
        let alphas: Vec<_> = samples.iter().map(|s| s.pagerank_alpha).collect();
        let min_alpha = alphas.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_alpha = alphas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        assert!(max_alpha - min_alpha > 0.1, "LHS should cover the range");
    }

    #[test]
    fn test_default_point_to_config() {
        let point = ParameterPoint::default();
        let config = point.to_ranking_config();

        // Check that default values match
        assert!((config.pagerank_alpha - 0.85).abs() < 1e-6);
        assert!((config.boost_mentioned_ident - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalized_distance_same() {
        let p = ParameterPoint::default();
        let dist = normalized_distance(&p, &p);
        assert!(dist < 1e-6, "Distance to self should be 0");
    }

    #[test]
    fn test_normalized_distance_different() {
        let mut p1 = ParameterPoint::default();
        let mut p2 = ParameterPoint::default();

        p1.pagerank_alpha = 0.7;
        p2.pagerank_alpha = 0.95;

        let dist = normalized_distance(&p1, &p2);
        assert!(dist > 0.0, "Different points should have distance > 0");
        assert!(dist < 1.0, "Normalized distance should be < 1");
    }
}
