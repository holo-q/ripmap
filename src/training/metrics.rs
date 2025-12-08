//! Information retrieval evaluation metrics for ranking quality.
//!
//! ## Metrics Overview
//!
//! | Metric         | What it measures                                    | Range   |
//! |----------------|-----------------------------------------------------|---------|
//! | NDCG@k         | Quality of top-k ranking (graded relevance)         | 0.0-1.0 |
//! | Precision@k    | Fraction of top-k that are relevant (binary)        | 0.0-1.0 |
//! | Recall@k       | Fraction of relevant items in top-k                 | 0.0-1.0 |
//! | MRR            | Reciprocal rank of first relevant item              | 0.0-1.0 |
//! | MAP            | Mean average precision (all relevant items)         | 0.0-1.0 |
//!
//! ## Weighted vs Binary Relevance
//!
//! Git-derived ground truth has **graded relevance**:
//! - Files that always change together: relevance ~1.0
//! - Occasional co-changes: relevance ~0.3
//! - Never co-changed: relevance 0.0
//!
//! NDCG is the primary metric because it handles graded relevance.
//! Precision/Recall/MRR binarize at a threshold (default 0.1).
//!
//! ## Aggregation
//!
//! When evaluating over multiple cases, we report:
//! - Mean across cases (primary)
//! - Std dev (for significance testing)
//! - Median (robust to outliers)
//! - Weighted mean (by case quality)

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Aggregated evaluation metrics over a dataset.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvalMetrics {
    /// NDCG@10 (primary metric for graded relevance)
    pub ndcg_at_10: f64,
    /// NDCG@5 (stricter top ranking quality)
    pub ndcg_at_5: f64,

    /// Precision@k (binary relevance at threshold 0.1)
    pub precision_at_5: f64,
    pub precision_at_10: f64,

    /// Recall@k
    pub recall_at_5: f64,
    pub recall_at_10: f64,

    /// Mean Reciprocal Rank
    pub mrr: f64,

    /// Mean Average Precision
    pub map: f64,

    /// Number of cases evaluated
    pub n_cases: usize,

    /// Standard deviations (for significance testing)
    pub ndcg_at_10_std: f64,
    pub mrr_std: f64,
}

impl EvalMetrics {
    /// Aggregate metrics from per-case results.
    pub fn aggregate(per_case: &[CaseMetrics]) -> Self {
        if per_case.is_empty() {
            return Self::default();
        }

        let n = per_case.len() as f64;

        let ndcg_10: Vec<_> = per_case.iter().map(|c| c.ndcg_at_10).collect();
        let ndcg_5: Vec<_> = per_case.iter().map(|c| c.ndcg_at_5).collect();
        let p_5: Vec<_> = per_case.iter().map(|c| c.precision_at_5).collect();
        let p_10: Vec<_> = per_case.iter().map(|c| c.precision_at_10).collect();
        let r_5: Vec<_> = per_case.iter().map(|c| c.recall_at_5).collect();
        let r_10: Vec<_> = per_case.iter().map(|c| c.recall_at_10).collect();
        let mrr: Vec<_> = per_case.iter().map(|c| c.mrr).collect();
        let map: Vec<_> = per_case.iter().map(|c| c.map).collect();

        Self {
            ndcg_at_10: mean(&ndcg_10),
            ndcg_at_5: mean(&ndcg_5),
            precision_at_5: mean(&p_5),
            precision_at_10: mean(&p_10),
            recall_at_5: mean(&r_5),
            recall_at_10: mean(&r_10),
            mrr: mean(&mrr),
            map: mean(&map),
            n_cases: per_case.len(),
            ndcg_at_10_std: std_dev(&ndcg_10),
            mrr_std: std_dev(&mrr),
        }
    }

    /// Aggregate with case weighting (weight by case quality).
    pub fn aggregate_weighted(per_case: &[(CaseMetrics, f64)]) -> Self {
        if per_case.is_empty() {
            return Self::default();
        }

        let total_weight: f64 = per_case.iter().map(|(_, w)| w).sum();
        if total_weight == 0.0 {
            return Self::default();
        }

        let weighted_mean = |f: fn(&CaseMetrics) -> f64| -> f64 {
            per_case.iter().map(|(c, w)| f(c) * w).sum::<f64>() / total_weight
        };

        Self {
            ndcg_at_10: weighted_mean(|c| c.ndcg_at_10),
            ndcg_at_5: weighted_mean(|c| c.ndcg_at_5),
            precision_at_5: weighted_mean(|c| c.precision_at_5),
            precision_at_10: weighted_mean(|c| c.precision_at_10),
            recall_at_5: weighted_mean(|c| c.recall_at_5),
            recall_at_10: weighted_mean(|c| c.recall_at_10),
            mrr: weighted_mean(|c| c.mrr),
            map: weighted_mean(|c| c.map),
            n_cases: per_case.len(),
            ndcg_at_10_std: 0.0, // TODO: weighted std dev
            mrr_std: 0.0,
        }
    }
}

/// Metrics for a single evaluation case.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CaseMetrics {
    pub ndcg_at_5: f64,
    pub ndcg_at_10: f64,
    pub precision_at_5: f64,
    pub precision_at_10: f64,
    pub recall_at_5: f64,
    pub recall_at_10: f64,
    pub mrr: f64,
    pub map: f64,
}

impl CaseMetrics {
    /// Compute all metrics for a single case.
    ///
    /// # Arguments
    ///
    /// * `ranking` - Our output ranking (file paths in order, best first)
    /// * `ground_truth` - Expected relevant files with weights: (file, relevance)
    /// * `relevance_threshold` - Minimum weight to count as "relevant" for binary metrics
    pub fn compute(
        ranking: &[String],
        ground_truth: &[(String, f64)],
        relevance_threshold: f64,
    ) -> Self {
        Self {
            ndcg_at_5: weighted_ndcg(ranking, ground_truth, 5),
            ndcg_at_10: weighted_ndcg(ranking, ground_truth, 10),
            precision_at_5: precision_at_k(ranking, ground_truth, 5, relevance_threshold),
            precision_at_10: precision_at_k(ranking, ground_truth, 10, relevance_threshold),
            recall_at_5: recall_at_k(ranking, ground_truth, 5, relevance_threshold),
            recall_at_10: recall_at_k(ranking, ground_truth, 10, relevance_threshold),
            mrr: mean_reciprocal_rank(ranking, ground_truth, relevance_threshold),
            map: mean_average_precision(ranking, ground_truth, relevance_threshold),
        }
    }
}

/// Normalized Discounted Cumulative Gain at position k.
///
/// NDCG handles **graded relevance**: files with higher coupling weight
/// contribute more to the score, and rank position matters (early = better).
///
/// ```text
/// DCG@k = Σᵢ (rel[i] / log₂(i + 2))  for i in 0..k
/// NDCG@k = DCG@k / IDCG@k
/// ```
///
/// Where IDCG is the ideal DCG (perfect ranking by relevance).
///
/// # Arguments
///
/// * `ranking` - Our output (file paths in ranked order)
/// * `ground_truth` - (file, relevance_weight) pairs from git oracle
/// * `k` - Cutoff position
///
/// # Returns
///
/// NDCG score in range [0.0, 1.0]. Higher is better.
pub fn weighted_ndcg(
    ranking: &[String],
    ground_truth: &[(String, f64)],
    k: usize,
) -> f64 {
    if ground_truth.is_empty() {
        return 0.0;
    }

    let truth_map: HashMap<_, _> = ground_truth.iter().cloned().collect();

    // DCG: sum of relevance / log₂(rank + 2)
    let dcg: f64 = ranking
        .iter()
        .take(k)
        .enumerate()
        .map(|(rank, file)| {
            let relevance = truth_map.get(file).copied().unwrap_or(0.0);
            relevance / (rank as f64 + 2.0).log2()
        })
        .sum();

    // Ideal DCG: sort ground truth by relevance descending
    let mut ideal_weights: Vec<_> = ground_truth.iter().map(|(_, w)| *w).collect();
    ideal_weights.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let idcg: f64 = ideal_weights
        .iter()
        .take(k)
        .enumerate()
        .map(|(rank, &rel)| rel / (rank as f64 + 2.0).log2())
        .sum();

    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

/// Precision at position k.
///
/// Fraction of top-k results that are relevant (binary relevance).
/// Files with weight >= threshold count as relevant.
///
/// ```text
/// P@k = |relevant ∩ top-k| / k
/// ```
pub fn precision_at_k(
    ranking: &[String],
    ground_truth: &[(String, f64)],
    k: usize,
    threshold: f64,
) -> f64 {
    let relevant: std::collections::HashSet<_> = ground_truth
        .iter()
        .filter(|(_, w)| *w >= threshold)
        .map(|(f, _)| f.as_str())
        .collect();

    if relevant.is_empty() || k == 0 {
        return 0.0;
    }

    let hits = ranking
        .iter()
        .take(k)
        .filter(|f| relevant.contains(f.as_str()))
        .count();

    hits as f64 / k.min(ranking.len()) as f64
}

/// Recall at position k.
///
/// Fraction of relevant items that appear in top-k.
///
/// ```text
/// R@k = |relevant ∩ top-k| / |relevant|
/// ```
pub fn recall_at_k(
    ranking: &[String],
    ground_truth: &[(String, f64)],
    k: usize,
    threshold: f64,
) -> f64 {
    let relevant: std::collections::HashSet<_> = ground_truth
        .iter()
        .filter(|(_, w)| *w >= threshold)
        .map(|(f, _)| f.as_str())
        .collect();

    if relevant.is_empty() {
        return 0.0;
    }

    let top_k: std::collections::HashSet<_> = ranking
        .iter()
        .take(k)
        .map(|f| f.as_str())
        .collect();

    let hits = relevant.intersection(&top_k).count();

    hits as f64 / relevant.len() as f64
}

/// Mean Reciprocal Rank.
///
/// Reciprocal of the rank of the first relevant item.
/// Measures how quickly we surface ANY relevant result.
///
/// ```text
/// MRR = 1 / rank_of_first_relevant
/// ```
///
/// If no relevant item in ranking, returns 0.
pub fn mean_reciprocal_rank(
    ranking: &[String],
    ground_truth: &[(String, f64)],
    threshold: f64,
) -> f64 {
    let relevant: std::collections::HashSet<_> = ground_truth
        .iter()
        .filter(|(_, w)| *w >= threshold)
        .map(|(f, _)| f.as_str())
        .collect();

    for (rank, file) in ranking.iter().enumerate() {
        if relevant.contains(file.as_str()) {
            return 1.0 / (rank as f64 + 1.0);
        }
    }

    0.0
}

/// Mean Average Precision.
///
/// Average of precision values at each relevant item's rank.
/// Measures overall quality of ranking for all relevant items.
///
/// ```text
/// AP = (1/|relevant|) × Σᵢ P@i × rel(i)
/// ```
///
/// Where the sum is over all positions and rel(i) = 1 if item i is relevant.
pub fn mean_average_precision(
    ranking: &[String],
    ground_truth: &[(String, f64)],
    threshold: f64,
) -> f64 {
    let relevant: std::collections::HashSet<_> = ground_truth
        .iter()
        .filter(|(_, w)| *w >= threshold)
        .map(|(f, _)| f.as_str())
        .collect();

    if relevant.is_empty() {
        return 0.0;
    }

    let mut relevant_seen = 0;
    let mut precision_sum = 0.0;

    for (rank, file) in ranking.iter().enumerate() {
        if relevant.contains(file.as_str()) {
            relevant_seen += 1;
            // Precision at this rank
            let precision = relevant_seen as f64 / (rank as f64 + 1.0);
            precision_sum += precision;
        }
    }

    precision_sum / relevant.len() as f64
}

// === Utility functions ===

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let variance = values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndcg_perfect_ranking() {
        // Perfect ranking: items in order of relevance
        let ranking = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let truth = vec![
            ("a".to_string(), 1.0),
            ("b".to_string(), 0.5),
            ("c".to_string(), 0.25),
        ];

        let ndcg = weighted_ndcg(&ranking, &truth, 3);
        assert!((ndcg - 1.0).abs() < 1e-6, "Perfect ranking should have NDCG=1.0, got {}", ndcg);
    }

    #[test]
    fn test_ndcg_reversed_ranking() {
        // Worst ranking: items in reverse order
        let ranking = vec!["c".to_string(), "b".to_string(), "a".to_string()];
        let truth = vec![
            ("a".to_string(), 1.0),
            ("b".to_string(), 0.5),
            ("c".to_string(), 0.25),
        ];

        let ndcg = weighted_ndcg(&ranking, &truth, 3);
        assert!(ndcg < 1.0, "Reversed ranking should have NDCG < 1.0, got {}", ndcg);
        assert!(ndcg > 0.0, "Reversed ranking should have NDCG > 0.0");
    }

    #[test]
    fn test_ndcg_partial_match() {
        // Only some items in ranking
        let ranking = vec!["a".to_string(), "x".to_string(), "y".to_string()];
        let truth = vec![
            ("a".to_string(), 1.0),
            ("b".to_string(), 0.5),
        ];

        let ndcg = weighted_ndcg(&ranking, &truth, 3);
        assert!(ndcg > 0.0, "Should get credit for 'a'");
        assert!(ndcg < 1.0, "Missing 'b' should hurt score");
    }

    #[test]
    fn test_precision_at_k() {
        let ranking = vec![
            "a".to_string(),
            "b".to_string(),
            "x".to_string(),
            "c".to_string(),
            "y".to_string(),
        ];
        let truth = vec![
            ("a".to_string(), 1.0),
            ("b".to_string(), 0.8),
            ("c".to_string(), 0.6),
        ];

        // P@2 = 2/2 (both a and b are relevant)
        let p2 = precision_at_k(&ranking, &truth, 2, 0.5);
        assert!((p2 - 1.0).abs() < 1e-6, "P@2 should be 1.0, got {}", p2);

        // P@3 = 2/3 (a, b relevant, x not)
        let p3 = precision_at_k(&ranking, &truth, 3, 0.5);
        assert!((p3 - 2.0 / 3.0).abs() < 1e-6, "P@3 should be 0.667, got {}", p3);

        // P@5 = 3/5 (a, b, c relevant)
        let p5 = precision_at_k(&ranking, &truth, 5, 0.5);
        assert!((p5 - 0.6).abs() < 1e-6, "P@5 should be 0.6, got {}", p5);
    }

    #[test]
    fn test_recall_at_k() {
        let ranking = vec![
            "a".to_string(),
            "x".to_string(),
            "b".to_string(),
        ];
        let truth = vec![
            ("a".to_string(), 1.0),
            ("b".to_string(), 0.8),
            ("c".to_string(), 0.6), // not in ranking
        ];

        // R@1 = 1/3 (only a in top-1)
        let r1 = recall_at_k(&ranking, &truth, 1, 0.5);
        assert!((r1 - 1.0 / 3.0).abs() < 1e-6, "R@1 should be 0.333, got {}", r1);

        // R@3 = 2/3 (a, b in top-3, c missing)
        let r3 = recall_at_k(&ranking, &truth, 3, 0.5);
        assert!((r3 - 2.0 / 3.0).abs() < 1e-6, "R@3 should be 0.667, got {}", r3);
    }

    #[test]
    fn test_mrr() {
        // Relevant item at rank 3 (0-indexed: 2)
        let ranking = vec![
            "x".to_string(),
            "y".to_string(),
            "a".to_string(),
        ];
        let truth = vec![("a".to_string(), 1.0)];

        let mrr = mean_reciprocal_rank(&ranking, &truth, 0.5);
        assert!((mrr - 1.0 / 3.0).abs() < 1e-6, "MRR should be 0.333, got {}", mrr);
    }

    #[test]
    fn test_mrr_first_position() {
        let ranking = vec!["a".to_string(), "x".to_string()];
        let truth = vec![("a".to_string(), 1.0)];

        let mrr = mean_reciprocal_rank(&ranking, &truth, 0.5);
        assert!((mrr - 1.0).abs() < 1e-6, "MRR should be 1.0 when first is relevant");
    }

    #[test]
    fn test_map() {
        // a at rank 1, b at rank 3
        let ranking = vec![
            "a".to_string(),
            "x".to_string(),
            "b".to_string(),
            "y".to_string(),
        ];
        let truth = vec![
            ("a".to_string(), 1.0),
            ("b".to_string(), 0.8),
        ];

        // At rank 1: P=1/1=1.0, At rank 3: P=2/3
        // MAP = (1.0 + 0.667) / 2 = 0.833
        let map = mean_average_precision(&ranking, &truth, 0.5);
        let expected = (1.0 + 2.0 / 3.0) / 2.0;
        assert!((map - expected).abs() < 1e-6, "MAP should be {}, got {}", expected, map);
    }

    #[test]
    fn test_case_metrics() {
        let ranking = vec![
            "a".to_string(),
            "b".to_string(),
            "x".to_string(),
            "c".to_string(),
        ];
        let truth = vec![
            ("a".to_string(), 1.0),
            ("b".to_string(), 0.8),
            ("c".to_string(), 0.6),
        ];

        let metrics = CaseMetrics::compute(&ranking, &truth, 0.5);

        assert!(metrics.ndcg_at_5 > 0.9, "NDCG@5 should be high");
        assert!(metrics.precision_at_5 > 0.5, "P@5 should be decent");
        assert!(metrics.mrr > 0.9, "MRR should be ~1.0");
    }

    #[test]
    fn test_aggregate() {
        let cases = vec![
            CaseMetrics {
                ndcg_at_10: 0.8,
                ndcg_at_5: 0.9,
                precision_at_5: 0.6,
                precision_at_10: 0.5,
                recall_at_5: 0.4,
                recall_at_10: 0.6,
                mrr: 1.0,
                map: 0.7,
            },
            CaseMetrics {
                ndcg_at_10: 0.6,
                ndcg_at_5: 0.7,
                precision_at_5: 0.4,
                precision_at_10: 0.3,
                recall_at_5: 0.2,
                recall_at_10: 0.4,
                mrr: 0.5,
                map: 0.5,
            },
        ];

        let agg = EvalMetrics::aggregate(&cases);

        assert!((agg.ndcg_at_10 - 0.7).abs() < 1e-6);
        assert!((agg.mrr - 0.75).abs() < 1e-6);
        assert_eq!(agg.n_cases, 2);
    }

    #[test]
    fn test_std_dev() {
        // 0, 10 -> mean 5, std dev ~7.07
        let values = vec![0.0, 10.0];
        let sd = std_dev(&values);
        assert!((sd - 7.071).abs() < 0.01, "Std dev should be ~7.07, got {}", sd);
    }
}
