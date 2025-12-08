//! Contextual boost calculation for ripmap.
//!
//! This module applies multiplicative boosts to base PageRank scores based on
//! various contextual signals:
//!
//! - **Mentioned identifiers**: Symbols explicitly referenced in queries/chat (10x)
//! - **Mentioned files**: Files explicitly named in context (5x)
//! - **Chat files**: Files being actively edited (20x)
//! - **Temporal coupling**: Files that co-change with chat files (3x)
//! - **Focus expansion**: Graph neighbors of focus symbols (5x)
//!
//! The final rank combines all signals multiplicatively:
//! ```text
//! final_rank = base_rank × boost × git_weight × caller_weight
//! ```
//!
//! This creates a powerful ranking signal that elevates contextually relevant
//! symbols to the top while still respecting structural importance (PageRank).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::types::{RankedTag, RankingConfig, Tag, TagKind};

/// Calculator for applying contextual boosts to symbol ranks.
///
/// Takes base PageRank scores and amplifies them based on contextual signals
/// like chat files, mentions, temporal coupling, and graph expansion.
///
/// The boost system is multiplicative: multiple signals stack to create
/// very high boosts for highly relevant symbols.
pub struct BoostCalculator {
    config: RankingConfig,
}

impl BoostCalculator {
    /// Create a new boost calculator with the given configuration.
    pub fn new(config: RankingConfig) -> Self {
        Self { config }
    }

    /// Apply boosts to tags based on contextual signals.
    ///
    /// Produces RankedTag instances with final scores computed as:
    /// ```text
    /// final_rank = base_rank × boost × git_weight × caller_weight
    /// ```
    ///
    /// # Arguments
    ///
    /// * `tags_by_file` - Map from absolute file path to its tags
    /// * `file_ranks` - PageRank scores for files (keyed by relative path)
    /// * `symbol_ranks` - Optional PageRank scores for symbols (keyed by (file, name))
    /// * `chat_fnames` - Files being actively edited (absolute paths)
    /// * `mentioned_fnames` - Files explicitly mentioned in query (relative paths)
    /// * `mentioned_idents` - Symbols explicitly mentioned in query
    /// * `temporal_boost_files` - Files that co-change with chat files (relative paths)
    /// * `git_weights` - Recency/churn-based git weights (keyed by relative path)
    /// * `caller_weights` - Reverse edge bias weights for debugging intent
    /// * `focus_expansion_weights` - Graph neighbor expansion weights for (file, symbol)
    ///
    /// # Returns
    ///
    /// Vector of RankedTag sorted by final rank (descending - highest first).
    /// Only includes definition tags (TagKind::Def).
    pub fn apply_boosts(
        &self,
        tags_by_file: &HashMap<String, Vec<Tag>>,
        file_ranks: &HashMap<String, f64>,
        symbol_ranks: Option<&HashMap<(Arc<str>, Arc<str>), f64>>,
        chat_fnames: &HashSet<String>,
        mentioned_fnames: &HashSet<String>,
        mentioned_idents: &HashSet<String>,
        temporal_boost_files: &HashSet<String>,
        git_weights: Option<&HashMap<String, f64>>,
        caller_weights: Option<&HashMap<String, f64>>,
        focus_expansion_weights: Option<&HashMap<(Arc<str>, Arc<str>), f64>>,
    ) -> Vec<RankedTag> {
        let mut result = Vec::new();

        // Convert chat_fnames (absolute paths) to relative for comparison
        let chat_rel_fnames: HashSet<String> =
            chat_fnames.iter().map(|f| extract_rel_fname(f)).collect();

        for (fname, tags) in tags_by_file {
            let rel_fname = extract_rel_fname(fname);

            // Get file-level rank and weights
            let file_rank = file_ranks.get(&rel_fname).copied().unwrap_or(0.0);
            let git_weight = git_weights
                .and_then(|w| w.get(&rel_fname))
                .copied()
                .unwrap_or(1.0);
            // Caller weight with hub damping: balance between "called = important"
            // and "utility function = noise".
            //
            // The interplay between boost_caller_weight and hub_damping:
            //   - boost_caller_weight amplifies the caller signal
            //   - hub_damping counteracts it to penalize "hub" nodes
            //
            // Effective formula:
            //   effective_boost = boost_caller_weight * (1.0 - hub_damping)
            //
            // With hub_damping = 0.0: full boost (called functions are important)
            // With hub_damping = 1.0: boost neutralized (caller count ignored)
            // With hub_damping > 1.0: penalty (hubs are downranked)
            let raw_caller_weight = caller_weights
                .and_then(|w| w.get(&rel_fname))
                .copied()
                .unwrap_or(1.0);

            // Apply hub damping: reduce or invert the caller weight effect
            let effective_boost = self.config.boost_caller_weight * (1.0 - self.config.hub_damping);
            let caller_weight = (1.0 + (raw_caller_weight - 1.0) * effective_boost).max(0.01);

            // Process only definition tags
            for tag in tags.iter().filter(|t| t.kind == TagKind::Def) {
                // Determine base rank: use symbol rank if available, else file rank
                let base_rank = symbol_ranks
                    .and_then(|sr| {
                        let key = (Arc::clone(&tag.rel_fname), Arc::clone(&tag.name));
                        sr.get(&key).copied()
                    })
                    .unwrap_or(file_rank);

                // Calculate contextual boost by multiplying all applicable signals
                let mut boost = 1.0;

                // Boost 1: Mentioned identifier (10x) - symbol explicitly in query
                if mentioned_idents.contains(tag.name.as_ref()) {
                    boost *= self.config.boost_mentioned_ident;
                }

                // Boost 2: Mentioned file (5x) - file explicitly in query
                if mentioned_fnames.contains(&rel_fname) {
                    boost *= self.config.boost_mentioned_file;
                }

                // Boost 3: Chat file (20x) - file being actively edited
                if chat_rel_fnames.contains(&rel_fname) {
                    boost *= self.config.boost_chat_file;
                }

                // Boost 4: Temporal coupling (3x) - co-changes with chat files
                if temporal_boost_files.contains(&rel_fname) {
                    boost *= self.config.boost_temporal_coupling;
                }

                // Boost 5: Focus expansion (5x × expansion_weight) - graph neighbor
                if let Some(expansion_weights) = focus_expansion_weights {
                    let key = (Arc::clone(&tag.rel_fname), Arc::clone(&tag.name));
                    if let Some(&expansion_weight) = expansion_weights.get(&key) {
                        boost *= self.config.boost_focus_expansion * expansion_weight;
                    }
                }

                // Compute final rank: base × boost × git × caller
                let final_rank = base_rank * boost * git_weight * caller_weight;

                result.push(RankedTag::new(final_rank, tag.clone()));
            }
        }

        // Sort by rank descending (highest first)
        result.sort();

        result
    }
}

/// Extract relative filename from absolute path.
///
/// Simplified heuristic: strips leading slash. In production, this would use
/// proper path resolution relative to repo root.
fn extract_rel_fname(abs_fname: &str) -> String {
    abs_fname.strip_prefix('/').unwrap_or(abs_fname).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TagKind;

    /// Helper to create a test tag
    fn make_tag(rel_fname: &str, name: &str, kind: TagKind) -> Tag {
        Tag {
            rel_fname: Arc::from(rel_fname),
            fname: Arc::from(format!("/{}", rel_fname)),
            line: 1,
            name: Arc::from(name),
            kind,
            node_type: Arc::from("function"),
            parent_name: None,
            parent_line: None,
            signature: None,
            fields: None,
            metadata: None,
        }
    }

    #[test]
    fn test_no_boosts() {
        // Without any contextual signals, should just use base ranks
        let config = RankingConfig::default();
        let calculator = BoostCalculator::new(config);

        let mut tags_by_file = HashMap::new();
        tags_by_file.insert(
            "/a.rs".to_string(),
            vec![make_tag("a.rs", "foo", TagKind::Def)],
        );

        let mut file_ranks = HashMap::new();
        file_ranks.insert("a.rs".to_string(), 0.5);

        let result = calculator.apply_boosts(
            &tags_by_file,
            &file_ranks,
            None,
            &HashSet::new(),
            &HashSet::new(),
            &HashSet::new(),
            &HashSet::new(),
            None,
            None,
            None,
        );

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].rank, 0.5); // base_rank × 1.0 (no boosts)
        assert_eq!(result[0].tag.name.as_ref(), "foo");
    }

    #[test]
    fn test_mentioned_ident_boost() {
        // Mentioned identifier should get 10x boost
        let config = RankingConfig::default();
        let calculator = BoostCalculator::new(config.clone());

        let mut tags_by_file = HashMap::new();
        tags_by_file.insert(
            "/a.rs".to_string(),
            vec![make_tag("a.rs", "foo", TagKind::Def)],
        );

        let mut file_ranks = HashMap::new();
        file_ranks.insert("a.rs".to_string(), 0.1);

        let mut mentioned_idents = HashSet::new();
        mentioned_idents.insert("foo".to_string());

        let result = calculator.apply_boosts(
            &tags_by_file,
            &file_ranks,
            None,
            &HashSet::new(),
            &HashSet::new(),
            &mentioned_idents,
            &HashSet::new(),
            None,
            None,
            None,
        );

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].rank, 0.1 * config.boost_mentioned_ident); // 0.1 × 10 = 1.0
    }

    #[test]
    fn test_mentioned_file_boost() {
        // Mentioned file should get 5x boost
        let config = RankingConfig::default();
        let calculator = BoostCalculator::new(config.clone());

        let mut tags_by_file = HashMap::new();
        tags_by_file.insert(
            "/a.rs".to_string(),
            vec![make_tag("a.rs", "foo", TagKind::Def)],
        );

        let mut file_ranks = HashMap::new();
        file_ranks.insert("a.rs".to_string(), 0.2);

        let mut mentioned_fnames = HashSet::new();
        mentioned_fnames.insert("a.rs".to_string());

        let result = calculator.apply_boosts(
            &tags_by_file,
            &file_ranks,
            None,
            &HashSet::new(),
            &mentioned_fnames,
            &HashSet::new(),
            &HashSet::new(),
            None,
            None,
            None,
        );

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].rank, 0.2 * config.boost_mentioned_file); // 0.2 × 5 = 1.0
    }

    #[test]
    fn test_chat_file_boost() {
        // Chat file should get 20x boost
        let config = RankingConfig::default();
        let calculator = BoostCalculator::new(config.clone());

        let mut tags_by_file = HashMap::new();
        tags_by_file.insert(
            "/a.rs".to_string(),
            vec![make_tag("a.rs", "foo", TagKind::Def)],
        );

        let mut file_ranks = HashMap::new();
        file_ranks.insert("a.rs".to_string(), 0.05);

        let mut chat_fnames = HashSet::new();
        chat_fnames.insert("/a.rs".to_string()); // Absolute path

        let result = calculator.apply_boosts(
            &tags_by_file,
            &file_ranks,
            None,
            &chat_fnames,
            &HashSet::new(),
            &HashSet::new(),
            &HashSet::new(),
            None,
            None,
            None,
        );

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].rank, 0.05 * config.boost_chat_file); // 0.05 × 20 = 1.0
    }

    #[test]
    fn test_temporal_coupling_boost() {
        // Temporal coupling should get 3x boost
        let config = RankingConfig::default();
        let calculator = BoostCalculator::new(config.clone());

        let mut tags_by_file = HashMap::new();
        tags_by_file.insert(
            "/a.rs".to_string(),
            vec![make_tag("a.rs", "foo", TagKind::Def)],
        );

        let mut file_ranks = HashMap::new();
        file_ranks.insert("a.rs".to_string(), 0.5);

        let mut temporal_boost_files = HashSet::new();
        temporal_boost_files.insert("a.rs".to_string());

        let result = calculator.apply_boosts(
            &tags_by_file,
            &file_ranks,
            None,
            &HashSet::new(),
            &HashSet::new(),
            &HashSet::new(),
            &temporal_boost_files,
            None,
            None,
            None,
        );

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].rank, 0.5 * config.boost_temporal_coupling); // 0.5 × 3 = 1.5
    }

    #[test]
    fn test_multiple_boosts_multiply() {
        // Multiple boosts should stack multiplicatively
        let config = RankingConfig::default();
        let calculator = BoostCalculator::new(config.clone());

        let mut tags_by_file = HashMap::new();
        tags_by_file.insert(
            "/a.rs".to_string(),
            vec![make_tag("a.rs", "foo", TagKind::Def)],
        );

        let mut file_ranks = HashMap::new();
        file_ranks.insert("a.rs".to_string(), 0.01);

        let mut mentioned_idents = HashSet::new();
        mentioned_idents.insert("foo".to_string());

        let mut chat_fnames = HashSet::new();
        chat_fnames.insert("/a.rs".to_string());

        let result = calculator.apply_boosts(
            &tags_by_file,
            &file_ranks,
            None,
            &chat_fnames,
            &HashSet::new(),
            &mentioned_idents,
            &HashSet::new(),
            None,
            None,
            None,
        );

        assert_eq!(result.len(), 1);
        // 0.01 × 10 (mentioned_ident) × 20 (chat_file) = 2.0
        assert_eq!(
            result[0].rank,
            0.01 * config.boost_mentioned_ident * config.boost_chat_file
        );
    }

    #[test]
    fn test_git_weight_multiplier() {
        // Git weight should multiply into final rank
        let config = RankingConfig::default();
        let calculator = BoostCalculator::new(config);

        let mut tags_by_file = HashMap::new();
        tags_by_file.insert(
            "/a.rs".to_string(),
            vec![make_tag("a.rs", "foo", TagKind::Def)],
        );

        let mut file_ranks = HashMap::new();
        file_ranks.insert("a.rs".to_string(), 1.0);

        let mut git_weights = HashMap::new();
        git_weights.insert("a.rs".to_string(), 2.0);

        let result = calculator.apply_boosts(
            &tags_by_file,
            &file_ranks,
            None,
            &HashSet::new(),
            &HashSet::new(),
            &HashSet::new(),
            &HashSet::new(),
            Some(&git_weights),
            None,
            None,
        );

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].rank, 1.0 * 2.0); // base × git_weight
    }

    #[test]
    fn test_caller_weight_multiplier() {
        // Caller weight should multiply into final rank
        let config = RankingConfig::default();
        let calculator = BoostCalculator::new(config);

        let mut tags_by_file = HashMap::new();
        tags_by_file.insert(
            "/a.rs".to_string(),
            vec![make_tag("a.rs", "foo", TagKind::Def)],
        );

        let mut file_ranks = HashMap::new();
        file_ranks.insert("a.rs".to_string(), 1.0);

        let mut caller_weights = HashMap::new();
        caller_weights.insert("a.rs".to_string(), 1.5);

        let result = calculator.apply_boosts(
            &tags_by_file,
            &file_ranks,
            None,
            &HashSet::new(),
            &HashSet::new(),
            &HashSet::new(),
            &HashSet::new(),
            None,
            Some(&caller_weights),
            None,
        );

        assert_eq!(result.len(), 1);
        // With boost_caller_weight=2.0 (default) and raw=1.5:
        // caller_weight = 1.0 + (1.5 - 1.0) * 2.0 = 2.0
        assert_eq!(result[0].rank, 1.0 * 2.0); // base × scaled_caller_weight
    }

    #[test]
    fn test_focus_expansion_weight() {
        // Focus expansion weight should boost graph neighbors
        let config = RankingConfig::default();
        let calculator = BoostCalculator::new(config.clone());

        let mut tags_by_file = HashMap::new();
        tags_by_file.insert(
            "/a.rs".to_string(),
            vec![make_tag("a.rs", "foo", TagKind::Def)],
        );

        let mut file_ranks = HashMap::new();
        file_ranks.insert("a.rs".to_string(), 0.1);

        let mut focus_expansion_weights = HashMap::new();
        focus_expansion_weights.insert((Arc::from("a.rs"), Arc::from("foo")), 0.8);

        let result = calculator.apply_boosts(
            &tags_by_file,
            &file_ranks,
            None,
            &HashSet::new(),
            &HashSet::new(),
            &HashSet::new(),
            &HashSet::new(),
            None,
            None,
            Some(&focus_expansion_weights),
        );

        assert_eq!(result.len(), 1);
        // 0.1 × (5.0 × 0.8) = 0.4
        assert_eq!(result[0].rank, 0.1 * config.boost_focus_expansion * 0.8);
    }

    #[test]
    fn test_symbol_rank_overrides_file_rank() {
        // When symbol rank is available, use it instead of file rank
        let config = RankingConfig::default();
        let calculator = BoostCalculator::new(config);

        let mut tags_by_file = HashMap::new();
        tags_by_file.insert(
            "/a.rs".to_string(),
            vec![make_tag("a.rs", "foo", TagKind::Def)],
        );

        let mut file_ranks = HashMap::new();
        file_ranks.insert("a.rs".to_string(), 0.1);

        let mut symbol_ranks = HashMap::new();
        symbol_ranks.insert((Arc::from("a.rs"), Arc::from("foo")), 0.9);

        let result = calculator.apply_boosts(
            &tags_by_file,
            &file_ranks,
            Some(&symbol_ranks),
            &HashSet::new(),
            &HashSet::new(),
            &HashSet::new(),
            &HashSet::new(),
            None,
            None,
            None,
        );

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].rank, 0.9); // Uses symbol rank, not file rank
    }

    #[test]
    fn test_only_definitions_included() {
        // Only TagKind::Def should be included, not TagKind::Ref
        let config = RankingConfig::default();
        let calculator = BoostCalculator::new(config);

        let mut tags_by_file = HashMap::new();
        tags_by_file.insert(
            "/a.rs".to_string(),
            vec![
                make_tag("a.rs", "foo", TagKind::Def),
                make_tag("a.rs", "bar", TagKind::Ref),
            ],
        );

        let mut file_ranks = HashMap::new();
        file_ranks.insert("a.rs".to_string(), 1.0);

        let result = calculator.apply_boosts(
            &tags_by_file,
            &file_ranks,
            None,
            &HashSet::new(),
            &HashSet::new(),
            &HashSet::new(),
            &HashSet::new(),
            None,
            None,
            None,
        );

        assert_eq!(result.len(), 1); // Only "foo" (Def), not "bar" (Ref)
        assert_eq!(result[0].tag.name.as_ref(), "foo");
    }

    #[test]
    fn test_sorting_descending() {
        // Results should be sorted by rank descending (highest first)
        let config = RankingConfig::default();
        let calculator = BoostCalculator::new(config);

        let mut tags_by_file = HashMap::new();
        tags_by_file.insert(
            "/a.rs".to_string(),
            vec![
                make_tag("a.rs", "low", TagKind::Def),
                make_tag("a.rs", "medium", TagKind::Def),
                make_tag("a.rs", "high", TagKind::Def),
            ],
        );

        let mut file_ranks = HashMap::new();
        file_ranks.insert("a.rs".to_string(), 1.0);

        let mut symbol_ranks = HashMap::new();
        symbol_ranks.insert((Arc::from("a.rs"), Arc::from("low")), 0.1);
        symbol_ranks.insert((Arc::from("a.rs"), Arc::from("medium")), 0.5);
        symbol_ranks.insert((Arc::from("a.rs"), Arc::from("high")), 0.9);

        let result = calculator.apply_boosts(
            &tags_by_file,
            &file_ranks,
            Some(&symbol_ranks),
            &HashSet::new(),
            &HashSet::new(),
            &HashSet::new(),
            &HashSet::new(),
            None,
            None,
            None,
        );

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].tag.name.as_ref(), "high");
        assert_eq!(result[1].tag.name.as_ref(), "medium");
        assert_eq!(result[2].tag.name.as_ref(), "low");
    }

    #[test]
    fn test_combined_all_weights() {
        // Test all weight/boost combinations together
        let config = RankingConfig::default();
        let calculator = BoostCalculator::new(config.clone());

        let mut tags_by_file = HashMap::new();
        tags_by_file.insert(
            "/a.rs".to_string(),
            vec![make_tag("a.rs", "foo", TagKind::Def)],
        );

        let mut file_ranks = HashMap::new();
        file_ranks.insert("a.rs".to_string(), 0.1);

        let mut mentioned_idents = HashSet::new();
        mentioned_idents.insert("foo".to_string());

        let mut mentioned_fnames = HashSet::new();
        mentioned_fnames.insert("a.rs".to_string());

        let mut chat_fnames = HashSet::new();
        chat_fnames.insert("/a.rs".to_string());

        let mut temporal_boost_files = HashSet::new();
        temporal_boost_files.insert("a.rs".to_string());

        let mut git_weights = HashMap::new();
        git_weights.insert("a.rs".to_string(), 2.0);

        let mut caller_weights = HashMap::new();
        caller_weights.insert("a.rs".to_string(), 1.5);

        let mut focus_expansion_weights = HashMap::new();
        focus_expansion_weights.insert((Arc::from("a.rs"), Arc::from("foo")), 0.5);

        let result = calculator.apply_boosts(
            &tags_by_file,
            &file_ranks,
            None,
            &chat_fnames,
            &mentioned_fnames,
            &mentioned_idents,
            &temporal_boost_files,
            Some(&git_weights),
            Some(&caller_weights),
            Some(&focus_expansion_weights),
        );

        assert_eq!(result.len(), 1);

        // Expected: 0.1 × 10 (ident) × 5 (file) × 20 (chat) × 3 (temporal)
        //           × (5 × 0.5) (focus) × 2.0 (git) × scaled_caller
        // With raw_caller=1.5, boost_caller_weight=2.0:
        //   scaled_caller = 1.0 + (1.5 - 1.0) * 2.0 = 2.0
        let raw_caller = 1.5;
        let scaled_caller = 1.0 + (raw_caller - 1.0) * config.boost_caller_weight;
        let expected = 0.1
            * config.boost_mentioned_ident
            * config.boost_mentioned_file
            * config.boost_chat_file
            * config.boost_temporal_coupling
            * (config.boost_focus_expansion * 0.5)
            * 2.0
            * scaled_caller;

        assert!((result[0].rank - expected).abs() < 1e-6);
    }

    #[test]
    fn test_extract_rel_fname() {
        assert_eq!(extract_rel_fname("/a.rs"), "a.rs");
        assert_eq!(extract_rel_fname("/src/lib.rs"), "src/lib.rs");
        assert_eq!(extract_rel_fname("no_slash.rs"), "no_slash.rs");
    }
}
