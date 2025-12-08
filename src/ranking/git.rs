//! Git-based weighting and temporal analysis.
//!
//! This module extracts temporal signals from git history to inform ranking:
//! - **Recency**: Files touched recently are likely more relevant (exponential decay)
//! - **Churn**: High commit frequency indicates volatility or active development (logarithmic)
//! - **Lifecycle phases**: Crystal (stable), Rotting (debt), Emergent (new), Evolving (normal)
//! - **Temporal coupling**: Files that change together often have hidden dependencies
//!
//! ## Design Philosophy
//!
//! Git history reveals the "temperature" of code - where attention flows, what's stable,
//! what's churning. This complements PageRank (structural importance) with temporal
//! patterns that only emerge over time.
//!
//! We use git log parsing instead of libgit2 for:
//! - **Speed**: Spawning git is faster than FFI overhead for large repos
//! - **Simplicity**: No dependency hell, works with any git version
//! - **Flexibility**: Easy to extend with custom git commands
//!
//! ## Weight Formulas
//!
//! **Recency boost** (exponential decay):
//! ```text
//! recency_boost = 1.0 + (MAX_BOOST - 1.0) * exp(-days / DECAY_DAYS)
//! ```
//! - Today: 10x boost
//! - 30 days ago: ~4.3x
//! - 60 days ago: ~1.6x
//! - Asymptotes to 1.0 (no boost) for old files
//!
//! **Churn boost** (logarithmic to dampen outliers):
//! ```text
//! excess = max(0, commit_count - THRESHOLD)
//! churn_boost = 1.0 + ln(1.0 + excess) * (MAX_BOOST - 1.0) / 5.0
//! ```
//! - 5 commits: 1.0x (baseline, at threshold)
//! - 10 commits: ~2.79x (ln(6) ≈ 1.79)
//! - 20 commits: ~3.97x (ln(16) ≈ 2.77)
//! - 50 commits: ~5.29x (near MAX_BOOST of 6.0)
//!
//! **Combined weight**:
//! ```text
//! weight = recency_boost.powf(recency_scale) * churn_boost.powf(churn_scale)
//! ```
//! Intent recipes adjust `recency_scale` and `churn_scale` for different use cases.

use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::process::Command;

use crate::types::{FilePhase, RankingConfig};

/// Git-based weight calculation and lifecycle analysis.
///
/// This calculator extracts temporal signals from git history:
/// - Recency: When was this file last touched?
/// - Churn: How frequently does it change?
/// - Age: How long has it been in the repo?
/// - Authors: Who has worked on it?
/// - Coupling: What files change together?
///
/// All operations are batched for performance - we parse git log once
/// for all files rather than invoking git per-file.
pub struct GitWeightCalculator {
    config: RankingConfig,
}

/// Per-file statistics extracted from git history.
///
/// These stats capture both temporal patterns (when) and collaboration
/// patterns (who) that inform file lifecycle classification.
#[derive(Debug, Clone)]
pub struct FileStats {
    /// Days since last modification (0 = modified today)
    pub last_modified_days: u32,
    /// Days since first appearance in repo
    pub first_seen_days: u32,
    /// Total number of commits touching this file
    pub commit_count: usize,
    /// Unique authors who have committed to this file
    pub authors: HashSet<String>,
}

impl GitWeightCalculator {
    /// Create a new calculator with the given configuration.
    pub fn new(config: RankingConfig) -> Self {
        Self { config }
    }

    /// Compute git-based weights for all files in one pass.
    ///
    /// Returns a map of `rel_fname -> weight` where weight combines:
    /// - Recency boost (exponential decay)
    /// - Churn boost (logarithmic)
    ///
    /// Files not found in git history get weight 1.0 (neutral).
    ///
    /// # Performance
    ///
    /// This batches all files into a single `git log` invocation to minimize
    /// process overhead. For large repos with thousands of files, this is
    /// 100-1000x faster than per-file git queries.
    pub fn compute_weights(
        &self,
        root: &Path,
        rel_fnames: &[String],
    ) -> Result<HashMap<String, f64>> {
        let stats = self
            .get_file_stats(root, rel_fnames)
            .context("Failed to get git file stats")?;

        let mut weights = HashMap::with_capacity(rel_fnames.len());

        for fname in rel_fnames {
            let weight = if let Some(file_stats) = stats.get(fname) {
                self.calculate_weight(file_stats)
            } else {
                // File not in git history - neutral weight
                1.0
            };
            weights.insert(fname.clone(), weight);
        }

        Ok(weights)
    }

    /// Extract file statistics from git history.
    ///
    /// Runs a single batched `git log` command that extracts:
    /// - Commit timestamps (for recency and age)
    /// - Author emails (for collaboration patterns)
    /// - File paths (for associating commits with files)
    ///
    /// # Git Command
    ///
    /// ```bash
    /// git log --format=%aI|%ae --name-only --all -n 500 -- [files...]
    /// ```
    ///
    /// Output format:
    /// ```text
    /// 2024-01-15T10:30:00+00:00|author@example.com
    /// src/lib.rs
    /// src/main.rs
    ///
    /// 2024-01-14T09:00:00+00:00|other@example.com
    /// src/lib.rs
    /// ```
    ///
    /// We parse this incrementally to build per-file stats.
    ///
    /// # Limits
    ///
    /// Capped at 500 commits to keep latency low. For most use cases,
    /// recent history (last 500 commits) captures the relevant temporal
    /// signal. For repos with >500 commits, this samples the most recent
    /// activity which is what we care about for ranking.
    pub fn get_file_stats(
        &self,
        root: &Path,
        rel_fnames: &[String],
    ) -> Result<HashMap<String, FileStats>> {
        if rel_fnames.is_empty() {
            return Ok(HashMap::new());
        }

        // Spawn git log with batched file arguments
        let output = Command::new("git")
            .arg("log")
            .arg("--format=%aI|%ae") // ISO timestamp | author email
            .arg("--name-only") // Show modified files after each commit
            .arg("--all") // All branches (not just HEAD)
            .arg("-n")
            .arg("500") // Limit to recent 500 commits for performance
            .arg("--") // Separator before file paths
            .args(rel_fnames)
            .current_dir(root)
            .output()
            .context("Failed to execute git log")?;

        if !output.status.success() {
            // Not a git repo or git not installed - return empty stats
            // This allows ripmap to work gracefully in non-git directories
            return Ok(HashMap::new());
        }

        let log_text = String::from_utf8_lossy(&output.stdout);
        self.parse_git_log(&log_text, rel_fnames)
    }

    /// Parse git log output into per-file statistics.
    ///
    /// The log format alternates between:
    /// 1. Commit metadata: `timestamp|author`
    /// 2. Modified files: one per line
    /// 3. Empty line (commit separator)
    ///
    /// We track:
    /// - First/last commit timestamps per file
    /// - Commit count per file
    /// - Unique authors per file
    fn parse_git_log(
        &self,
        log_text: &str,
        rel_fnames: &[String],
    ) -> Result<HashMap<String, FileStats>> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .context("System time is before UNIX epoch")?
            .as_secs() as i64;

        // Build a set for fast file lookup
        let fname_set: HashSet<&str> = rel_fnames.iter().map(|s| s.as_str()).collect();

        // Accumulators for per-file data
        let mut first_seen: HashMap<String, i64> = HashMap::new();
        let mut last_modified: HashMap<String, i64> = HashMap::new();
        let mut commit_counts: HashMap<String, usize> = HashMap::new();
        let mut authors_map: HashMap<String, HashSet<String>> = HashMap::new();

        let mut current_timestamp: Option<i64> = None;
        let mut current_author: Option<String> = None;

        for line in log_text.lines() {
            let line = line.trim();

            if line.is_empty() {
                // Commit separator - reset state
                current_timestamp = None;
                current_author = None;
                continue;
            }

            if let Some(pipe_idx) = line.find('|') {
                // Commit header: timestamp|author
                let (timestamp_str, author_str) = line.split_at(pipe_idx);
                let author_str = &author_str[1..]; // Skip the '|'

                // Parse ISO 8601 timestamp to Unix epoch
                if let Ok(timestamp) = parse_iso8601(timestamp_str) {
                    current_timestamp = Some(timestamp);
                    current_author = Some(author_str.to_string());
                }
            } else if let Some(ts) = current_timestamp {
                // This is a file path - check if it's in our target set
                if fname_set.contains(line) {
                    let fname = line.to_string();

                    // Update first seen (minimum timestamp)
                    first_seen
                        .entry(fname.clone())
                        .and_modify(|t| *t = (*t).min(ts))
                        .or_insert(ts);

                    // Update last modified (maximum timestamp)
                    last_modified
                        .entry(fname.clone())
                        .and_modify(|t| *t = (*t).max(ts))
                        .or_insert(ts);

                    // Increment commit count
                    *commit_counts.entry(fname.clone()).or_insert(0) += 1;

                    // Track unique authors
                    if let Some(ref author) = current_author {
                        authors_map
                            .entry(fname.clone())
                            .or_insert_with(HashSet::new)
                            .insert(author.clone());
                    }
                }
            }
        }

        // Convert raw data into FileStats
        let mut stats = HashMap::new();
        for fname in fname_set {
            if let Some(&last_ts) = last_modified.get(fname) {
                let last_modified_days = days_since(now, last_ts);
                let first_ts = first_seen.get(fname).copied().unwrap_or(last_ts);
                let first_seen_days = days_since(now, first_ts);
                let commit_count = commit_counts.get(fname).copied().unwrap_or(0);
                let authors = authors_map.get(fname).cloned().unwrap_or_default();

                stats.insert(
                    fname.to_string(),
                    FileStats {
                        last_modified_days,
                        first_seen_days,
                        commit_count,
                        authors,
                    },
                );
            }
        }

        Ok(stats)
    }

    /// Calculate combined weight from recency and churn.
    ///
    /// This is the core weighting formula that combines two signals:
    /// 1. **Recency**: Exponential decay favoring recently-touched files
    /// 2. **Churn**: Logarithmic boost for frequently-changed files
    ///
    /// The intent recipe controls how much each signal contributes via
    /// exponentiation (allowing non-linear scaling).
    fn calculate_weight(&self, stats: &FileStats) -> f64 {
        let recency_boost = self.recency_boost(stats.last_modified_days);
        let churn_boost = self.churn_boost(stats.commit_count);

        // Combine boosts multiplicatively
        // Intent recipes can scale these via exponentiation in the caller
        recency_boost * churn_boost
    }

    /// Recency boost: exponential decay from last modification.
    ///
    /// Formula: `1.0 + (MAX - 1.0) * exp(-days / DECAY)`
    ///
    /// This gives:
    /// - MAX boost for files touched today
    /// - ~50% of MAX at DECAY_DAYS
    /// - Asymptotic approach to 1.0 (no boost) for old files
    fn recency_boost(&self, days: u32) -> f64 {
        let max_boost = self.config.git_recency_max_boost;
        let decay_days = self.config.git_recency_decay_days;

        1.0 + (max_boost - 1.0) * (-f64::from(days) / decay_days).exp()
    }

    /// Churn boost: logarithmic increase with commit count.
    ///
    /// Formula: `1.0 + ln(1 + excess) * (MAX - 1.0) / 5.0`
    /// where `excess = max(0, commits - THRESHOLD)`
    ///
    /// Logarithmic scaling prevents extreme outliers from dominating.
    /// A file with 1000 commits won't be 200x more important than one
    /// with 5 commits - the boost saturates gracefully.
    fn churn_boost(&self, commit_count: usize) -> f64 {
        let threshold = self.config.git_churn_threshold;
        let max_boost = self.config.git_churn_max_boost;

        let excess = commit_count.saturating_sub(threshold) as f64;
        1.0 + (1.0 + excess).ln() * (max_boost - 1.0) / 5.0
    }

    /// Classify file lifecycle phase based on age and activity.
    ///
    /// **Crystal**: Old and quiet - settled, safe to depend on
    /// - Age >= 180 days, quiet >= 30 days
    ///
    /// **Rotting**: Old but churning - tech debt surfacing
    /// - Age >= 90 days, touched in last 14 days, commits > median * 1.5
    ///
    /// **Emergent**: Brand new - still finding its shape
    /// - Age <= 30 days
    ///
    /// **Evolving**: Normal active development
    /// - Everything else
    ///
    /// This classification helps identify:
    /// - Safe foundation code (Crystal)
    /// - Potential refactoring targets (Rotting)
    /// - Unstable new code (Emergent)
    pub fn classify_phase(&self, stats: &FileStats) -> FilePhase {
        let age = stats.first_seen_days;
        let quiet = stats.last_modified_days;

        // Crystal: old and stable
        if age >= self.config.phase_crystal_min_age_days
            && quiet >= self.config.phase_crystal_min_quiet_days
        {
            return FilePhase::Crystal;
        }

        // Emergent: brand new
        if age <= self.config.phase_emergent_max_age_days {
            return FilePhase::Emergent;
        }

        // Rotting: old but recently churning
        // We'd ideally check if commit_count > median * 1.5, but computing
        // the median requires all stats. For now, use a simple heuristic:
        // old + recently active + high commit count
        if age >= self.config.phase_rotting_min_age_days
            && quiet <= self.config.phase_rotting_max_quiet_days
            && stats.commit_count >= self.config.git_badge_churn_commits
        {
            return FilePhase::Rotting;
        }

        // Default: evolving
        FilePhase::Evolving
    }

    /// Compute badges for files based on temporal signals.
    ///
    /// Badges are lightweight annotations that appear in grepmap output:
    /// - `[recent]`: Modified in last N days
    /// - `[high-churn]`: Commits >= threshold
    /// - `[crystal]`, `[rotting]`, `[emergent]`, `[evolving]`: Lifecycle phase
    ///
    /// Badges help users quickly identify temporal patterns at a glance.
    pub fn get_badges(&self, stats: &HashMap<String, FileStats>) -> HashMap<String, Vec<String>> {
        let mut badges = HashMap::new();

        for (fname, file_stats) in stats {
            let mut file_badges = Vec::new();

            // Recent badge
            if file_stats.last_modified_days <= self.config.git_badge_recent_days {
                file_badges.push("recent".to_string());
            }

            // High-churn badge
            if file_stats.commit_count >= self.config.git_badge_churn_commits {
                file_badges.push("high-churn".to_string());
            }

            // Lifecycle phase badge
            let phase = self.classify_phase(file_stats);
            file_badges.push(phase.badge().to_string());

            badges.insert(fname.clone(), file_badges);
        }

        badges
    }

    /// Compute temporal coupling: files that change together.
    ///
    /// Temporal coupling reveals hidden dependencies not visible in the code:
    /// - Config files that must stay in sync
    /// - Tests that track implementation files
    /// - Related modules that evolve together
    ///
    /// We use **Jaccard similarity** on commit sets:
    /// ```text
    /// coupling(A, B) = |commits_A ∩ commits_B| / |commits_A ∪ commits_B|
    /// ```
    ///
    /// Returns for each file a list of (other_file, score) pairs sorted by
    /// score descending. Only includes pairs with score >= 0.3 to reduce noise.
    ///
    /// # Implementation
    ///
    /// We re-parse git log to build commit sets per file, then compute
    /// pairwise Jaccard similarity. For N files, this is O(N^2) comparisons,
    /// but N is typically small (files in focus) and Jaccard is very fast
    /// (set intersection/union on small sets).
    pub fn compute_temporal_coupling(
        &self,
        root: &Path,
        rel_fnames: &[String],
    ) -> Result<HashMap<String, Vec<(String, f64)>>> {
        if rel_fnames.is_empty() {
            return Ok(HashMap::new());
        }

        // Run git log to get commit hashes per file
        let output = Command::new("git")
            .arg("log")
            .arg("--format=%H") // Commit hash only
            .arg("--name-only")
            .arg("--all")
            .arg("-n")
            .arg("500")
            .arg("--")
            .args(rel_fnames)
            .current_dir(root)
            .output()
            .context("Failed to execute git log for temporal coupling")?;

        if !output.status.success() {
            return Ok(HashMap::new());
        }

        let log_text = String::from_utf8_lossy(&output.stdout);

        // Parse into file -> set of commit hashes
        let mut file_commits: HashMap<String, HashSet<String>> = HashMap::new();
        let fname_set: HashSet<&str> = rel_fnames.iter().map(|s| s.as_str()).collect();

        let mut current_commit: Option<String> = None;

        for line in log_text.lines() {
            let line = line.trim();

            if line.is_empty() {
                current_commit = None;
                continue;
            }

            // If line is 40 hex chars, it's a commit hash
            if line.len() == 40 && line.chars().all(|c| c.is_ascii_hexdigit()) {
                current_commit = Some(line.to_string());
            } else if let Some(ref commit) = current_commit {
                // This is a file path
                if fname_set.contains(line) {
                    file_commits
                        .entry(line.to_string())
                        .or_insert_with(HashSet::new)
                        .insert(commit.clone());
                }
            }
        }

        // Compute pairwise Jaccard similarity
        let mut coupling: HashMap<String, Vec<(String, f64)>> = HashMap::new();

        for (file_a, commits_a) in &file_commits {
            let mut pairs = Vec::new();

            for (file_b, commits_b) in &file_commits {
                if file_a == file_b {
                    continue;
                }

                let intersection = commits_a.intersection(commits_b).count();
                let union = commits_a.union(commits_b).count();

                if union > 0 {
                    let score = intersection as f64 / union as f64;

                    // Filter low scores to reduce noise
                    if score >= 0.3 {
                        pairs.push((file_b.clone(), score));
                    }
                }
            }

            // Sort by score descending
            pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            if !pairs.is_empty() {
                coupling.insert(file_a.clone(), pairs);
            }
        }

        Ok(coupling)
    }
}

/// Parse ISO 8601 timestamp to Unix epoch seconds.
///
/// Handles the format output by `git log --format=%aI`:
/// `2024-01-15T10:30:00+00:00`
///
/// We do a simple manual parse instead of pulling in a datetime crate
/// to keep dependencies lean. This works for git's consistent format.
fn parse_iso8601(s: &str) -> Result<i64> {
    // Format: YYYY-MM-DDTHH:MM:SS+HH:MM
    // Split on 'T'
    let parts: Vec<&str> = s.split('T').collect();
    if parts.len() != 2 {
        anyhow::bail!("Invalid ISO 8601 format");
    }

    let date = parts[0];
    let time_tz = parts[1];

    // Parse date: YYYY-MM-DD
    let date_parts: Vec<&str> = date.split('-').collect();
    if date_parts.len() != 3 {
        anyhow::bail!("Invalid date format");
    }
    let year: i32 = date_parts[0].parse()?;
    let month: i32 = date_parts[1].parse()?;
    let day: i32 = date_parts[2].parse()?;

    // Parse time: HH:MM:SS+TZ or HH:MM:SS-TZ
    let tz_split_pos = time_tz.find('+').or_else(|| time_tz.find('-'));
    let (time, _tz) = if let Some(pos) = tz_split_pos {
        (&time_tz[..pos], &time_tz[pos..])
    } else {
        (time_tz, "")
    };

    let time_parts: Vec<&str> = time.split(':').collect();
    if time_parts.len() != 3 {
        anyhow::bail!("Invalid time format");
    }
    let hour: i32 = time_parts[0].parse()?;
    let minute: i32 = time_parts[1].parse()?;
    let second: i32 = time_parts[2].parse()?;

    // Simplified Unix epoch calculation (ignoring leap years, timezones)
    // This is approximate but sufficient for day-level granularity
    let days_since_epoch = (year - 1970) * 365 + (year - 1969) / 4 // Leap years
        + days_in_months(month - 1)
        + day
        - 1;

    let seconds = i64::from(days_since_epoch) * 86400
        + i64::from(hour) * 3600
        + i64::from(minute) * 60
        + i64::from(second);

    Ok(seconds)
}

/// Days in each month (non-leap year).
fn days_in_months(months: i32) -> i32 {
    const DAYS: [i32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    DAYS.iter().take(months as usize).sum()
}

/// Compute days between two Unix timestamps.
fn days_since(now_secs: i64, then_secs: i64) -> u32 {
    let diff = now_secs.saturating_sub(then_secs);
    (diff / 86400) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recency_boost() {
        let config = RankingConfig::default();
        let calc = GitWeightCalculator::new(config.clone());

        // Today: should be close to MAX_BOOST
        let boost_today = calc.recency_boost(0);
        assert!(boost_today > config.git_recency_max_boost * 0.99);

        // 30 days: should be around 4-5x
        let boost_30d = calc.recency_boost(30);
        assert!(boost_30d > 3.0 && boost_30d < 6.0);

        // 180 days: should be close to 1.0
        let boost_180d = calc.recency_boost(180);
        assert!(boost_180d < 2.0);
    }

    #[test]
    fn test_churn_boost() {
        let config = RankingConfig::default();
        let calc = GitWeightCalculator::new(config.clone());

        // At threshold: should be 1.0
        let boost_threshold = calc.churn_boost(config.git_churn_threshold);
        assert!((boost_threshold - 1.0).abs() < 0.01);

        // 10 commits: excess=5, ln(6)≈1.79, boost≈2.79
        let boost_10 = calc.churn_boost(10);
        assert!(boost_10 > 2.5 && boost_10 < 3.0);

        // 50 commits: should approach MAX_BOOST
        let boost_50 = calc.churn_boost(50);
        assert!(boost_50 > 4.0 && boost_50 < config.git_churn_max_boost);
    }

    #[test]
    fn test_classify_phase() {
        let config = RankingConfig::default();
        let calc = GitWeightCalculator::new(config);

        // Crystal: old and stable
        let crystal = FileStats {
            last_modified_days: 60,
            first_seen_days: 200,
            commit_count: 5,
            authors: HashSet::new(),
        };
        assert_eq!(calc.classify_phase(&crystal), FilePhase::Crystal);

        // Emergent: brand new
        let emergent = FileStats {
            last_modified_days: 1,
            first_seen_days: 10,
            commit_count: 3,
            authors: HashSet::new(),
        };
        assert_eq!(calc.classify_phase(&emergent), FilePhase::Emergent);

        // Rotting: old but churning
        let rotting = FileStats {
            last_modified_days: 2,
            first_seen_days: 100,
            commit_count: 15,
            authors: HashSet::new(),
        };
        assert_eq!(calc.classify_phase(&rotting), FilePhase::Rotting);

        // Evolving: normal
        let evolving = FileStats {
            last_modified_days: 20,
            first_seen_days: 60,
            commit_count: 7,
            authors: HashSet::new(),
        };
        assert_eq!(calc.classify_phase(&evolving), FilePhase::Evolving);
    }

    #[test]
    fn test_parse_iso8601() {
        let timestamp = "2024-01-15T10:30:00+00:00";
        let result = parse_iso8601(timestamp);
        assert!(result.is_ok());

        // Approximate check - should be in 2024 range
        let secs = result.unwrap();
        assert!(secs > 1_700_000_000); // After 2023
        assert!(secs < 1_800_000_000); // Before 2027
    }

    #[test]
    fn test_days_since() {
        let now = 1_700_000_000;
        let then = now - 86400 * 7; // 7 days ago
        assert_eq!(days_since(now, then), 7);

        let same = days_since(now, now);
        assert_eq!(same, 0);
    }

    #[test]
    fn test_get_badges() {
        let config = RankingConfig::default();
        let calc = GitWeightCalculator::new(config);

        let mut stats = HashMap::new();

        // Recent and high-churn file
        stats.insert(
            "hot.rs".to_string(),
            FileStats {
                last_modified_days: 2,
                first_seen_days: 100,
                commit_count: 15,
                authors: HashSet::new(),
            },
        );

        // Old stable file
        stats.insert(
            "stable.rs".to_string(),
            FileStats {
                last_modified_days: 60,
                first_seen_days: 200,
                commit_count: 5,
                authors: HashSet::new(),
            },
        );

        let badges = calc.get_badges(&stats);

        // hot.rs should have [recent], [high-churn], [rotting]
        let hot_badges = &badges["hot.rs"];
        assert!(hot_badges.contains(&"recent".to_string()));
        assert!(hot_badges.contains(&"high-churn".to_string()));
        assert!(hot_badges.contains(&"rotting".to_string()));

        // stable.rs should have [crystal]
        let stable_badges = &badges["stable.rs"];
        assert!(stable_badges.contains(&"crystal".to_string()));
    }

    #[test]
    fn test_parse_git_log() {
        let config = RankingConfig::default();
        let calc = GitWeightCalculator::new(config);

        let log = r#"2024-01-15T10:30:00+00:00|alice@example.com
src/lib.rs
src/main.rs

2024-01-14T09:00:00+00:00|bob@example.com
src/lib.rs

2024-01-10T15:20:00+00:00|alice@example.com
src/main.rs
"#;

        let files = vec!["src/lib.rs".to_string(), "src/main.rs".to_string()];
        let stats = calc.parse_git_log(log, &files).unwrap();

        // src/lib.rs: 2 commits, 2 authors
        let lib_stats = &stats["src/lib.rs"];
        assert_eq!(lib_stats.commit_count, 2);
        assert_eq!(lib_stats.authors.len(), 2);

        // src/main.rs: 2 commits, 1 author
        let main_stats = &stats["src/main.rs"];
        assert_eq!(main_stats.commit_count, 2);
        assert_eq!(main_stats.authors.len(), 1);
    }
}
