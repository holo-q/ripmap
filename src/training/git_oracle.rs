//! Git history as ground truth oracle for ranking evaluation.
//!
//! ## The Retrocausal Principle
//!
//! Every git commit encodes a developer's cognitive state: "these files were
//! connected in my mind when I made this change." We exploit this as training
//! signal - given one file from a commit as "focus", the other files SHOULD
//! rank highly in our output.
//!
//! ## Commit Quality Hierarchy
//!
//! Not all commits provide equal signal quality:
//!
//! | Commit Type       | Signal Quality | Reasoning                              |
//! |-------------------|----------------|----------------------------------------|
//! | Bugfix (2-6)      | GOLD (1.5x)    | Causal: these files caused/fixed issue |
//! | Feature (3-8)     | Strong (1.2x)  | Semantic: implementation unit          |
//! | Test+Impl         | Moderate (1.0x)| Mechanical but validates coupling      |
//! | Refactor (10+)    | Weak (0.4x)    | Often mechanical rename/move           |
//! | WIP/checkpoint    | Skip (0.0x)    | Incomplete thought, noise              |
//! | Formatting        | Skip (0.0x)    | No semantic content                    |
//!
//! ## Coupling Strength
//!
//! Files that change together frequently have higher coupling weight.
//! We use a Jaccard-like metric:
//!
//! ```text
//! coupling(A,B) = cochange_count(A,B) / (changes(A) + changes(B) - cochange_count(A,B))
//! ```
//!
//! This normalizes for file activity - a pair that changes together 5/10 times
//! is more coupled than a pair that changes together 5/100 times.
//!
//! ## Session Clustering
//!
//! Commits within a short time window (default 30min) are likely part of the
//! same cognitive task. We can cluster these into "sessions" for stronger
//! signal - files touched across commits in a session are still related.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::types::Intent;

/// A single training case extracted from git history.
///
/// Each case represents: "given `seed_file` as focus, `expected_related`
/// files should rank highly because they were changed together."
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitCase {
    /// The commit this case was derived from
    pub commit_sha: String,

    /// Commit timestamp (Unix epoch seconds)
    pub timestamp: i64,

    /// The file we treat as "focus" / query seed
    pub seed_file: String,

    /// Other files from the commit that should rank high
    pub expected_related: Vec<String>,

    /// Original commit message
    pub message: String,

    /// Inferred intent from commit message parsing
    pub inferred_intent: Intent,

    /// Quality weight for this commit (0.0-2.0)
    /// Based on commit type, file count, message clarity
    pub quality_weight: f64,

    /// Number of files in original commit
    pub commit_file_count: usize,
}

/// A case with coupling-weighted expected files.
///
/// Instead of binary "related or not", each expected file has a weight
/// based on historical co-change frequency. Files that ALWAYS change
/// with the seed have weight ~1.0, occasional co-changes have lower weight.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedCase {
    pub seed_file: String,
    pub commit_sha: String,
    pub inferred_intent: Intent,
    /// (file, coupling_weight) - higher = more historically coupled
    pub expected_related: Vec<(String, f64)>,
    /// Overall case quality (commit quality × coupling strength)
    pub case_weight: f64,
}

/// Raw commit data from git log
#[derive(Debug, Clone)]
struct RawCommit {
    sha: String,
    timestamp: i64,
    message: String,
    files: Vec<String>,
}

/// Extract training cases from a git repository.
///
/// # Algorithm
///
/// 1. Walk git log in reverse chronological order
/// 2. Filter commits by quality (skip WIP, formatting, huge refactors)
/// 3. For each qualifying commit, generate N cases (one per file as seed)
/// 4. Weight cases by commit quality signals
///
/// # Arguments
///
/// * `repo` - Path to git repository root
/// * `max_commits` - Maximum commits to process (for speed)
/// * `min_files` - Minimum files per commit (default 2)
/// * `max_files` - Maximum files per commit (default 15)
///
/// # Returns
///
/// Vector of GitCase, each representing one (seed, expected) pair
pub fn extract_cases(
    repo: &Path,
    max_commits: usize,
    min_files: usize,
    max_files: usize,
) -> Vec<GitCase> {
    let commits = parse_git_log(repo, max_commits);
    let mut cases = Vec::new();

    for commit in commits {
        // Filter by file count
        if commit.files.len() < min_files || commit.files.len() > max_files {
            continue;
        }

        // Compute commit quality weight
        let quality = compute_commit_quality(&commit);

        // Skip low-quality commits entirely
        if quality < 0.1 {
            continue;
        }

        let intent = parse_intent(&commit.message);

        // Generate one case per file (each takes turn as seed)
        for (i, seed) in commit.files.iter().enumerate() {
            // Skip non-source files as seeds
            if !is_source_file(seed) {
                continue;
            }

            let expected: Vec<_> = commit.files
                .iter()
                .enumerate()
                .filter(|(j, f)| *j != i && is_source_file(f))
                .map(|(_, f)| f.clone())
                .collect();

            // Need at least one expected file
            if expected.is_empty() {
                continue;
            }

            cases.push(GitCase {
                commit_sha: commit.sha.clone(),
                timestamp: commit.timestamp,
                seed_file: seed.clone(),
                expected_related: expected,
                message: commit.message.clone(),
                inferred_intent: intent,
                quality_weight: quality,
                commit_file_count: commit.files.len(),
            });
        }
    }

    cases
}

/// Compute file coupling strengths from co-change history.
///
/// Uses Jaccard-like metric normalized by file activity:
/// ```text
/// coupling(A,B) = cochange(A,B) / (count(A) + count(B) - cochange(A,B))
/// ```
///
/// # Arguments
///
/// * `repo` - Path to git repository
/// * `max_commits` - How far back to look
///
/// # Returns
///
/// HashMap from (file_a, file_b) -> coupling strength (0.0-1.0)
/// Keys are normalized so file_a < file_b lexicographically.
pub fn compute_coupling_weights(
    repo: &Path,
    max_commits: usize,
) -> HashMap<(String, String), f64> {
    let commits = parse_git_log(repo, max_commits);

    let mut cochange_counts: HashMap<(String, String), usize> = HashMap::new();
    let mut file_counts: HashMap<String, usize> = HashMap::new();

    for commit in commits {
        // Skip huge commits (refactors add noise to coupling)
        if commit.files.len() > 20 {
            continue;
        }

        let source_files: Vec<_> = commit.files
            .iter()
            .filter(|f| is_source_file(f))
            .cloned()
            .collect();

        // Count individual file appearances
        for f in &source_files {
            *file_counts.entry(f.clone()).or_default() += 1;
        }

        // Count co-changes (all pairs)
        for i in 0..source_files.len() {
            for j in (i + 1)..source_files.len() {
                let pair = normalize_pair(&source_files[i], &source_files[j]);
                *cochange_counts.entry(pair).or_default() += 1;
            }
        }
    }

    // Convert to Jaccard-like coupling strength
    cochange_counts
        .into_iter()
        .filter_map(|((a, b), co)| {
            let ca = file_counts.get(&a)?;
            let cb = file_counts.get(&b)?;
            let union = ca + cb - co;
            if union == 0 {
                return None;
            }
            let jaccard = co as f64 / union as f64;
            Some(((a, b), jaccard))
        })
        .collect()
}

/// Enhance cases with coupling weights.
///
/// Transforms GitCase (binary related) into WeightedCase (graded relevance)
/// by looking up historical coupling strength for each expected file.
pub fn weight_cases(
    cases: Vec<GitCase>,
    coupling: &HashMap<(String, String), f64>,
) -> Vec<WeightedCase> {
    cases
        .into_iter()
        .map(|case| {
            let weighted_expected: Vec<_> = case.expected_related
                .iter()
                .map(|f| {
                    let pair = normalize_pair(&case.seed_file, f);
                    let weight = coupling.get(&pair).copied().unwrap_or(0.1);
                    (f.clone(), weight)
                })
                .collect();

            // Case weight = commit quality × average coupling
            let avg_coupling: f64 = if weighted_expected.is_empty() {
                0.0
            } else {
                weighted_expected.iter().map(|(_, w)| w).sum::<f64>()
                    / weighted_expected.len() as f64
            };

            WeightedCase {
                seed_file: case.seed_file,
                commit_sha: case.commit_sha,
                inferred_intent: case.inferred_intent,
                expected_related: weighted_expected,
                case_weight: case.quality_weight * (0.5 + avg_coupling),
            }
        })
        .collect()
}

/// Cluster commits into sessions by time proximity.
///
/// Commits within `session_gap` seconds of each other are grouped.
/// This captures multi-commit tasks where files across commits are related.
pub fn cluster_into_sessions(
    cases: &[GitCase],
    session_gap_secs: i64,
) -> Vec<Vec<&GitCase>> {
    if cases.is_empty() {
        return vec![];
    }

    // Sort by timestamp
    let mut sorted: Vec<_> = cases.iter().collect();
    sorted.sort_by_key(|c| c.timestamp);

    let mut sessions = Vec::new();
    let mut current_session = vec![sorted[0]];

    for case in sorted.into_iter().skip(1) {
        let last_ts = current_session.last().unwrap().timestamp;
        if case.timestamp - last_ts <= session_gap_secs {
            current_session.push(case);
        } else {
            if !current_session.is_empty() {
                sessions.push(current_session);
            }
            current_session = vec![case];
        }
    }

    if !current_session.is_empty() {
        sessions.push(current_session);
    }

    sessions
}

// === Private implementation ===

/// Parse git log output into structured commits.
fn parse_git_log(repo: &Path, max_commits: usize) -> Vec<RawCommit> {
    // Format: SHA|timestamp|message
    // Followed by list of files
    let output = Command::new("git")
        .current_dir(repo)
        .args([
            "log",
            &format!("-{}", max_commits),
            "--pretty=format:%H|%ct|%s",
            "--name-only",
        ])
        .output();

    let output = match output {
        Ok(o) => o,
        Err(e) => {
            eprintln!("Failed to run git log: {}", e);
            return vec![];
        }
    };

    if !output.status.success() {
        eprintln!("git log failed: {}", String::from_utf8_lossy(&output.stderr));
        return vec![];
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut commits = Vec::new();
    let mut current_commit: Option<RawCommit> = None;

    for line in stdout.lines() {
        if line.is_empty() {
            continue;
        }

        // Check if this is a commit header line (SHA|timestamp|message)
        if line.contains('|') && line.len() >= 40 {
            let parts: Vec<_> = line.splitn(3, '|').collect();
            if parts.len() == 3 && parts[0].len() == 40 {
                // Save previous commit
                if let Some(commit) = current_commit.take() {
                    if !commit.files.is_empty() {
                        commits.push(commit);
                    }
                }

                // Parse new commit
                let sha = parts[0].to_string();
                let timestamp = parts[1].parse().unwrap_or(0);
                let message = parts[2].to_string();

                current_commit = Some(RawCommit {
                    sha,
                    timestamp,
                    message,
                    files: Vec::new(),
                });
                continue;
            }
        }

        // Otherwise it's a file path
        if let Some(ref mut commit) = current_commit {
            commit.files.push(line.to_string());
        }
    }

    // Don't forget the last commit
    if let Some(commit) = current_commit {
        if !commit.files.is_empty() {
            commits.push(commit);
        }
    }

    commits
}

/// Compute quality weight for a commit (0.0 - 2.0).
///
/// High quality signals:
/// - Bugfix commit message (1.5x)
/// - Feature/implement message (1.2x)
/// - Cross-directory changes (1.2x)
/// - Moderate file count (2-8 optimal)
///
/// Low quality signals:
/// - WIP/checkpoint message (0.2x)
/// - Formatting/lint commits (0.0x)
/// - Huge refactors (0.4x)
/// - Merge commits (0.3x)
fn compute_commit_quality(commit: &RawCommit) -> f64 {
    let mut weight: f64 = 1.0;
    let msg = commit.message.to_lowercase();

    // === Message-based signals ===

    // Bugfixes are gold - clear causal relationship
    if msg.starts_with("fix")
        || msg.contains("bugfix")
        || msg.contains("hotfix")
        || msg.starts_with("bug:")
    {
        weight *= 1.5;
    }
    // Features are strong signal
    else if msg.starts_with("feat")
        || msg.contains("implement")
        || msg.starts_with("add ")
        || msg.starts_with("add:")
    {
        weight *= 1.2;
    }
    // Refactors are weaker (often mechanical)
    else if msg.starts_with("refactor")
        || msg.contains("rename")
        || msg.contains("move ")
    {
        weight *= 0.6;
    }

    // WIP/checkpoint = incomplete thought = noise
    if msg.contains("wip")
        || msg.contains("work in progress")
        || msg.contains("save")
        || msg.contains("checkpoint")
        || msg.contains("tmp")
    {
        weight *= 0.2;
    }

    // Formatting/lint = no semantic content
    if msg.contains("format")
        || msg.contains("lint")
        || msg.contains("prettier")
        || msg.contains("style:")
        || msg.contains("chore: format")
    {
        return 0.0; // Skip entirely
    }

    // Merge commits are often noisy
    if msg.starts_with("merge") {
        weight *= 0.3;
    }

    // === File count signals ===

    let n = commit.files.len();
    if n <= 1 {
        return 0.0; // No relational signal
    } else if n <= 6 {
        weight *= 1.2; // Sweet spot
    } else if n <= 10 {
        weight *= 1.0; // Normal
    } else if n <= 15 {
        weight *= 0.7; // Getting noisy
    } else {
        weight *= 0.4; // Probably a refactor
    }

    // === Directory diversity ===
    // Files in different directories = meaningful cross-cutting change
    let unique_dirs: HashSet<_> = commit.files
        .iter()
        .filter_map(|f| Path::new(f).parent())
        .filter_map(|p| p.to_str())
        .collect();

    if unique_dirs.len() >= 3 {
        weight *= 1.2; // Cross-cutting changes are meaningful
    }

    // === Source vs config ratio ===
    // Pure test additions or pure config changes are less interesting
    let source_count = commit.files.iter().filter(|f| is_source_file(f)).count();
    let source_ratio = source_count as f64 / n as f64;

    if source_ratio < 0.3 {
        weight *= 0.5; // Mostly non-source
    }

    weight.min(2.0) // Cap at 2.0
}

/// Parse commit message to infer developer intent.
fn parse_intent(message: &str) -> Intent {
    let msg = message.to_lowercase();

    if msg.contains("fix")
        || msg.contains("bug")
        || msg.contains("issue")
        || msg.contains("error")
        || msg.contains("crash")
        || msg.contains("debug")
    {
        Intent::Debug
    } else if msg.contains("refactor")
        || msg.contains("clean")
        || msg.contains("rename")
        || msg.contains("reorganize")
        || msg.contains("restructure")
    {
        Intent::Refactor
    } else if msg.contains("add")
        || msg.contains("implement")
        || msg.contains("feature")
        || msg.contains("new")
        || msg.contains("create")
        || msg.contains("support")
    {
        Intent::Extend
    } else {
        Intent::Explore
    }
}

/// Check if a file is a source code file (not config, docs, etc.)
fn is_source_file(path: &str) -> bool {
    let source_extensions = [
        // Rust
        ".rs",
        // Python
        ".py",
        // JavaScript/TypeScript
        ".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs",
        // Go
        ".go",
        // C/C++
        ".c", ".h", ".cpp", ".hpp", ".cc", ".hh",
        // Java/Kotlin
        ".java", ".kt", ".kts",
        // Ruby
        ".rb",
        // PHP
        ".php",
        // Swift
        ".swift",
        // C#
        ".cs",
        // Scala
        ".scala",
        // Elixir
        ".ex", ".exs",
        // Haskell
        ".hs",
        // OCaml
        ".ml", ".mli",
        // Zig
        ".zig",
    ];

    // Check extension
    let has_source_ext = source_extensions.iter().any(|ext| path.ends_with(ext));
    if !has_source_ext {
        return false;
    }

    // Exclude tests (optional - they do provide coupling signal)
    // For now, include tests as they validate coupling

    // Exclude vendor/deps
    let vendor_patterns = [
        "node_modules/",
        "vendor/",
        "third_party/",
        "__pycache__/",
        "target/",
        ".git/",
        "dist/",
        "build/",
    ];

    !vendor_patterns.iter().any(|p| path.contains(p))
}

/// Normalize a file pair for consistent map keys.
/// Returns (smaller, larger) lexicographically.
fn normalize_pair(a: &str, b: &str) -> (String, String) {
    if a < b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_source_file() {
        assert!(is_source_file("src/main.rs"));
        assert!(is_source_file("lib/parser.py"));
        assert!(is_source_file("components/Button.tsx"));
        assert!(!is_source_file("README.md"));
        assert!(!is_source_file("package.json"));
        assert!(!is_source_file("node_modules/lodash/index.js"));
    }

    #[test]
    fn test_normalize_pair() {
        assert_eq!(
            normalize_pair("b.rs", "a.rs"),
            ("a.rs".to_string(), "b.rs".to_string())
        );
        assert_eq!(
            normalize_pair("a.rs", "b.rs"),
            ("a.rs".to_string(), "b.rs".to_string())
        );
    }

    #[test]
    fn test_parse_intent() {
        assert_eq!(parse_intent("fix: null pointer in parser"), Intent::Debug);
        assert_eq!(parse_intent("refactor: clean up auth module"), Intent::Refactor);
        assert_eq!(parse_intent("add: new user registration"), Intent::Extend);
        assert_eq!(parse_intent("update docs"), Intent::Explore);
    }

    #[test]
    fn test_commit_quality_bugfix() {
        let commit = RawCommit {
            sha: "abc123".repeat(7),
            timestamp: 0,
            message: "fix: crash on empty input".to_string(),
            files: vec!["src/parser.rs".to_string(), "src/input.rs".to_string()],
        };
        let quality = compute_commit_quality(&commit);
        assert!(quality > 1.0, "Bugfix should have high quality: {}", quality);
    }

    #[test]
    fn test_commit_quality_wip() {
        let commit = RawCommit {
            sha: "abc123".repeat(7),
            timestamp: 0,
            message: "WIP: still working on this".to_string(),
            files: vec!["src/a.rs".to_string(), "src/b.rs".to_string()],
        };
        let quality = compute_commit_quality(&commit);
        assert!(quality < 0.5, "WIP should have low quality: {}", quality);
    }

    #[test]
    fn test_commit_quality_formatting() {
        let commit = RawCommit {
            sha: "abc123".repeat(7),
            timestamp: 0,
            message: "chore: format code with prettier".to_string(),
            files: vec!["src/a.rs".to_string(), "src/b.rs".to_string()],
        };
        let quality = compute_commit_quality(&commit);
        assert_eq!(quality, 0.0, "Formatting should be skipped");
    }

    #[test]
    fn test_commit_quality_large_refactor() {
        let commit = RawCommit {
            sha: "abc123".repeat(7),
            timestamp: 0,
            message: "refactor: rename foo to bar everywhere".to_string(),
            files: (0..20).map(|i| format!("src/file{}.rs", i)).collect(),
        };
        let quality = compute_commit_quality(&commit);
        assert!(quality < 0.5, "Large refactor should have low quality: {}", quality);
    }
}
