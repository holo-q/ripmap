//! Curated repository list for benchmark evaluation.
//!
//! ## Selection Criteria
//!
//! The ideal benchmark repository has:
//!
//! 1. **Atomic commit discipline** - Each commit is a logical unit, not a WIP dump
//! 2. **Moderate size** - 50-500 files, enough structure but not too slow
//! 3. **Active development** - Recent commits provide relevant patterns
//! 4. **Multi-file commits** - Average 2-6 files per commit (relational signal)
//! 5. **Diverse intents** - Mix of bugfix, feature, refactor commits
//! 6. **Clear architecture** - Well-structured code has meaningful PageRank
//!
//! ## Commit Rhythm Analysis
//!
//! Different projects have different commit patterns:
//!
//! | Pattern              | Signal Quality | Typical Projects           |
//! |----------------------|----------------|----------------------------|
//! | Squash-merge         | High           | Feature-branch workflows   |
//! | Atomic commits       | High           | Solo/small team projects   |
//! | Conventional commits | High           | OSS with contributors      |
//! | WIP commits          | Low            | Personal projects          |
//! | Mega-commits         | Low            | Infrequent committers      |
//!
//! ## Language Diversity
//!
//! Include repos from multiple languages to ensure the ranking
//! algorithm generalizes and isn't overfit to one ecosystem.

use serde::{Deserialize, Serialize};

/// Specification for a benchmark repository.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoSpec {
    /// GitHub URL (for cloning)
    pub url: &'static str,

    /// Short name for logging
    pub name: &'static str,

    /// Primary language
    pub language: Language,

    /// Why this repo was selected
    pub rationale: &'static str,

    /// Expected commit quality (0.0-1.0)
    pub expected_quality: f64,

    /// Approximate size category
    pub size: RepoSize,

    /// Is it actively maintained?
    pub active: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Language {
    Rust,
    Python,
    TypeScript,
    Go,
    Mixed,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RepoSize {
    /// <100 files
    Small,
    /// 100-300 files
    Medium,
    /// 300-600 files
    Large,
}

/// Curated list of repositories for benchmarking.
///
/// Selection rationale for each:
///
/// ## Rust Ecosystem
///
/// - **ripgrep**: BurntSushi's meticulous commit hygiene, excellent
///   documentation, clear module structure. Gold standard for Rust.
///
/// - **bat**: Well-structured file viewer, clean separation of concerns,
///   active development. Good PageRank signal from dependency structure.
///
/// - **just**: Task runner with very clean commits, small codebase makes
///   it good for fast iteration during benchmark development.
///
/// - **starship**: Prompt customizer with excellent modular architecture.
///   Many modules with clear dependencies = strong graph signal.
///
/// - **tokei**: Code counter, small but well-architected. Fast to process.
///
/// ## Python Ecosystem
///
/// - **httpx**: Async HTTP library, encode team's excellent code quality.
///   Clean commits, modern Python, good module structure.
///
/// - **rich**: Terminal rendering library, active development, very clear
///   commit messages. Will Mcgugan's disciplined style.
///
/// - **textual**: TUI framework, very active, excellent commit discipline.
///   Complex enough to have meaningful graph structure.
///
/// - **pydantic**: Data validation, extremely active, clear conventional
///   commit style. Good mix of features/fixes.
///
/// ## TypeScript Ecosystem
///
/// - **zod**: Validation library, clean functional style, good commits.
///   Small but high quality.
///
/// ## Go Ecosystem
///
/// - **bubbletea**: TUI library from Charm, excellent commit quality,
///   idiomatic Go with clean architecture.
///
/// - **lazygit**: Git UI, active development, good commit messages.
///   Practical codebase with real complexity.
pub const CURATED_REPOS: &[RepoSpec] = &[
    // === Rust ===
    RepoSpec {
        url: "https://github.com/BurntSushi/ripgrep",
        name: "ripgrep",
        language: Language::Rust,
        rationale: "Gold standard commit hygiene. BurntSushi's meticulous style with clear atomic commits, excellent documentation. Moderate size with rich dependency graph.",
        expected_quality: 0.95,
        size: RepoSize::Medium,
        active: true,
    },
    RepoSpec {
        url: "https://github.com/sharkdp/bat",
        name: "bat",
        language: Language::Rust,
        rationale: "Well-structured file viewer with clean module separation. Active development, good conventional commit style. Syntax highlighting system provides complex dep graph.",
        expected_quality: 0.85,
        size: RepoSize::Medium,
        active: true,
    },
    RepoSpec {
        url: "https://github.com/casey/just",
        name: "just",
        language: Language::Rust,
        rationale: "Task runner with very clean, focused commits. Small codebase ideal for fast iteration. Casey's disciplined style.",
        expected_quality: 0.90,
        size: RepoSize::Small,
        active: true,
    },
    RepoSpec {
        url: "https://github.com/starship/starship",
        name: "starship",
        language: Language::Rust,
        rationale: "Prompt customizer with highly modular architecture. Many independent modules with clear dependencies. Excellent for PageRank signal.",
        expected_quality: 0.80,
        size: RepoSize::Large,
        active: true,
    },
    RepoSpec {
        url: "https://github.com/XAMPPRocky/tokei",
        name: "tokei",
        language: Language::Rust,
        rationale: "Code counter, compact but well-architected. Fast to process, good for quick iterations. Clean language-focused modules.",
        expected_quality: 0.85,
        size: RepoSize::Small,
        active: true,
    },
    RepoSpec {
        url: "https://github.com/eza-community/eza",
        name: "eza",
        language: Language::Rust,
        rationale: "Modern ls replacement, fork of exa. Active community, good commit quality. File system logic provides interesting coupling patterns.",
        expected_quality: 0.80,
        size: RepoSize::Medium,
        active: true,
    },
    // === Python ===
    RepoSpec {
        url: "https://github.com/encode/httpx",
        name: "httpx",
        language: Language::Python,
        rationale: "Async HTTP library from encode team. Excellent code quality, clean commits, modern Python. Good module structure for graph analysis.",
        expected_quality: 0.90,
        size: RepoSize::Medium,
        active: true,
    },
    RepoSpec {
        url: "https://github.com/Textualize/rich",
        name: "rich",
        language: Language::Python,
        rationale: "Terminal rendering by Will McGugan. Very active, disciplined commit style, clear messages. Complex rendering system = rich graph.",
        expected_quality: 0.85,
        size: RepoSize::Medium,
        active: true,
    },
    RepoSpec {
        url: "https://github.com/Textualize/textual",
        name: "textual",
        language: Language::Python,
        rationale: "TUI framework, extremely active development. Excellent conventional commits. Widget system provides clear dependency graph.",
        expected_quality: 0.85,
        size: RepoSize::Large,
        active: true,
    },
    RepoSpec {
        url: "https://github.com/pydantic/pydantic",
        name: "pydantic",
        language: Language::Python,
        rationale: "Data validation library, massive adoption. Very active with clear conventional commits. Mix of core logic and integrations.",
        expected_quality: 0.80,
        size: RepoSize::Large,
        active: true,
    },
    RepoSpec {
        url: "https://github.com/astral-sh/ruff",
        name: "ruff",
        language: Language::Rust,
        rationale: "Python linter in Rust. Extremely active, excellent commit discipline. Rule system provides clear module structure. Cross-ecosystem expertise.",
        expected_quality: 0.90,
        size: RepoSize::Large,
        active: true,
    },
    // === TypeScript ===
    RepoSpec {
        url: "https://github.com/colinhacks/zod",
        name: "zod",
        language: Language::TypeScript,
        rationale: "Schema validation library. Clean functional style, focused codebase. Good commits, popular enough for real-world patterns.",
        expected_quality: 0.80,
        size: RepoSize::Small,
        active: true,
    },
    // === Go ===
    RepoSpec {
        url: "https://github.com/charmbracelet/bubbletea",
        name: "bubbletea",
        language: Language::Go,
        rationale: "TUI library from Charm. Excellent commit quality, idiomatic Go. Elm-style architecture provides clear message flow graph.",
        expected_quality: 0.90,
        size: RepoSize::Small,
        active: true,
    },
    RepoSpec {
        url: "https://github.com/charmbracelet/lipgloss",
        name: "lipgloss",
        language: Language::Go,
        rationale: "Style definitions for TUIs. Very clean, focused codebase. Charm team's disciplined style. Good for testing small repo behavior.",
        expected_quality: 0.90,
        size: RepoSize::Small,
        active: true,
    },
    RepoSpec {
        url: "https://github.com/jesseduffield/lazygit",
        name: "lazygit",
        language: Language::Go,
        rationale: "Git UI, practical codebase with real complexity. Active development, good commit messages. Tests real-world code patterns.",
        expected_quality: 0.75,
        size: RepoSize::Large,
        active: true,
    },
];

impl RepoSpec {
    /// Get clone command for this repo.
    pub fn clone_cmd(&self, dest: &str) -> String {
        format!("git clone --depth 1000 {} {}", self.url, dest)
    }

    /// Estimated number of commits to analyze (based on depth clone).
    pub fn estimated_commits(&self) -> usize {
        match self.size {
            RepoSize::Small => 500,
            RepoSize::Medium => 800,
            RepoSize::Large => 1000,
        }
    }
}

/// Filter repos by criteria.
pub fn filter_repos(
    language: Option<Language>,
    min_quality: Option<f64>,
    max_size: Option<RepoSize>,
) -> Vec<&'static RepoSpec> {
    CURATED_REPOS
        .iter()
        .filter(|r| {
            if let Some(lang) = language {
                if r.language != lang {
                    return false;
                }
            }
            if let Some(min_q) = min_quality {
                if r.expected_quality < min_q {
                    return false;
                }
            }
            if let Some(max_s) = max_size {
                let size_ord = |s: RepoSize| match s {
                    RepoSize::Small => 0,
                    RepoSize::Medium => 1,
                    RepoSize::Large => 2,
                };
                if size_ord(r.size) > size_ord(max_s) {
                    return false;
                }
            }
            true
        })
        .collect()
}

/// Get a quick subset for development/testing.
pub fn quick_repos() -> Vec<&'static RepoSpec> {
    filter_repos(None, Some(0.85), Some(RepoSize::Medium))
        .into_iter()
        .take(5)
        .collect()
}

/// Get all Rust repos (for Rust-specific analysis).
pub fn rust_repos() -> Vec<&'static RepoSpec> {
    filter_repos(Some(Language::Rust), None, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curated_repos_not_empty() {
        assert!(!CURATED_REPOS.is_empty());
    }

    #[test]
    fn test_all_repos_have_url() {
        for repo in CURATED_REPOS {
            assert!(repo.url.starts_with("https://github.com/"));
        }
    }

    #[test]
    fn test_filter_by_language() {
        let rust_repos = filter_repos(Some(Language::Rust), None, None);
        assert!(!rust_repos.is_empty());
        for repo in rust_repos {
            assert_eq!(repo.language, Language::Rust);
        }
    }

    #[test]
    fn test_filter_by_quality() {
        let high_quality = filter_repos(None, Some(0.90), None);
        for repo in high_quality {
            assert!(repo.expected_quality >= 0.90);
        }
    }

    #[test]
    fn test_quick_repos() {
        let quick = quick_repos();
        assert!(!quick.is_empty());
        assert!(quick.len() <= 5);
    }

    #[test]
    fn test_clone_cmd() {
        let repo = &CURATED_REPOS[0];
        let cmd = repo.clone_cmd("/tmp/test");
        assert!(cmd.contains("git clone"));
        assert!(cmd.contains("--depth"));
    }
}
