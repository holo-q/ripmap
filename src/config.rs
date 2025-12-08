//! Configuration loading from pyproject.toml and ripmap.toml.
//!
//! Follows conventions from ruff, black, mypy for familiarity:
//! - `[tool.ripmap]` section in pyproject.toml (preferred)
//! - Falls back to `[tool.ruff]` exclude patterns if no ripmap config
//! - Falls back to `[tool.pyright]` exclude patterns as last resort
//! - Standalone ripmap.toml as override
//!
//! ## Example
//!
//! ```toml
//! [tool.ripmap]
//! include = ["src/**", "lib/**"]
//! exclude = ["**/generated/**"]
//! extend-exclude = ["**/vendor/**"]
//! src = ["src", "lib"]
//! ```
//!
//! Or ripmap will inherit from existing tool configs:
//! ```toml
//! [tool.ruff]
//! exclude = ["migrations", "generated"]
//! ```

use owo_colors::OwoColorize;
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Default exclude patterns (common non-source directories).
pub const DEFAULT_EXCLUDES: &[&str] = &[
    "**/node_modules/**",
    "**/.git/**",
    "**/target/**",
    "**/build/**",
    "**/dist/**",
    "**/__pycache__/**",
    "**/.venv/**",
    "**/venv/**",
    "**/.tox/**",
    "**/.mypy_cache/**",
    "**/.pytest_cache/**",
    "**/.ruff_cache/**",
    "**/vendor/**",
    "**/third_party/**",
    "**/.next/**",
    "**/.nuxt/**",
];

/// Ripmap configuration.
#[derive(Debug, Clone, Default)]
pub struct Config {
    /// Source file for this config (for display), with tool name if inherited.
    pub source: Option<String>,

    /// Glob patterns for files to include. If empty, include all source files.
    pub include: Vec<String>,

    /// Glob patterns for files to exclude. Replaces defaults if set.
    pub exclude: Vec<String>,

    /// Additional exclude patterns (extends defaults).
    pub extend_exclude: Vec<String>,

    /// Source root directories (affects depth weighting).
    pub src: Vec<PathBuf>,
}

/// Raw config as deserialized from TOML (ripmap-native format).
#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
struct RawConfig {
    include: Option<Vec<String>>,
    exclude: Option<Vec<String>>,
    extend_exclude: Option<Vec<String>>,
    src: Option<Vec<String>>,
}

/// Ruff config structure (for fallback).
#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
struct RuffConfig {
    include: Option<Vec<String>>,
    exclude: Option<Vec<String>>,
    extend_exclude: Option<Vec<String>>,
    src: Option<Vec<String>>,
}

/// Ty (astral type checker) config structure.
/// Has nested `src` section with include/exclude.
#[derive(Debug, Deserialize, Default)]
struct TyConfig {
    src: Option<TySrcConfig>,
}

#[derive(Debug, Deserialize, Default)]
struct TySrcConfig {
    include: Option<Vec<String>>,
    exclude: Option<Vec<String>>,
}

/// Pyright config structure (for fallback).
#[derive(Debug, Deserialize, Default)]
struct PyrightConfig {
    include: Option<Vec<String>>,
    exclude: Option<Vec<String>>,
}

/// Wrapper for pyproject.toml structure.
#[derive(Debug, Deserialize, Default)]
struct PyProject {
    tool: Option<PyProjectTool>,
}

#[derive(Debug, Deserialize, Default)]
struct PyProjectTool {
    ripmap: Option<RawConfig>,
    ty: Option<TyConfig>,
    ruff: Option<RuffConfig>,
    pyright: Option<PyrightConfig>,
}

impl Config {
    /// Load configuration from the given directory.
    ///
    /// Search order:
    /// 1. ripmap.toml in directory
    /// 2. pyproject.toml [tool.ripmap] in directory
    /// 3. Walk up to find pyproject.toml (like ruff)
    /// 4. Default config if nothing found
    pub fn load(directory: &Path) -> Self {
        // Try ripmap.toml first
        let ripmap_toml = directory.join("ripmap.toml");
        if ripmap_toml.exists() {
            if let Some(config) = Self::load_ripmap_toml(&ripmap_toml) {
                return config;
            }
        }

        // Try pyproject.toml in current directory
        let pyproject = directory.join("pyproject.toml");
        if pyproject.exists() {
            if let Some(config) = Self::load_pyproject(&pyproject) {
                return config;
            }
        }

        // Walk up to find pyproject.toml
        let mut current = directory.to_path_buf();
        while let Some(parent) = current.parent() {
            let pyproject = parent.join("pyproject.toml");
            if pyproject.exists() {
                if let Some(config) = Self::load_pyproject(&pyproject) {
                    return config;
                }
            }
            current = parent.to_path_buf();
        }

        // Default config
        Self::default()
    }

    fn load_ripmap_toml(path: &Path) -> Option<Self> {
        let content = std::fs::read_to_string(path).ok()?;
        let raw: RawConfig = toml::from_str(&content).ok()?;
        Some(Self::from_raw(raw, path.to_path_buf()))
    }

    fn load_pyproject(path: &Path) -> Option<Self> {
        let content = std::fs::read_to_string(path).ok()?;
        let pyproject: PyProject = toml::from_str(&content).ok()?;
        let tool = pyproject.tool?;

        // Cascade: ripmap → ty → ruff → pyright
        // Each fallback indicates its source for transparency

        // 1. Native ripmap config (preferred)
        if let Some(raw) = tool.ripmap {
            return Some(Self::from_raw(raw, path.to_path_buf()));
        }

        // 2. Ty (astral type checker) - has nested src.include/exclude
        if let Some(ty) = tool.ty {
            if let Some(src) = ty.src {
                if src.include.is_some() || src.exclude.is_some() {
                    return Some(Self {
                        source: Some(format!("{} [tool.ty]", path.display())),
                        include: src.include.unwrap_or_default(),
                        exclude: src.exclude.unwrap_or_default(),
                        extend_exclude: Vec::new(),
                        src: Vec::new(),
                    });
                }
            }
        }

        // 3. Ruff - same structure as ripmap
        if let Some(ruff) = tool.ruff {
            if ruff.include.is_some() || ruff.exclude.is_some() || ruff.extend_exclude.is_some() {
                return Some(Self {
                    source: Some(format!("{} [tool.ruff]", path.display())),
                    include: ruff.include.unwrap_or_default(),
                    exclude: ruff.exclude.unwrap_or_default(),
                    extend_exclude: ruff.extend_exclude.unwrap_or_default(),
                    src: ruff
                        .src
                        .unwrap_or_default()
                        .into_iter()
                        .map(PathBuf::from)
                        .collect(),
                });
            }
        }

        // 4. Pyright - simpler include/exclude
        // (but really, you should be using ty by now)
        if let Some(pyright) = tool.pyright {
            if pyright.include.is_some() || pyright.exclude.is_some() {
                // Generate schizo sparkle prefix algorithmically
                let sparkle_chars = ['~', '*', '`', '.', '+', '^'];
                let colors = [
                    |s: &str| s.bright_magenta().to_string(),
                    |s: &str| s.bright_cyan().to_string(),
                    |s: &str| s.bright_yellow().to_string(),
                    |s: &str| s.bright_green().to_string(),
                    |s: &str| s.bright_red().to_string(),
                    |s: &str| s.bright_blue().to_string(),
                ];
                let prefix: String = (0..12)
                    .map(|i| {
                        let c = sparkle_chars[i % sparkle_chars.len()];
                        colors[i % colors.len()](&c.to_string())
                    })
                    .collect();
                let suffix: String = (0..12)
                    .rev()
                    .map(|i| {
                        let c = sparkle_chars[(i + 3) % sparkle_chars.len()];
                        colors[(i + 2) % colors.len()](&c.to_string())
                    })
                    .collect();

                eprintln!(
                    "   {} {} is now banned under international law. use {} instead {}",
                    prefix,
                    "pyright".bright_red().strikethrough(),
                    "ty".bright_cyan().bold().underline(),
                    suffix
                );

                return Some(Self {
                    source: Some(format!("{} [tool.pyright]", path.display())),
                    include: pyright.include.unwrap_or_default(),
                    exclude: pyright.exclude.unwrap_or_default(),
                    extend_exclude: Vec::new(),
                    src: Vec::new(),
                });
            }
        }

        None
    }

    fn from_raw(raw: RawConfig, source: PathBuf) -> Self {
        Self {
            source: Some(source.display().to_string()),
            include: raw.include.unwrap_or_default(),
            exclude: raw.exclude.unwrap_or_default(),
            extend_exclude: raw.extend_exclude.unwrap_or_default(),
            src: raw
                .src
                .unwrap_or_default()
                .into_iter()
                .map(PathBuf::from)
                .collect(),
        }
    }

    /// Get effective exclude patterns (defaults + extend-exclude, or custom exclude).
    pub fn effective_excludes(&self) -> Vec<String> {
        if !self.exclude.is_empty() {
            // Custom exclude replaces defaults
            self.exclude.clone()
        } else {
            // Use defaults + extend-exclude
            let mut patterns: Vec<String> =
                DEFAULT_EXCLUDES.iter().map(|s| s.to_string()).collect();
            patterns.extend(self.extend_exclude.clone());
            patterns
        }
    }

    /// Check if a path matches any include pattern.
    /// Returns true if no include patterns (include all), or if path matches any pattern.
    /// Handles both glob patterns (with *, ?, [) and directory prefixes (e.g., "src").
    pub fn matches_include(&self, path: &Path) -> bool {
        if self.include.is_empty() {
            return true;
        }
        let path_str = path.to_string_lossy();
        self.include.iter().any(|pattern| Self::matches_pattern(pattern, &path_str))
    }

    /// Check if a path matches any exclude pattern.
    /// Handles both glob patterns and directory prefixes.
    pub fn matches_exclude(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();
        self.effective_excludes()
            .iter()
            .any(|pattern| Self::matches_pattern(pattern, &path_str))
    }

    /// Match a pattern against a path, handling both globs and directory prefixes.
    fn matches_pattern(pattern: &str, path: &str) -> bool {
        // If pattern contains glob characters, use glob matching
        if pattern.contains('*') || pattern.contains('?') || pattern.contains('[') {
            glob_match::glob_match(pattern, path)
        } else {
            // Treat as directory prefix: "src" matches "src/foo.py", "src/bar/baz.rs"
            // Also handle trailing slash: "src/" same as "src"
            let prefix = pattern.trim_end_matches('/');
            path == prefix || path.starts_with(&format!("{}/", prefix))
        }
    }

    /// Check if a path should be included (matches include AND not exclude).
    pub fn should_include(&self, path: &Path) -> bool {
        self.matches_include(path) && !self.matches_exclude(path)
    }

    /// Format config for verbose display.
    pub fn display_summary(&self) -> String {
        let mut lines = Vec::new();

        if let Some(ref source) = self.source {
            lines.push(format!("   Config: {}", source));
        } else {
            lines.push("   Config: (defaults)".to_string());
        }

        if !self.include.is_empty() {
            lines.push(format!("   Include: {}", self.include.join(", ")));
        }

        let excludes = self.effective_excludes();
        if !excludes.is_empty() {
            // Show first few, then count
            if excludes.len() <= 3 {
                lines.push(format!("   Exclude: {}", excludes.join(", ")));
            } else {
                lines.push(format!(
                    "   Exclude: {}, ... (+{} more)",
                    excludes[..2].join(", "),
                    excludes.len() - 2
                ));
            }
        }

        if !self.src.is_empty() {
            let src_strs: Vec<_> = self.src.iter().map(|p| p.display().to_string()).collect();
            lines.push(format!("   Src roots: {}", src_strs.join(", ")));
        }

        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_excludes() {
        let config = Config::default();
        assert!(config.matches_exclude(Path::new("foo/node_modules/bar.js")));
        assert!(config.matches_exclude(Path::new("project/.git/config")));
        assert!(config.matches_exclude(Path::new("src/__pycache__/mod.pyc")));
        assert!(!config.matches_exclude(Path::new("src/main.py")));
    }

    #[test]
    fn test_include_patterns() {
        let config = Config {
            include: vec!["src/**".to_string(), "lib/**".to_string()],
            ..Default::default()
        };
        assert!(config.matches_include(Path::new("src/main.py")));
        assert!(config.matches_include(Path::new("lib/utils.py")));
        assert!(!config.matches_include(Path::new("tests/test_main.py")));
    }

    #[test]
    fn test_extend_exclude() {
        let config = Config {
            extend_exclude: vec!["**/generated/**".to_string()],
            ..Default::default()
        };
        // Should still have defaults
        assert!(config.matches_exclude(Path::new("node_modules/foo.js")));
        // Plus the extension
        assert!(config.matches_exclude(Path::new("src/generated/schema.py")));
    }

    #[test]
    fn test_directory_prefix_patterns() {
        // Test include with directory prefix (no glob chars)
        let config = Config {
            include: vec!["src".to_string()],
            ..Default::default()
        };
        assert!(config.matches_include(Path::new("src/main.py")));
        assert!(config.matches_include(Path::new("src/lib/utils.py")));
        assert!(!config.matches_include(Path::new("tests/test_main.py")));
        assert!(!config.matches_include(Path::new("srcfoo/bar.py"))); // "srcfoo" != "src/"

        // Test exclude with directory prefix
        let config = Config {
            exclude: vec!["vendor".to_string(), "src/gui/old/".to_string()],
            ..Default::default()
        };
        assert!(config.matches_exclude(Path::new("vendor/lib.py")));
        assert!(config.matches_exclude(Path::new("src/gui/old/widget.py")));
        assert!(!config.matches_exclude(Path::new("src/gui/new/widget.py")));
    }
}
