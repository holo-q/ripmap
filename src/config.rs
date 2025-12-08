//! Configuration loading from pyproject.toml and ripmap.toml.
//!
//! Follows conventions from ruff, black, mypy for familiarity:
//! - `[tool.ripmap]` section in pyproject.toml
//! - Standalone ripmap.toml as fallback
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

use std::path::{Path, PathBuf};
use serde::Deserialize;

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
    /// Source file for this config (for display).
    pub source: Option<PathBuf>,

    /// Glob patterns for files to include. If empty, include all source files.
    pub include: Vec<String>,

    /// Glob patterns for files to exclude. Replaces defaults if set.
    pub exclude: Vec<String>,

    /// Additional exclude patterns (extends defaults).
    pub extend_exclude: Vec<String>,

    /// Source root directories (affects depth weighting).
    pub src: Vec<PathBuf>,
}

/// Raw config as deserialized from TOML.
#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
struct RawConfig {
    include: Option<Vec<String>>,
    exclude: Option<Vec<String>>,
    extend_exclude: Option<Vec<String>>,
    src: Option<Vec<String>>,
}

/// Wrapper for pyproject.toml structure.
#[derive(Debug, Deserialize)]
struct PyProject {
    tool: Option<PyProjectTool>,
}

#[derive(Debug, Deserialize)]
struct PyProjectTool {
    ripmap: Option<RawConfig>,
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
        let raw = pyproject.tool?.ripmap?;
        Some(Self::from_raw(raw, path.to_path_buf()))
    }

    fn from_raw(raw: RawConfig, source: PathBuf) -> Self {
        Self {
            source: Some(source),
            include: raw.include.unwrap_or_default(),
            exclude: raw.exclude.unwrap_or_default(),
            extend_exclude: raw.extend_exclude.unwrap_or_default(),
            src: raw.src.unwrap_or_default().into_iter().map(PathBuf::from).collect(),
        }
    }

    /// Get effective exclude patterns (defaults + extend-exclude, or custom exclude).
    pub fn effective_excludes(&self) -> Vec<String> {
        if !self.exclude.is_empty() {
            // Custom exclude replaces defaults
            self.exclude.clone()
        } else {
            // Use defaults + extend-exclude
            let mut patterns: Vec<String> = DEFAULT_EXCLUDES.iter().map(|s| s.to_string()).collect();
            patterns.extend(self.extend_exclude.clone());
            patterns
        }
    }

    /// Check if a path matches any include pattern.
    /// Returns true if no include patterns (include all), or if path matches any pattern.
    pub fn matches_include(&self, path: &Path) -> bool {
        if self.include.is_empty() {
            return true;
        }
        let path_str = path.to_string_lossy();
        self.include.iter().any(|pattern| {
            glob_match::glob_match(pattern, &path_str)
        })
    }

    /// Check if a path matches any exclude pattern.
    pub fn matches_exclude(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();
        self.effective_excludes().iter().any(|pattern| {
            glob_match::glob_match(pattern, &path_str)
        })
    }

    /// Check if a path should be included (matches include AND not exclude).
    pub fn should_include(&self, path: &Path) -> bool {
        self.matches_include(path) && !self.matches_exclude(path)
    }

    /// Format config for verbose display.
    pub fn display_summary(&self) -> String {
        let mut lines = Vec::new();

        if let Some(ref source) = self.source {
            lines.push(format!("   Config: {}", source.display()));
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
                lines.push(format!("   Exclude: {}, ... (+{} more)",
                    excludes[..2].join(", "), excludes.len() - 2));
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
}
