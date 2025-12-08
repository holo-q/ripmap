//! Test↔Source file coupling detection.
//!
//! Codex optimization identified that path-aware test↔crate coupling edges
//! are a missing architectural feature. This module detects test files and
//! links them to the source files they test.
//!
//! ## Rationale
//!
//! When debugging or exploring code, test files are highly relevant to their
//! implementation files. But there's no explicit code reference between them -
//! the coupling is implicit in file naming conventions:
//!
//! ```text
//! tests/test_parser.rs  →  src/parser.rs
//! foo_test.py           →  foo.py
//! Button.spec.tsx       →  Button.tsx
//! ```
//!
//! By detecting these patterns and creating synthetic edges, we can:
//! 1. Boost test files when focusing on implementation
//! 2. Boost implementation when focusing on tests
//! 3. Surface test↔impl pairs in focus expansion
//!
//! ## Usage
//!
//! ```ignore
//! let detector = TestCouplingDetector::new();
//! let edges = detector.detect_couplings(&file_list);
//! // edges: Vec<(test_file, source_file, confidence)>
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Detects test↔source file couplings based on naming conventions.
///
/// Supports multiple language conventions:
/// - Rust: `tests/test_*.rs`, `*_test.rs`
/// - Python: `test_*.py`, `*_test.py`, `tests/test_*.py`
/// - JavaScript/TypeScript: `*.spec.ts`, `*.test.js`, `__tests__/*.js`
/// - Go: `*_test.go`
pub struct TestCouplingDetector {
    /// Minimum confidence to include a coupling
    min_confidence: f64,
}

impl TestCouplingDetector {
    pub fn new() -> Self {
        Self {
            min_confidence: 0.5,
        }
    }

    /// Set minimum confidence threshold for couplings.
    pub fn with_min_confidence(mut self, min: f64) -> Self {
        self.min_confidence = min;
        self
    }

    /// Detect test↔source couplings from a list of file paths.
    ///
    /// Returns edges as (test_file, source_file, confidence).
    /// Confidence is based on how strong the naming match is:
    /// - 0.9: Exact match (test_foo.py → foo.py exists)
    /// - 0.7: Directory match (tests/test_foo.py → src/foo.py exists)
    /// - 0.5: Pattern match but source not found
    pub fn detect_couplings(&self, files: &[impl AsRef<Path>]) -> Vec<(Arc<str>, Arc<str>, f64)> {
        // Build a set of all files for existence checking
        let file_set: HashMap<String, &Path> = files
            .iter()
            .map(|f| {
                let p = f.as_ref();
                let name = p
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                (name, p)
            })
            .collect();

        // Also index by stem (without extension) for flexible matching
        let stem_map: HashMap<String, Vec<&Path>> = files
            .iter()
            .map(|f| f.as_ref())
            .filter_map(|p| {
                let stem = p.file_stem()?.to_string_lossy().to_string();
                Some((stem, p))
            })
            .fold(HashMap::new(), |mut map, (stem, path)| {
                map.entry(stem).or_default().push(path);
                map
            });

        let mut couplings = Vec::new();

        for file in files {
            let path = file.as_ref();
            let path_str = path.to_string_lossy();

            // Check if this is a test file
            if let Some((source_stem, confidence_base)) = self.is_test_file(path) {
                // Try to find the corresponding source file
                if let Some(source_candidates) = stem_map.get(&source_stem) {
                    for source_path in source_candidates {
                        // Skip if source is also a test file
                        if self.is_test_file(source_path).is_some() {
                            continue;
                        }

                        // Calculate final confidence based on path proximity
                        let confidence =
                            self.calculate_confidence(path, source_path, confidence_base);

                        if confidence >= self.min_confidence {
                            couplings.push((
                                Arc::from(path_str.as_ref()),
                                Arc::from(source_path.to_string_lossy().as_ref()),
                                confidence,
                            ));
                        }
                    }
                }
            }
        }

        couplings
    }

    /// Check if a file is a test file and extract the source stem it tests.
    ///
    /// Returns (source_stem, base_confidence) if it's a test file.
    fn is_test_file(&self, path: &Path) -> Option<(String, f64)> {
        let file_name = path.file_name()?.to_string_lossy();
        let stem = path.file_stem()?.to_string_lossy();
        let path_str = path.to_string_lossy();

        // Rust: tests/test_foo.rs, foo_test.rs
        if file_name.ends_with(".rs") {
            if stem.starts_with("test_") {
                return Some((stem[5..].to_string(), 0.9));
            }
            if stem.ends_with("_test") {
                return Some((stem[..stem.len() - 5].to_string(), 0.9));
            }
            // In tests/ directory
            if path_str.contains("/tests/") || path_str.contains("\\tests\\") {
                return Some((stem.to_string(), 0.7));
            }
        }

        // Python: test_foo.py, foo_test.py
        if file_name.ends_with(".py") {
            if stem.starts_with("test_") {
                return Some((stem[5..].to_string(), 0.9));
            }
            if stem.ends_with("_test") {
                return Some((stem[..stem.len() - 5].to_string(), 0.9));
            }
        }

        // JavaScript/TypeScript: foo.spec.ts, foo.test.js
        if file_name.ends_with(".ts")
            || file_name.ends_with(".tsx")
            || file_name.ends_with(".js")
            || file_name.ends_with(".jsx")
        {
            if stem.ends_with(".spec") {
                return Some((stem[..stem.len() - 5].to_string(), 0.9));
            }
            if stem.ends_with(".test") {
                return Some((stem[..stem.len() - 5].to_string(), 0.9));
            }
            // __tests__ directory convention
            if path_str.contains("/__tests__/") || path_str.contains("\\__tests__\\") {
                return Some((stem.to_string(), 0.8));
            }
        }

        // Go: foo_test.go
        if file_name.ends_with(".go") && stem.ends_with("_test") {
            return Some((stem[..stem.len() - 5].to_string(), 0.9));
        }

        None
    }

    /// Calculate confidence based on path proximity.
    ///
    /// Higher confidence when test and source are in related directories:
    /// - Same directory: +0.1
    /// - tests/ → src/: +0.05
    /// - Completely unrelated: no bonus
    fn calculate_confidence(&self, test_path: &Path, source_path: &Path, base: f64) -> f64 {
        let test_parent = test_path.parent().map(|p| p.to_string_lossy().to_string());
        let source_parent = source_path
            .parent()
            .map(|p| p.to_string_lossy().to_string());

        match (test_parent, source_parent) {
            (Some(tp), Some(sp)) => {
                // Same directory
                if tp == sp {
                    return (base + 0.1).min(1.0);
                }

                // tests/ → src/ pattern
                if (tp.contains("/tests") || tp.contains("\\tests"))
                    && (sp.contains("/src") || sp.contains("\\src"))
                {
                    return (base + 0.05).min(1.0);
                }

                // Adjacent directories (e.g., foo/tests and foo/src)
                let tp_parts: Vec<_> = tp.split(['/', '\\']).collect();
                let sp_parts: Vec<_> = sp.split(['/', '\\']).collect();
                if tp_parts.len() > 1 && sp_parts.len() > 1 {
                    if tp_parts[..tp_parts.len() - 1] == sp_parts[..sp_parts.len() - 1] {
                        return (base + 0.05).min(1.0);
                    }
                }

                base
            }
            _ => base,
        }
    }

    /// Convert couplings to symbol graph edges format.
    ///
    /// Returns edges in the format expected by FocusResolver.expand_via_graph:
    /// (from_file, from_symbol, to_file, to_symbol)
    ///
    /// For file-level coupling, we use a synthetic symbol name "__file__".
    pub fn as_symbol_edges(
        &self,
        couplings: &[(Arc<str>, Arc<str>, f64)],
    ) -> Vec<(Arc<str>, Arc<str>, Arc<str>, Arc<str>)> {
        let file_symbol: Arc<str> = Arc::from("__file__");

        couplings
            .iter()
            .flat_map(|(test, source, _conf)| {
                // Bidirectional: test↔source
                vec![
                    (
                        test.clone(),
                        file_symbol.clone(),
                        source.clone(),
                        file_symbol.clone(),
                    ),
                    (
                        source.clone(),
                        file_symbol.clone(),
                        test.clone(),
                        file_symbol.clone(),
                    ),
                ]
            })
            .collect()
    }
}

impl Default for TestCouplingDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_detect_rust_test_files() {
        let detector = TestCouplingDetector::new();

        let files: Vec<PathBuf> = vec![
            "src/parser.rs".into(),
            "tests/test_parser.rs".into(),
            "src/lexer.rs".into(),
            "src/lexer_test.rs".into(),
        ];

        let couplings = detector.detect_couplings(&files);

        // Should find test_parser → parser and lexer_test → lexer
        assert_eq!(couplings.len(), 2);

        let test_files: Vec<_> = couplings.iter().map(|(t, _, _)| t.as_ref()).collect();
        assert!(test_files.contains(&"tests/test_parser.rs"));
        assert!(test_files.contains(&"src/lexer_test.rs"));
    }

    #[test]
    fn test_detect_python_test_files() {
        let detector = TestCouplingDetector::new();

        let files: Vec<PathBuf> = vec![
            "mymodule.py".into(),
            "test_mymodule.py".into(),
            "utils.py".into(),
            "utils_test.py".into(),
        ];

        let couplings = detector.detect_couplings(&files);
        assert_eq!(couplings.len(), 2);
    }

    #[test]
    fn test_detect_js_spec_files() {
        let detector = TestCouplingDetector::new();

        let files: Vec<PathBuf> = vec![
            "Button.tsx".into(),
            "Button.spec.tsx".into(),
            "utils.ts".into(),
            "utils.test.ts".into(),
        ];

        let couplings = detector.detect_couplings(&files);
        assert_eq!(couplings.len(), 2);
    }

    #[test]
    fn test_no_self_coupling() {
        let detector = TestCouplingDetector::new();

        // Test files shouldn't couple to other test files
        let files: Vec<PathBuf> = vec!["test_foo.py".into(), "test_bar.py".into()];

        let couplings = detector.detect_couplings(&files);
        assert!(couplings.is_empty());
    }

    #[test]
    fn test_confidence_same_directory() {
        let detector = TestCouplingDetector::new();

        let files: Vec<PathBuf> = vec!["src/foo.rs".into(), "src/foo_test.rs".into()];

        let couplings = detector.detect_couplings(&files);
        assert_eq!(couplings.len(), 1);
        // Same directory should boost confidence
        assert!(couplings[0].2 > 0.9);
    }

    #[test]
    fn test_as_symbol_edges() {
        let detector = TestCouplingDetector::new();

        let couplings = vec![(Arc::from("test_foo.py"), Arc::from("foo.py"), 0.9)];

        let edges = detector.as_symbol_edges(&couplings);

        // Should be bidirectional
        assert_eq!(edges.len(), 2);
    }
}
