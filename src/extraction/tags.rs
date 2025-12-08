//! Tag extraction orchestration.
//!
//! This module ties together tree-sitter and regex parsers to provide the main
//! entry point for extracting tags from source files. It uses tree-sitter when
//! available (for full AST accuracy) and falls back to regex for unsupported
//! languages.
//!
//! # Parser Selection Strategy
//!
//! 1. Check file extension to determine language
//! 2. If tree-sitter supports the language → use AST-based extraction
//! 3. Otherwise → fall back to regex patterns
//!
//! This ensures we always get results while maximizing accuracy for supported
//! languages.

use std::path::Path;
use std::cell::RefCell;

use anyhow::Result;

use crate::types::Tag;
use crate::extraction::Parser;
use crate::extraction::treesitter::{TreeSitterParser, extension_to_language};

thread_local! {
    /// Thread-local tree-sitter parser (tree-sitter parsers are not thread-safe)
    static TS_PARSER: RefCell<TreeSitterParser> = RefCell::new(TreeSitterParser::new());
}

/// Extract symbol tags from a source file.
///
/// This is the main entry point for tag extraction. Given a file path and
/// its relative name (for display), it uses tree-sitter for supported languages
/// and falls back to regex patterns otherwise.
///
/// # Arguments
/// * `path` - Absolute path to the source file to parse
/// * `rel_fname` - Relative path for display in output (e.g., "src/lib.rs")
/// * `parser` - Regex parser instance (fallback for unsupported languages)
///
/// # Returns
/// Vector of extracted tags. Empty vector if file can't be parsed or has no symbols.
///
/// # Parser Selection
/// - Python, Rust, JavaScript, TypeScript, Go, Java, C, C++, Ruby, PHP → tree-sitter
/// - Other languages → regex fallback
pub fn extract_tags(
    path: &Path,
    rel_fname: &str,
    parser: &Parser,
) -> Result<Vec<Tag>> {
    // Determine language from extension
    let ext = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    let language = extension_to_language(ext);

    // Try tree-sitter first for supported languages
    if let Some(lang) = language {
        if TreeSitterParser::supports_language(lang) {
            let content = std::fs::read_to_string(path)?;
            let fname = path.to_string_lossy().to_string();

            let tags = TS_PARSER.with(|p| {
                p.borrow_mut().extract_tags(&content, lang, &fname, rel_fname)
            });

            // If tree-sitter found tags, use them
            if !tags.is_empty() {
                return Ok(tags);
            }
            // Otherwise fall through to regex fallback
        }
    }

    // Fallback to regex-based parsing
    parser.parse_file(path, rel_fname)
}

/// Extract tags using only tree-sitter (no fallback).
///
/// Use this when you specifically need AST-based extraction and want to
/// know if tree-sitter supported the language.
pub fn extract_tags_treesitter(
    content: &str,
    language: &str,
    fname: &str,
    rel_fname: &str,
) -> Vec<Tag> {
    TS_PARSER.with(|p| {
        p.borrow_mut().extract_tags(content, language, fname, rel_fname)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_extract_tags_python() {
        // Create a temporary Python file for testing
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_extract.py");
        std::fs::write(&test_file, "class Foo:\n    def bar(self):\n        pass\n").unwrap();

        let parser = Parser::new();
        let tags = extract_tags(&test_file, "test_extract.py", &parser).unwrap();

        assert!(tags.len() >= 1); // At least the class
        assert!(tags.iter().any(|t| t.name.as_ref() == "Foo"));

        // Cleanup
        std::fs::remove_file(test_file).ok();
    }

    #[test]
    fn test_extract_tags_rust() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_extract.rs");
        std::fs::write(&test_file, "pub fn foo() {}\nstruct Bar {}\n").unwrap();

        let parser = Parser::new();
        let tags = extract_tags(&test_file, "test_extract.rs", &parser).unwrap();

        assert!(tags.len() >= 2);
        assert!(tags.iter().any(|t| t.name.as_ref() == "foo"));
        assert!(tags.iter().any(|t| t.name.as_ref() == "Bar"));

        std::fs::remove_file(test_file).ok();
    }

    #[test]
    fn test_extract_tags_nonexistent_file() {
        let parser = Parser::new();
        let result = extract_tags(Path::new("/nonexistent/file.py"), "file.py", &parser);
        assert!(result.is_err());
    }

    #[test]
    fn test_treesitter_python_direct() {
        let code = r#"
class MyClass:
    def method(self):
        pass

def standalone():
    return 42
"#;
        let tags = extract_tags_treesitter(code, "python", "/test.py", "test.py");

        let names: Vec<&str> = tags.iter().map(|t| t.name.as_ref()).collect();
        assert!(names.contains(&"MyClass"), "Should find MyClass, got: {:?}", names);
        assert!(names.contains(&"method"), "Should find method, got: {:?}", names);
        assert!(names.contains(&"standalone"), "Should find standalone, got: {:?}", names);
    }

    #[test]
    fn test_treesitter_rust_direct() {
        let code = r#"
struct MyStruct {
    field: i32,
}

fn standalone() {
    println!("hello");
}
"#;
        let tags = extract_tags_treesitter(code, "rust", "/test.rs", "test.rs");

        let names: Vec<&str> = tags.iter().map(|t| t.name.as_ref()).collect();
        assert!(names.contains(&"MyStruct"), "Should find MyStruct, got: {:?}", names);
        assert!(names.contains(&"standalone"), "Should find standalone, got: {:?}", names);
    }

    #[test]
    fn test_treesitter_javascript_direct() {
        let code = r#"
class MyClass {
    method() { return 1; }
}

function standalone() {
    console.log("hello");
}
"#;
        let tags = extract_tags_treesitter(code, "javascript", "/test.js", "test.js");

        let names: Vec<&str> = tags.iter().map(|t| t.name.as_ref()).collect();
        assert!(names.contains(&"MyClass"), "Should find MyClass, got: {:?}", names);
        assert!(names.contains(&"method"), "Should find method, got: {:?}", names);
        assert!(names.contains(&"standalone"), "Should find standalone, got: {:?}", names);
    }
}
