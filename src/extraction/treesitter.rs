//! Tree-sitter based code parsing with .scm query support.
//!
//! This module provides AST-aware tag extraction using tree-sitter grammars
//! and query files. It replaces regex-based parsing with full structural
//! understanding of the code.
//!
//! # Query Format
//!
//! The .scm query files use tree-sitter's query syntax with captures:
//! - `@name.definition.class` - class/struct name
//! - `@definition.class` - entire class node
//! - `@name.definition.function` - function name
//! - `@definition.function` - entire function node
//! - `@name.reference.call` - function call name
//! - `@reference.call` - entire call node

use std::collections::HashMap;
use std::sync::Arc;

use once_cell::sync::Lazy;
use streaming_iterator::StreamingIterator;
use tree_sitter::{Language, Parser as TsParser, Query, QueryCursor};

use crate::types::{Tag, TagKind};

/// Embedded query files - compiled into the binary
mod queries {
    pub const PYTHON: &str = include_str!("../../queries/python-tags.scm");
    pub const RUST: &str = include_str!("../../queries/rust-tags.scm");
    pub const JAVASCRIPT: &str = include_str!("../../queries/javascript-tags.scm");
    pub const TYPESCRIPT: &str = include_str!("../../queries/typescript-tags.scm");
    pub const GO: &str = include_str!("../../queries/go-tags.scm");
    pub const JAVA: &str = include_str!("../../queries/java-tags.scm");
    pub const C: &str = include_str!("../../queries/c-tags.scm");
    pub const CPP: &str = include_str!("../../queries/cpp-tags.scm");
    pub const RUBY: &str = include_str!("../../queries/ruby-tags.scm");
    pub const PHP: &str = include_str!("../../queries/php-tags.scm");
    pub const C_SHARP: &str = include_str!("../../queries/c_sharp-tags.scm");
    pub const KOTLIN: &str = include_str!("../../queries/kotlin-tags.scm");
    pub const SCALA: &str = include_str!("../../queries/scala-tags.scm");
}

/// Language configuration with grammar and query
struct LangConfig {
    language: Language,
    query: Query,
}

/// Get tree-sitter language by name
fn get_language(name: &str) -> Option<Language> {
    match name {
        "python" => Some(tree_sitter_python::LANGUAGE.into()),
        "rust" => Some(tree_sitter_rust::LANGUAGE.into()),
        "javascript" | "jsx" => Some(tree_sitter_javascript::LANGUAGE.into()),
        "typescript" | "tsx" => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        "go" => Some(tree_sitter_go::LANGUAGE.into()),
        "java" => Some(tree_sitter_java::LANGUAGE.into()),
        "c" => Some(tree_sitter_c::LANGUAGE.into()),
        "cpp" | "c++" | "cc" | "cxx" => Some(tree_sitter_cpp::LANGUAGE.into()),
        "ruby" => Some(tree_sitter_ruby::LANGUAGE.into()),
        "php" => Some(tree_sitter_php::LANGUAGE_PHP.into()),
        _ => None,
    }
}

/// Get query source for a language
fn get_query_source(name: &str) -> Option<&'static str> {
    match name {
        "python" => Some(queries::PYTHON),
        "rust" => Some(queries::RUST),
        "javascript" | "jsx" => Some(queries::JAVASCRIPT),
        "typescript" | "tsx" => Some(queries::TYPESCRIPT),
        "go" => Some(queries::GO),
        "java" => Some(queries::JAVA),
        "c" => Some(queries::C),
        "cpp" | "c++" | "cc" | "cxx" => Some(queries::CPP),
        "ruby" => Some(queries::RUBY),
        "php" => Some(queries::PHP),
        _ => None,
    }
}

/// Map file extension to language name
pub fn extension_to_language(ext: &str) -> Option<&'static str> {
    match ext {
        "py" | "pyi" | "pyw" => Some("python"),
        "rs" => Some("rust"),
        "js" | "mjs" | "cjs" => Some("javascript"),
        "jsx" => Some("jsx"),
        "ts" | "mts" | "cts" => Some("typescript"),
        "tsx" => Some("tsx"),
        "go" => Some("go"),
        "java" => Some("java"),
        "c" | "h" => Some("c"),
        "cpp" | "cc" | "cxx" | "hpp" | "hxx" | "hh" => Some("cpp"),
        "rb" | "rake" | "gemspec" => Some("ruby"),
        "php" | "php3" | "php4" | "php5" | "phtml" => Some("php"),
        "cs" => Some("c_sharp"),
        "kt" | "kts" => Some("kotlin"),
        "scala" | "sc" => Some("scala"),
        _ => None,
    }
}

/// Cached language configurations
static LANG_CONFIGS: Lazy<HashMap<&'static str, LangConfig>> = Lazy::new(|| {
    let mut configs = HashMap::new();

    for lang_name in &["python", "rust", "javascript", "typescript", "go", "java", "c", "cpp", "ruby", "php"] {
        if let (Some(language), Some(query_src)) = (get_language(lang_name), get_query_source(lang_name)) {
            // Try to compile the query, skip if it fails (query syntax might not match grammar version)
            match Query::new(&language, query_src) {
                Ok(query) => {
                    configs.insert(*lang_name, LangConfig { language, query });
                }
                Err(e) => {
                    eprintln!("Warning: Failed to compile query for {}: {}", lang_name, e);
                }
            }
        }
    }

    configs
});

/// Tree-sitter based parser for extracting tags from source code.
pub struct TreeSitterParser {
    /// Thread-local parser instances (tree-sitter parsers are not thread-safe)
    parser: TsParser,
}

impl TreeSitterParser {
    /// Create a new tree-sitter parser.
    pub fn new() -> Self {
        Self {
            parser: TsParser::new(),
        }
    }

    /// Check if a language is supported.
    pub fn supports_language(lang: &str) -> bool {
        LANG_CONFIGS.contains_key(lang)
    }

    /// Extract tags from source code using tree-sitter queries.
    pub fn extract_tags(
        &mut self,
        content: &str,
        language: &str,
        fname: &str,
        rel_fname: &str,
    ) -> Vec<Tag> {
        let config = match LANG_CONFIGS.get(language) {
            Some(c) => c,
            None => return Vec::new(),
        };

        // Set language and parse
        if self.parser.set_language(&config.language).is_err() {
            return Vec::new();
        }

        let tree = match self.parser.parse(content, None) {
            Some(t) => t,
            None => return Vec::new(),
        };

        let mut tags = Vec::new();
        let mut cursor = QueryCursor::new();

        // Track capture names for processing
        let capture_names: Vec<String> = config.query.capture_names().iter().map(|s| s.to_string()).collect();

        // Use streaming iterator pattern for tree-sitter 0.24+
        let mut matches = cursor.matches(&config.query, tree.root_node(), content.as_bytes());
        while let Some(m) = matches.next() {
            let mut name: Option<String> = None;
            let mut node_type: Option<&str> = None;
            let mut kind: Option<TagKind> = None;
            let mut line: Option<u32> = None;

            for capture in m.captures {
                let capture_name = capture_names.get(capture.index as usize).map(|s| s.as_str()).unwrap_or("");
                let node = capture.node;
                let text = node.utf8_text(content.as_bytes()).unwrap_or("").to_string();

                // Parse capture name: @name.definition.class, @definition.function, etc.
                if capture_name.starts_with("name.") {
                    line = Some(node.start_position().row as u32 + 1);
                    name = Some(text);

                    // Extract kind from capture name
                    if capture_name.contains(".definition.") {
                        kind = Some(TagKind::Def);
                    } else if capture_name.contains(".reference.") {
                        kind = Some(TagKind::Ref);
                    }

                    // Extract node type from capture name
                    if capture_name.ends_with(".class") {
                        node_type = Some("class");
                    } else if capture_name.ends_with(".function") {
                        node_type = Some("function");
                    } else if capture_name.ends_with(".method") {
                        node_type = Some("method");
                    } else if capture_name.ends_with(".call") {
                        node_type = Some("call");
                    } else if capture_name.ends_with(".interface") {
                        node_type = Some("interface");
                    } else if capture_name.ends_with(".module") {
                        node_type = Some("module");
                    } else if capture_name.ends_with(".macro") {
                        node_type = Some("macro");
                    } else if capture_name.ends_with(".implementation") {
                        node_type = Some("impl");
                    }
                }
            }

            // Create tag if we have the required fields
            if let (Some(name), Some(node_type), Some(kind), Some(line)) = (name, node_type, kind, line) {
                // Skip empty names or very short names (likely noise)
                if name.is_empty() || (name.len() == 1 && !name.chars().next().unwrap().is_alphabetic()) {
                    continue;
                }

                tags.push(Tag {
                    rel_fname: Arc::from(rel_fname),
                    fname: Arc::from(fname),
                    line,
                    name: Arc::from(name.as_str()),
                    kind,
                    node_type: Arc::from(node_type),
                    parent_name: None,  // TODO: extract from AST parent traversal
                    parent_line: None,
                    signature: None,    // TODO: extract from AST
                    fields: None,
                metadata: None,
                });
            }
        }

        tags
    }
}

impl Default for TreeSitterParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extension_mapping() {
        assert_eq!(extension_to_language("py"), Some("python"));
        assert_eq!(extension_to_language("rs"), Some("rust"));
        assert_eq!(extension_to_language("js"), Some("javascript"));
        assert_eq!(extension_to_language("ts"), Some("typescript"));
        assert_eq!(extension_to_language("unknown"), None);
    }

    #[test]
    fn test_python_parsing() {
        let mut parser = TreeSitterParser::new();
        let code = r#"
class MyClass:
    def method(self):
        pass

def standalone_function():
    return 42

standalone_function()
"#;
        let tags = parser.extract_tags(code, "python", "/test.py", "test.py");

        // Should find class, method, function, and call
        let names: Vec<&str> = tags.iter().map(|t| t.name.as_ref()).collect();
        assert!(names.contains(&"MyClass"), "Should find MyClass");
        assert!(names.contains(&"method"), "Should find method");
        assert!(names.contains(&"standalone_function"), "Should find standalone_function");
    }

    #[test]
    fn test_rust_parsing() {
        let mut parser = TreeSitterParser::new();
        let code = r#"
struct MyStruct {
    field: i32,
}

impl MyStruct {
    fn new() -> Self {
        Self { field: 0 }
    }
}

fn standalone() {
    println!("hello");
}
"#;
        let tags = parser.extract_tags(code, "rust", "/test.rs", "test.rs");

        let names: Vec<&str> = tags.iter().map(|t| t.name.as_ref()).collect();
        assert!(names.contains(&"MyStruct"), "Should find MyStruct");
        assert!(names.contains(&"new"), "Should find new method");
        assert!(names.contains(&"standalone"), "Should find standalone function");
    }

    #[test]
    fn test_javascript_parsing() {
        let mut parser = TreeSitterParser::new();
        let code = r#"
class MyClass {
    constructor() {}
    method() { return 1; }
}

function standalone() {
    console.log("hello");
}

const arrow = () => 42;
"#;
        let tags = parser.extract_tags(code, "javascript", "/test.js", "test.js");

        let names: Vec<&str> = tags.iter().map(|t| t.name.as_ref()).collect();
        assert!(names.contains(&"MyClass"), "Should find MyClass");
        assert!(names.contains(&"method"), "Should find method");
        assert!(names.contains(&"standalone"), "Should find standalone function");
    }

    #[test]
    fn test_unsupported_language() {
        let mut parser = TreeSitterParser::new();
        let tags = parser.extract_tags("content", "unsupported", "/test.xyz", "test.xyz");
        assert!(tags.is_empty());
    }
}
