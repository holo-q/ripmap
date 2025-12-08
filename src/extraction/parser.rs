//! Tree-sitter parser with regex fallback.
//!
//! This module provides source code parsing for tag extraction.
//! Currently implements a **regex-based fallback** that handles common patterns
//! across Python, Rust, and JavaScript. Tree-sitter integration will be added
//! later for higher accuracy and AST-based extraction.
//!
//! Design rationale:
//! - Regex is "good enough" for 80% of cases and gets us shipping fast
//! - Patterns focus on DEFINITIONS (functions, classes, methods)
//! - Line number tracking via byte offset -> newline counting
//! - Each language gets its own parser function to keep patterns clean

use std::path::Path;
use std::sync::Arc;
use crate::types::{Tag, TagKind};
use anyhow::Result;
use regex::Regex;
use once_cell::sync::Lazy;

/// Main parser struct - currently stateless, but allows future tree-sitter state.
pub struct Parser;

impl Parser {
    pub fn new() -> Self {
        Self
    }

    /// Parse a file and extract tags using regex patterns.
    /// This is a fallback - tree-sitter will be added later for accuracy.
    ///
    /// Returns a vector of tags representing symbol definitions found in the file.
    pub fn parse_file(&self, path: &Path, rel_fname: &str) -> Result<Vec<Tag>> {
        let content = std::fs::read_to_string(path)?;
        let lang = detect_language(path);

        match lang {
            Language::Python => parse_python(&content, path, rel_fname),
            Language::Rust => parse_rust(&content, path, rel_fname),
            Language::JavaScript => parse_javascript(&content, path, rel_fname),
            Language::TypeScript => parse_typescript(&content, path, rel_fname),
            Language::Unknown => Ok(vec![]),
        }
    }
}

impl Default for Parser {
    fn default() -> Self {
        Self::new()
    }
}

/// Supported languages for regex extraction.
/// Each language has its own pattern set optimized for common definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Language {
    Python,
    Rust,
    JavaScript,
    TypeScript,
    Unknown,
}

/// Detect language from file extension.
/// Simple but effective - maps extensions to Language enum.
fn detect_language(path: &Path) -> Language {
    match path.extension().and_then(|e| e.to_str()) {
        Some("py") => Language::Python,
        Some("rs") => Language::Rust,
        Some("js" | "jsx") => Language::JavaScript,
        Some("ts" | "tsx") => Language::TypeScript,
        _ => Language::Unknown,
    }
}

/// Calculate 1-indexed line number from byte offset.
/// Counts newlines before the offset to determine the line.
fn line_number(content: &str, byte_offset: usize) -> u32 {
    content[..byte_offset].matches('\n').count() as u32 + 1
}

// ============================================================================
// PYTHON PARSING
// ============================================================================

/// Regex patterns for Python symbol extraction.
/// Cached as static to avoid recompilation on every parse.
mod python_patterns {
    use super::*;

    /// Match class definitions: `class Foo:` or `class Foo(Bar):`
    pub static CLASS: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?m)^class\s+(\w+)").expect("Invalid Python class regex")
    });

    /// Match top-level function definitions: `def foo(`
    pub static FUNCTION: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?m)^def\s+(\w+)\s*\(").expect("Invalid Python function regex")
    });

    /// Match method definitions (indented): `    def bar(`
    pub static METHOD: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?m)^(?:    |\t)def\s+(\w+)\s*\(").expect("Invalid Python method regex")
    });

    /// Match top-level assignments: `FOO = ...` (constants/globals)
    pub static ASSIGNMENT: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?m)^([A-Z_][A-Z0-9_]*)\s*=").expect("Invalid Python assignment regex")
    });
}

/// Parse Python source code for class/function/method definitions.
///
/// Extraction strategy:
/// - Classes: `class Name:` patterns at start of line
/// - Functions: `def name(` at start of line (top-level)
/// - Methods: `def name(` indented (inside classes)
/// - Constants: `UPPERCASE_NAME =` at start of line
///
/// Limitations (to be fixed with tree-sitter):
/// - Can't accurately determine parent scope (which class owns which method)
/// - Doesn't handle nested functions
/// - May miss async def, decorators
fn parse_python(content: &str, path: &Path, rel_fname: &str) -> Result<Vec<Tag>> {
    let mut tags = Vec::new();
    let fname: Arc<str> = Arc::from(path.to_string_lossy().into_owned());
    let rel: Arc<str> = Arc::from(rel_fname);

    // Extract classes
    for cap in python_patterns::CLASS.captures_iter(content) {
        let line = line_number(content, cap.get(0).unwrap().start());
        tags.push(Tag {
            rel_fname: rel.clone(),
            fname: fname.clone(),
            line,
            name: Arc::from(&cap[1]),
            kind: TagKind::Def,
            node_type: Arc::from("class"),
            parent_name: None,
            parent_line: None,
            signature: None,
            fields: None, // TODO: Extract class fields when tree-sitter lands
        metadata: None,
        });
    }

    // Extract top-level functions
    for cap in python_patterns::FUNCTION.captures_iter(content) {
        let line = line_number(content, cap.get(0).unwrap().start());
        tags.push(Tag {
            rel_fname: rel.clone(),
            fname: fname.clone(),
            line,
            name: Arc::from(&cap[1]),
            kind: TagKind::Def,
            node_type: Arc::from("function"),
            parent_name: None,
            parent_line: None,
            signature: None, // TODO: Extract signature params when tree-sitter lands
            fields: None,
            metadata: None,
        });
    }

    // Extract methods (indented functions - likely class methods)
    for cap in python_patterns::METHOD.captures_iter(content) {
        let line = line_number(content, cap.get(0).unwrap().start());
        tags.push(Tag {
            rel_fname: rel.clone(),
            fname: fname.clone(),
            line,
            name: Arc::from(&cap[1]),
            kind: TagKind::Def,
            node_type: Arc::from("method"),
            parent_name: None, // TODO: Link to parent class when tree-sitter lands
            parent_line: None,
            signature: None,
            fields: None,
            metadata: None,
        });
    }

    // Extract top-level constants/globals (UPPERCASE assignments)
    for cap in python_patterns::ASSIGNMENT.captures_iter(content) {
        let line = line_number(content, cap.get(0).unwrap().start());
        tags.push(Tag {
            rel_fname: rel.clone(),
            fname: fname.clone(),
            line,
            name: Arc::from(&cap[1]),
            kind: TagKind::Def,
            node_type: Arc::from("constant"),
            parent_name: None,
            parent_line: None,
            signature: None,
            fields: None,
            metadata: None,
        });
    }

    Ok(tags)
}

// ============================================================================
// RUST PARSING
// ============================================================================

/// Regex patterns for Rust symbol extraction.
mod rust_patterns {
    use super::*;

    /// Match function definitions: `fn foo(` or `pub fn foo<T>(`
    pub static FUNCTION: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?m)^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)").expect("Invalid Rust fn regex")
    });

    /// Match struct definitions: `struct Foo` or `pub struct Foo<T>`
    pub static STRUCT: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?m)^\s*(?:pub\s+)?struct\s+(\w+)").expect("Invalid Rust struct regex")
    });

    /// Match enum definitions: `enum Bar` or `pub enum Bar`
    pub static ENUM: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?m)^\s*(?:pub\s+)?enum\s+(\w+)").expect("Invalid Rust enum regex")
    });

    /// Match trait definitions: `trait Baz` or `pub trait Baz`
    pub static TRAIT: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?m)^\s*(?:pub\s+)?trait\s+(\w+)").expect("Invalid Rust trait regex")
    });

    /// Match impl blocks: `impl Foo` or `impl Trait for Foo`
    pub static IMPL: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?m)^\s*impl(?:\s+\w+\s+for)?\s+(\w+)").expect("Invalid Rust impl regex")
    });

    /// Match const definitions: `const FOO:` or `pub const FOO:`
    pub static CONST: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?m)^\s*(?:pub\s+)?const\s+(\w+)").expect("Invalid Rust const regex")
    });

    /// Match static definitions: `static BAR:` or `pub static BAR:`
    pub static STATIC: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?m)^\s*(?:pub\s+)?static\s+(\w+)").expect("Invalid Rust static regex")
    });
}

/// Parse Rust source code for function/struct/enum/trait definitions.
///
/// Extraction strategy:
/// - Functions: `fn name` patterns (handles pub/async modifiers)
/// - Structs: `struct Name` patterns
/// - Enums: `enum Name` patterns
/// - Traits: `trait Name` patterns
/// - Impls: `impl Name` patterns (for tracking methods)
/// - Constants: `const NAME:` patterns
/// - Statics: `static NAME:` patterns
///
/// Limitations (to be fixed with tree-sitter):
/// - Can't link impl block methods to their parent struct
/// - Doesn't extract generic parameters
/// - May miss some visibility modifiers
fn parse_rust(content: &str, path: &Path, rel_fname: &str) -> Result<Vec<Tag>> {
    let mut tags = Vec::new();
    let fname: Arc<str> = Arc::from(path.to_string_lossy().into_owned());
    let rel: Arc<str> = Arc::from(rel_fname);

    // Extract functions
    for cap in rust_patterns::FUNCTION.captures_iter(content) {
        let line = line_number(content, cap.get(0).unwrap().start());
        tags.push(Tag {
            rel_fname: rel.clone(),
            fname: fname.clone(),
            line,
            name: Arc::from(&cap[1]),
            kind: TagKind::Def,
            node_type: Arc::from("function"),
            parent_name: None,
            parent_line: None,
            signature: None,
            fields: None,
            metadata: None,
        });
    }

    // Extract structs
    for cap in rust_patterns::STRUCT.captures_iter(content) {
        let line = line_number(content, cap.get(0).unwrap().start());
        tags.push(Tag {
            rel_fname: rel.clone(),
            fname: fname.clone(),
            line,
            name: Arc::from(&cap[1]),
            kind: TagKind::Def,
            node_type: Arc::from("struct"),
            parent_name: None,
            parent_line: None,
            signature: None,
            fields: None, // TODO: Extract struct fields when tree-sitter lands
        metadata: None,
        });
    }

    // Extract enums
    for cap in rust_patterns::ENUM.captures_iter(content) {
        let line = line_number(content, cap.get(0).unwrap().start());
        tags.push(Tag {
            rel_fname: rel.clone(),
            fname: fname.clone(),
            line,
            name: Arc::from(&cap[1]),
            kind: TagKind::Def,
            node_type: Arc::from("enum"),
            parent_name: None,
            parent_line: None,
            signature: None,
            fields: None,
            metadata: None,
        });
    }

    // Extract traits
    for cap in rust_patterns::TRAIT.captures_iter(content) {
        let line = line_number(content, cap.get(0).unwrap().start());
        tags.push(Tag {
            rel_fname: rel.clone(),
            fname: fname.clone(),
            line,
            name: Arc::from(&cap[1]),
            kind: TagKind::Def,
            node_type: Arc::from("trait"),
            parent_name: None,
            parent_line: None,
            signature: None,
            fields: None,
            metadata: None,
        });
    }

    // Extract impl blocks (for tracking methods later)
    for cap in rust_patterns::IMPL.captures_iter(content) {
        let line = line_number(content, cap.get(0).unwrap().start());
        tags.push(Tag {
            rel_fname: rel.clone(),
            fname: fname.clone(),
            line,
            name: Arc::from(&cap[1]),
            kind: TagKind::Def,
            node_type: Arc::from("impl"),
            parent_name: None,
            parent_line: None,
            signature: None,
            fields: None,
            metadata: None,
        });
    }

    // Extract constants
    for cap in rust_patterns::CONST.captures_iter(content) {
        let line = line_number(content, cap.get(0).unwrap().start());
        tags.push(Tag {
            rel_fname: rel.clone(),
            fname: fname.clone(),
            line,
            name: Arc::from(&cap[1]),
            kind: TagKind::Def,
            node_type: Arc::from("const"),
            parent_name: None,
            parent_line: None,
            signature: None,
            fields: None,
            metadata: None,
        });
    }

    // Extract statics
    for cap in rust_patterns::STATIC.captures_iter(content) {
        let line = line_number(content, cap.get(0).unwrap().start());
        tags.push(Tag {
            rel_fname: rel.clone(),
            fname: fname.clone(),
            line,
            name: Arc::from(&cap[1]),
            kind: TagKind::Def,
            node_type: Arc::from("static"),
            parent_name: None,
            parent_line: None,
            signature: None,
            fields: None,
            metadata: None,
        });
    }

    Ok(tags)
}

// ============================================================================
// JAVASCRIPT PARSING
// ============================================================================

/// Regex patterns for JavaScript symbol extraction.
mod js_patterns {
    use super::*;

    /// Match function declarations: `function foo(` or `async function foo(`
    pub static FUNCTION: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?m)^\s*(?:async\s+)?function\s+(\w+)\s*\(").expect("Invalid JS function regex")
    });

    /// Match class definitions: `class Foo` or `export class Foo`
    pub static CLASS: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?m)^\s*(?:export\s+)?class\s+(\w+)").expect("Invalid JS class regex")
    });

    /// Match const arrow functions: `const foo = (`
    pub static CONST_ARROW: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?m)^\s*(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\(").expect("Invalid JS const arrow regex")
    });

    /// Match const regular assignments: `const FOO = ...`
    pub static CONST_ASSIGN: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?m)^\s*(?:export\s+)?const\s+([A-Z_][A-Z0-9_]*)\s*=").expect("Invalid JS const regex")
    });

    /// Match method definitions in classes: `  methodName(` or `  async methodName(`
    pub static METHOD: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?m)^\s+(?:async\s+)?(\w+)\s*\(").expect("Invalid JS method regex")
    });
}

/// Parse JavaScript source code for class/function definitions.
///
/// Extraction strategy:
/// - Functions: `function name(` patterns
/// - Classes: `class Name` patterns
/// - Arrow functions: `const name = (` patterns
/// - Constants: `const UPPERCASE_NAME =` patterns
/// - Methods: indented `methodName(` inside classes
///
/// Limitations (to be fixed with tree-sitter):
/// - Can't distinguish between arrow functions and other const assignments
/// - Can't accurately link methods to their parent class
/// - Doesn't handle destructuring assignments
fn parse_javascript(content: &str, path: &Path, rel_fname: &str) -> Result<Vec<Tag>> {
    let mut tags = Vec::new();
    let fname: Arc<str> = Arc::from(path.to_string_lossy().into_owned());
    let rel: Arc<str> = Arc::from(rel_fname);

    // Extract classes
    for cap in js_patterns::CLASS.captures_iter(content) {
        let line = line_number(content, cap.get(0).unwrap().start());
        tags.push(Tag {
            rel_fname: rel.clone(),
            fname: fname.clone(),
            line,
            name: Arc::from(&cap[1]),
            kind: TagKind::Def,
            node_type: Arc::from("class"),
            parent_name: None,
            parent_line: None,
            signature: None,
            fields: None,
            metadata: None,
        });
    }

    // Extract function declarations
    for cap in js_patterns::FUNCTION.captures_iter(content) {
        let line = line_number(content, cap.get(0).unwrap().start());
        tags.push(Tag {
            rel_fname: rel.clone(),
            fname: fname.clone(),
            line,
            name: Arc::from(&cap[1]),
            kind: TagKind::Def,
            node_type: Arc::from("function"),
            parent_name: None,
            parent_line: None,
            signature: None,
            fields: None,
            metadata: None,
        });
    }

    // Extract const arrow functions
    for cap in js_patterns::CONST_ARROW.captures_iter(content) {
        let line = line_number(content, cap.get(0).unwrap().start());
        tags.push(Tag {
            rel_fname: rel.clone(),
            fname: fname.clone(),
            line,
            name: Arc::from(&cap[1]),
            kind: TagKind::Def,
            node_type: Arc::from("function"),
            parent_name: None,
            parent_line: None,
            signature: None,
            fields: None,
            metadata: None,
        });
    }

    // Extract const assignments (constants)
    for cap in js_patterns::CONST_ASSIGN.captures_iter(content) {
        let line = line_number(content, cap.get(0).unwrap().start());
        tags.push(Tag {
            rel_fname: rel.clone(),
            fname: fname.clone(),
            line,
            name: Arc::from(&cap[1]),
            kind: TagKind::Def,
            node_type: Arc::from("constant"),
            parent_name: None,
            parent_line: None,
            signature: None,
            fields: None,
            metadata: None,
        });
    }

    Ok(tags)
}

// ============================================================================
// TYPESCRIPT PARSING
// ============================================================================

/// Regex patterns for TypeScript symbol extraction.
/// TypeScript extends JavaScript patterns with type annotations.
mod ts_patterns {
    use super::*;

    /// Match interface definitions: `interface Foo` or `export interface Foo`
    pub static INTERFACE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?m)^\s*(?:export\s+)?interface\s+(\w+)").expect("Invalid TS interface regex")
    });

    /// Match type aliases: `type Foo =` or `export type Foo =`
    pub static TYPE_ALIAS: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?m)^\s*(?:export\s+)?type\s+(\w+)\s*=").expect("Invalid TS type regex")
    });

    /// Match enum definitions: `enum Color` or `export enum Color`
    pub static ENUM: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(?m)^\s*(?:export\s+)?enum\s+(\w+)").expect("Invalid TS enum regex")
    });
}

/// Parse TypeScript source code for type/interface/enum definitions.
///
/// TypeScript gets both JavaScript patterns (classes, functions) PLUS
/// type-specific patterns (interfaces, type aliases, enums).
fn parse_typescript(content: &str, path: &Path, rel_fname: &str) -> Result<Vec<Tag>> {
    // Start with JavaScript patterns
    let mut tags = parse_javascript(content, path, rel_fname)?;

    let fname: Arc<str> = Arc::from(path.to_string_lossy().into_owned());
    let rel: Arc<str> = Arc::from(rel_fname);

    // Add TypeScript-specific patterns

    // Extract interfaces
    for cap in ts_patterns::INTERFACE.captures_iter(content) {
        let line = line_number(content, cap.get(0).unwrap().start());
        tags.push(Tag {
            rel_fname: rel.clone(),
            fname: fname.clone(),
            line,
            name: Arc::from(&cap[1]),
            kind: TagKind::Def,
            node_type: Arc::from("interface"),
            parent_name: None,
            parent_line: None,
            signature: None,
            fields: None,
            metadata: None,
        });
    }

    // Extract type aliases
    for cap in ts_patterns::TYPE_ALIAS.captures_iter(content) {
        let line = line_number(content, cap.get(0).unwrap().start());
        tags.push(Tag {
            rel_fname: rel.clone(),
            fname: fname.clone(),
            line,
            name: Arc::from(&cap[1]),
            kind: TagKind::Def,
            node_type: Arc::from("type"),
            parent_name: None,
            parent_line: None,
            signature: None,
            fields: None,
            metadata: None,
        });
    }

    // Extract enums
    for cap in ts_patterns::ENUM.captures_iter(content) {
        let line = line_number(content, cap.get(0).unwrap().start());
        tags.push(Tag {
            rel_fname: rel.clone(),
            fname: fname.clone(),
            line,
            name: Arc::from(&cap[1]),
            kind: TagKind::Def,
            node_type: Arc::from("enum"),
            parent_name: None,
            parent_line: None,
            signature: None,
            fields: None,
            metadata: None,
        });
    }

    Ok(tags)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_number() {
        let content = "line 1\nline 2\nline 3\n";
        assert_eq!(line_number(content, 0), 1);
        assert_eq!(line_number(content, 7), 2);
        assert_eq!(line_number(content, 14), 3);
    }

    #[test]
    fn test_detect_language() {
        assert_eq!(detect_language(Path::new("foo.py")), Language::Python);
        assert_eq!(detect_language(Path::new("bar.rs")), Language::Rust);
        assert_eq!(detect_language(Path::new("baz.js")), Language::JavaScript);
        assert_eq!(detect_language(Path::new("qux.ts")), Language::TypeScript);
        assert_eq!(detect_language(Path::new("unknown.txt")), Language::Unknown);
    }

    #[test]
    fn test_parse_python_class() {
        let content = "class Foo:\n    pass\n";
        let path = Path::new("test.py");
        let tags = parse_python(content, path, "test.py").unwrap();
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].name.as_ref(), "Foo");
        assert_eq!(tags[0].node_type.as_ref(), "class");
        assert_eq!(tags[0].line, 1);
    }

    #[test]
    fn test_parse_python_function() {
        let content = "def bar():\n    pass\n";
        let path = Path::new("test.py");
        let tags = parse_python(content, path, "test.py").unwrap();
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].name.as_ref(), "bar");
        assert_eq!(tags[0].node_type.as_ref(), "function");
    }

    #[test]
    fn test_parse_rust_function() {
        let content = "pub fn foo() {}\n";
        let path = Path::new("test.rs");
        let tags = parse_rust(content, path, "test.rs").unwrap();
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].name.as_ref(), "foo");
        assert_eq!(tags[0].node_type.as_ref(), "function");
    }

    #[test]
    fn test_parse_rust_struct() {
        let content = "pub struct Bar {\n    field: i32\n}\n";
        let path = Path::new("test.rs");
        let tags = parse_rust(content, path, "test.rs").unwrap();
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].name.as_ref(), "Bar");
        assert_eq!(tags[0].node_type.as_ref(), "struct");
    }

    #[test]
    fn test_parse_javascript_class() {
        let content = "class MyClass {\n  constructor() {}\n}\n";
        let path = Path::new("test.js");
        let tags = parse_javascript(content, path, "test.js").unwrap();
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].name.as_ref(), "MyClass");
        assert_eq!(tags[0].node_type.as_ref(), "class");
    }

    #[test]
    fn test_parse_typescript_interface() {
        let content = "export interface IFoo {\n  bar: string;\n}\n";
        let path = Path::new("test.ts");
        let tags = parse_typescript(content, path, "test.ts").unwrap();
        assert!(tags.iter().any(|t| t.name.as_ref() == "IFoo" && t.node_type.as_ref() == "interface"));
    }
}
