//! Pluggable resolution strategies for call graph construction.
//!
//! Each strategy is an independent signal that can be enabled/disabled.
//! Strategies have confidence levels - higher confidence wins.
//!
//! # Strategy Hierarchy (by confidence)
//!
//! 1. SameFileStrategy (0.9) - caller and callee in same file
//! 2. TypeHintStrategy (0.85) - resolved via type annotations
//! 3. ImportStrategy (0.8) - resolved via import statements
//! 4. NameMatchStrategy (0.5) - fuzzy name matching (fallback)

use std::collections::HashMap;
use std::sync::Arc;
use crate::types::Tag;
use super::graph::FunctionId;

/// A resolution candidate with confidence score.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub target: FunctionId,
    pub confidence: f64,
    pub type_hint: Option<String>,
}

/// Context passed to resolution strategies.
/// Contains all extracted tags and derived indexes.
pub struct ResolutionContext<'a> {
    /// All extracted tags
    pub tags: &'a [Tag],
    /// Function definitions indexed by name
    pub definitions: HashMap<&'a str, Vec<&'a Tag>>,
    /// Type annotations: variable -> type
    pub type_map: HashMap<String, String>,
    /// Import map: imported_name -> (module, original_name)
    pub imports: HashMap<String, (String, String)>,
}

impl<'a> ResolutionContext<'a> {
    /// Build context from tags, extracting indexes for fast lookup.
    pub fn new(tags: &'a [Tag]) -> Self {
        let mut definitions: HashMap<&str, Vec<&Tag>> = HashMap::new();
        let mut type_map = HashMap::new();
        let mut imports = HashMap::new();

        for tag in tags {
            // Index definitions by name
            if tag.kind.is_definition() {
                definitions.entry(tag.name.as_ref()).or_default().push(tag);
            }

            // Extract type annotations from tag metadata
            // These come from tree-sitter captures like @var.type
            if let Some(ref meta) = tag.metadata {
                if let Some(var_type) = meta.get("var_type") {
                    if let Some(var_name) = meta.get("var_name") {
                        type_map.insert(var_name.clone(), var_type.clone());
                    }
                }
                if let Some(receiver_type) = meta.get("receiver_type") {
                    if let Some(receiver) = meta.get("receiver") {
                        type_map.insert(receiver.clone(), receiver_type.clone());
                    }
                }
                // Import tracking
                if let Some(import_module) = meta.get("import_module") {
                    if let Some(import_name) = meta.get("import_name") {
                        imports.insert(
                            import_name.clone(),
                            (import_module.clone(), import_name.clone()),
                        );
                    }
                }
            }
        }

        Self {
            tags,
            definitions,
            type_map,
            imports,
        }
    }

    /// Find definitions matching a name
    pub fn find_definitions(&self, name: &str) -> Vec<&Tag> {
        self.definitions.get(name).cloned().unwrap_or_default()
    }

    /// Get type for a variable/receiver
    pub fn get_type(&self, name: &str) -> Option<&String> {
        self.type_map.get(name)
    }

    /// Check if name is imported, return source module
    pub fn get_import(&self, name: &str) -> Option<&(String, String)> {
        self.imports.get(name)
    }
}

/// Trait for pluggable resolution strategies.
///
/// Each strategy implements one heuristic for resolving calls.
/// The resolver combines all enabled strategies, picking highest confidence.
pub trait ResolutionStrategy: Send + Sync {
    /// Strategy name for debugging/display
    fn name(&self) -> &'static str;

    /// Attempt to resolve a call reference to its definition.
    /// Returns all possible candidates with confidence scores.
    fn resolve(
        &self,
        call: &Tag,
        context: &ResolutionContext,
    ) -> Vec<Candidate>;

    /// Whether this strategy applies to the given language
    fn supports_language(&self, lang: &str) -> bool {
        // Default: support all languages
        let _ = lang;
        true
    }
}

// =============================================================================
// Strategy Implementations
// =============================================================================

/// Highest confidence: definitions in the same file.
///
/// If foo() is called and foo is defined in the same file, it's almost
/// certainly that definition (unless shadowed, which we ignore for now).
pub struct SameFileStrategy {
    pub confidence: f64,
}

impl SameFileStrategy {
    pub fn new() -> Self {
        Self { confidence: 0.9 }
    }
}

impl Default for SameFileStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl ResolutionStrategy for SameFileStrategy {
    fn name(&self) -> &'static str {
        "same_file"
    }

    fn resolve(&self, call: &Tag, context: &ResolutionContext) -> Vec<Candidate> {
        let defs = context.find_definitions(&call.name);

        defs.into_iter()
            .filter(|def| def.rel_fname == call.rel_fname)
            .map(|def| Candidate {
                target: FunctionId::new(
                    def.rel_fname.clone(),
                    def.name.clone(),
                    def.line,
                ).with_parent_opt(def.parent_name.clone()),
                confidence: self.confidence,
                type_hint: None,
            })
            .collect()
    }
}

/// Type hint resolution: use Python/TypeScript type annotations.
///
/// Given `x: MyClass` and `x.method()`, resolve to `MyClass.method`.
pub struct TypeHintStrategy {
    pub confidence: f64,
}

impl TypeHintStrategy {
    pub fn new() -> Self {
        Self { confidence: 0.85 }
    }
}

impl Default for TypeHintStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl ResolutionStrategy for TypeHintStrategy {
    fn name(&self) -> &'static str {
        "type_hint"
    }

    fn supports_language(&self, lang: &str) -> bool {
        matches!(lang, "python" | "typescript" | "tsx")
    }

    fn resolve(&self, call: &Tag, context: &ResolutionContext) -> Vec<Candidate> {
        // Only applies to method calls with a receiver
        let receiver = call.metadata.as_ref()
            .and_then(|m| m.get("receiver"))
            .map(|s| s.as_str());

        let Some(receiver) = receiver else {
            return vec![];
        };

        // Look up the receiver's type
        let Some(receiver_type) = context.get_type(receiver) else {
            return vec![];
        };

        // Find method definitions in that class
        let defs = context.find_definitions(&call.name);

        defs.into_iter()
            .filter(|def| {
                // Check if this definition is a method of the receiver's type
                def.parent_name.as_ref().map_or(false, |p| p.as_ref() == receiver_type)
            })
            .map(|def| Candidate {
                target: FunctionId::new(
                    def.rel_fname.clone(),
                    def.name.clone(),
                    def.line,
                ).with_parent(receiver_type.as_str()),
                confidence: self.confidence,
                type_hint: Some(format!("{}: {}", receiver, receiver_type)),
            })
            .collect()
    }
}

/// Import-based resolution: track what's imported from where.
///
/// Given `from mymodule import helper` and call to `helper()`,
/// resolve to the definition in mymodule.
pub struct ImportStrategy {
    pub confidence: f64,
}

impl ImportStrategy {
    pub fn new() -> Self {
        Self { confidence: 0.8 }
    }
}

impl Default for ImportStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl ResolutionStrategy for ImportStrategy {
    fn name(&self) -> &'static str {
        "import"
    }

    fn resolve(&self, call: &Tag, context: &ResolutionContext) -> Vec<Candidate> {
        // Check if this call target is imported
        let Some((module, original_name)) = context.get_import(&call.name) else {
            return vec![];
        };

        // Find definitions in files that match the module path
        let defs = context.find_definitions(original_name);

        defs.into_iter()
            .filter(|def| {
                // Check if the file path contains the module name
                // e.g., "mypackage/utils.py" matches import from "mypackage.utils"
                let module_path = module.replace('.', "/");
                def.rel_fname.contains(&module_path)
            })
            .map(|def| Candidate {
                target: FunctionId::new(
                    def.rel_fname.clone(),
                    def.name.clone(),
                    def.line,
                ).with_parent_opt(def.parent_name.clone()),
                confidence: self.confidence,
                type_hint: Some(format!("from {} import {}", module, original_name)),
            })
            .collect()
    }
}

/// Fallback: name matching across the codebase.
///
/// If no other strategy resolves, match by name alone.
/// Lower confidence because name collisions are common.
pub struct NameMatchStrategy {
    pub confidence: f64,
    /// Prefer matches in "nearby" files (same directory)
    pub proximity_boost: f64,
}

impl NameMatchStrategy {
    pub fn new() -> Self {
        Self {
            confidence: 0.5,
            proximity_boost: 0.1,
        }
    }
}

impl Default for NameMatchStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl ResolutionStrategy for NameMatchStrategy {
    fn name(&self) -> &'static str {
        "name_match"
    }

    fn resolve(&self, call: &Tag, context: &ResolutionContext) -> Vec<Candidate> {
        let defs = context.find_definitions(&call.name);
        let call_dir = std::path::Path::new(call.rel_fname.as_ref())
            .parent()
            .map(|p| p.to_string_lossy().to_string());

        defs.into_iter()
            .map(|def| {
                // Boost confidence for same-directory matches
                let def_dir = std::path::Path::new(def.rel_fname.as_ref())
                    .parent()
                    .map(|p| p.to_string_lossy().to_string());

                let confidence = if call_dir == def_dir {
                    self.confidence + self.proximity_boost
                } else {
                    self.confidence
                };

                Candidate {
                    target: FunctionId::new(
                        def.rel_fname.clone(),
                        def.name.clone(),
                        def.line,
                    ).with_parent_opt(def.parent_name.clone()),
                    confidence,
                    type_hint: None,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TagKind;

    fn make_def(file: &str, name: &str, line: u32, parent: Option<&str>) -> Tag {
        Tag {
            rel_fname: Arc::from(file),
            fname: Arc::from(file),
            line,
            name: Arc::from(name),
            kind: TagKind::Def,
            node_type: Arc::from("function"),
            parent_name: parent.map(|s| Arc::from(s)),
            parent_line: None,
            signature: None,
            fields: None,
            metadata: None,
        }
    }

    fn make_call(file: &str, name: &str, line: u32, receiver: Option<&str>) -> Tag {
        let metadata = receiver.map(|r| {
            let mut m = HashMap::new();
            m.insert("receiver".to_string(), r.to_string());
            m
        });

        Tag {
            rel_fname: Arc::from(file),
            fname: Arc::from(file),
            line,
            name: Arc::from(name),
            kind: TagKind::Ref,
            node_type: Arc::from("call"),
            parent_name: None,
            parent_line: None,
            signature: None,
            fields: None,
            metadata,
        }
    }

    #[test]
    fn test_same_file_strategy() {
        let tags = vec![
            make_def("test.py", "helper", 5, None),
            make_def("other.py", "helper", 10, None),
            make_call("test.py", "helper", 20, None),
        ];
        let ctx = ResolutionContext::new(&tags);
        let strategy = SameFileStrategy::new();

        let call = &tags[2];
        let candidates = strategy.resolve(call, &ctx);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].target.file.as_ref(), "test.py");
        assert_eq!(candidates[0].target.line, 5);
    }

    #[test]
    fn test_name_match_strategy() {
        let tags = vec![
            make_def("test.py", "helper", 5, None),
            make_def("other.py", "helper", 10, None),
            make_call("another.py", "helper", 20, None),
        ];
        let ctx = ResolutionContext::new(&tags);
        let strategy = NameMatchStrategy::new();

        let call = &tags[2];
        let candidates = strategy.resolve(call, &ctx);

        assert_eq!(candidates.len(), 2);
    }
}
