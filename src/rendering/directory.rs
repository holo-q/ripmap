//! Directory-style hierarchical rendering for ripmap.
//!
//! Renders a structured symbol map organized by file, then by symbol type:
//!
//! ```text
//! grepmap/facade.py: [bridge] [emergent] [recent] [high-churn]
//!   ⇄ changes with: grepmap/core/config.py(32%), grepmap/rendering/directory.py(30%)
//!   class GrepMap [api]:
//!     def: get_rel_fname(self, fname: str) -> str [api]
//!          _build_file_graph_from_symbol_graph(self, symbol_graph)
//!          get_ranked_tags(self, focus_targets: List[str], ...) -> Tuple[...]
//!     const: DEFAULT_MAP_TOKENS
//! ```
//!
//! Key design decisions:
//! - Group by file first (natural mental model for navigating codebases)
//! - Separate classes → methods → functions → constants (structure hierarchy)
//! - Badge visibility shows file/symbol role at a glance
//! - Temporal coupling shows implicit dependencies
//! - Token counting enables binary search for budget fitting

use crate::types::{DetailLevel, RankedTag, Tag};
use crate::callgraph::{CallGraph, FunctionId};
use super::colors::{Badge, Colorizer};
use std::collections::HashMap;
use std::sync::Arc;

/// Directory-style renderer - hierarchical symbol overview with badges.
///
/// Optimized for:
/// - Quick orientation ("what's in this codebase?")
/// - Understanding file roles (badges reveal structure)
/// - Spotting temporal patterns (coupling, churn)
/// - Fitting within LLM token budgets
pub struct DirectoryRenderer {
    /// Token counter for budget-aware rendering.
    /// Different LLMs use different tokenizers (cl100k_base, o200k_base, etc).
    token_counter: Box<dyn Fn(&str) -> usize + Send + Sync>,
}

impl DirectoryRenderer {
    /// Create a new directory renderer with a token counting function.
    ///
    /// The token counter is used for binary search to fit within budgets.
    /// Pass a closure that uses tiktoken-rs or your tokenizer of choice.
    pub fn new(token_counter: Box<dyn Fn(&str) -> usize + Send + Sync>) -> Self {
        Self { token_counter }
    }

    /// Create with a simple character-based token estimator (1 token ≈ 4 chars).
    /// Fast but less accurate than tiktoken.
    pub fn with_char_estimator() -> Self {
        Self::new(Box::new(|s: &str| (s.len() + 3) / 4))
    }

    /// Render ranked tags as a hierarchical directory-style map.
    ///
    /// # Arguments
    /// - `tags`: Ranked symbols to render (should be pre-sorted by rank)
    /// - `detail`: Level of detail for signatures/types
    /// - `badges`: File/symbol badges (structural, temporal, lifecycle)
    /// - `temporal_mates`: Files that change together (file -> [(other_file, coupling_score)])
    ///
    /// # Returns
    /// Formatted string with ANSI colors, ready for terminal output.
    pub fn render(
        &self,
        tags: &[RankedTag],
        detail: DetailLevel,
        badges: &HashMap<String, Vec<Badge>>,
        temporal_mates: &HashMap<String, Vec<(String, f64)>>,
    ) -> String {
        // Group tags by file
        let grouped = self.group_by_file(tags);

        let mut output = String::new();

        for (file_path, file_tags) in grouped {
            // Render file header with badges
            output.push_str(&self.render_file_header(&file_path, badges));
            output.push('\n');

            // Render temporal coupling if present
            if let Some(mates) = temporal_mates.get(file_path.as_ref()) {
                if !mates.is_empty() {
                    output.push_str(&self.render_temporal_coupling(mates));
                    output.push('\n');
                }
            }

            // Organize symbols by type (classes, functions, constants)
            let organized = self.organize_symbols(&file_tags);

            // Render classes with their methods
            for (class_name, class_tag, methods, fields) in organized.classes {
                output.push_str(&self.render_class(
                    &class_name,
                    &class_tag,
                    &methods,
                    &fields,
                    detail,
                    badges,
                ));
            }

            // Render standalone functions
            if !organized.functions.is_empty() {
                output.push_str(&self.render_functions(&organized.functions, detail, badges));
            }

            // Render constants
            if !organized.constants.is_empty() {
                output.push_str(&self.render_constants(&organized.constants, badges));
            }

            output.push('\n'); // Blank line between files
        }

        output
    }

    /// Estimate token count of rendered output.
    /// Uses the provided token counter for accurate estimation.
    pub fn estimate_tokens(&self, output: &str) -> usize {
        (self.token_counter)(output)
    }

    /// Render with call graph information showing calls/called-by for functions.
    ///
    /// This enhanced render shows:
    /// - `→ calls: foo(), bar()` for functions this function calls
    /// - `← called by: main(), helper()` for callers of this function
    ///
    /// # Arguments
    /// - `tags`: Ranked symbols to render
    /// - `detail`: Level of detail for signatures/types
    /// - `badges`: File/symbol badges
    /// - `temporal_mates`: Files that change together
    /// - `call_graph`: Optional call graph for showing call relationships
    pub fn render_with_calls(
        &self,
        tags: &[RankedTag],
        detail: DetailLevel,
        badges: &HashMap<String, Vec<Badge>>,
        temporal_mates: &HashMap<String, Vec<(String, f64)>>,
        call_graph: Option<&CallGraph>,
    ) -> String {
        // Group tags by file
        let grouped = self.group_by_file(tags);

        let mut output = String::new();

        for (file_path, file_tags) in grouped {
            // Render file header with badges
            output.push_str(&self.render_file_header(&file_path, badges));
            output.push('\n');

            // Render temporal coupling if present
            if let Some(mates) = temporal_mates.get(file_path.as_ref()) {
                if !mates.is_empty() {
                    output.push_str(&self.render_temporal_coupling(mates));
                    output.push('\n');
                }
            }

            // Organize symbols by type (classes, functions, constants)
            let organized = self.organize_symbols(&file_tags);

            // Render classes with their methods
            for (class_name, class_tag, methods, fields) in &organized.classes {
                output.push_str(&self.render_class_with_calls(
                    class_name,
                    class_tag,
                    methods,
                    fields,
                    detail,
                    badges,
                    call_graph,
                ));
            }

            // Render standalone functions with call info
            if !organized.functions.is_empty() {
                output.push_str(&self.render_functions_with_calls(
                    &organized.functions,
                    detail,
                    badges,
                    call_graph,
                ));
            }

            // Render constants
            if !organized.constants.is_empty() {
                output.push_str(&self.render_constants(&organized.constants, badges));
            }

            output.push('\n'); // Blank line between files
        }

        output
    }

    /// Render functions with call graph information.
    fn render_functions_with_calls(
        &self,
        functions: &[&RankedTag],
        detail: DetailLevel,
        badges: &HashMap<String, Vec<Badge>>,
        call_graph: Option<&CallGraph>,
    ) -> String {
        let mut output = String::from("    def:\n");

        for f in functions {
            // Function name and signature
            let mut line = String::from("      ");
            line.push_str(&Colorizer::function_name(&f.tag.name));

            if let Some(sig) = &f.tag.signature {
                line.push_str(&sig.render(detail));
            } else if detail >= DetailLevel::Medium {
                line.push_str("(...)");
            }

            // Add symbol-level badges if present
            let badge_key = format!("{}::{}", f.tag.rel_fname, f.tag.name);
            if let Some(symbol_badges) = badges.get(&badge_key) {
                if !symbol_badges.is_empty() {
                    line.push(' ');
                    line.push_str(&Colorizer::badge_group(symbol_badges));
                }
            }

            output.push_str(&line);
            output.push('\n');

            // Add call relationships if call graph is available
            if let Some(graph) = call_graph {
                let func_id = FunctionId::new(
                    f.tag.rel_fname.clone(),
                    f.tag.name.clone(),
                    f.tag.line,
                );

                // Show what this function calls
                let calls = graph.calls_from(&func_id);
                if !calls.is_empty() {
                    let call_names: Vec<_> = calls
                        .iter()
                        .take(5)
                        .map(|(target, edge)| {
                            let conf_str = if edge.confidence < 1.0 {
                                format!("({}%)", (edge.confidence * 100.0) as u32)
                            } else {
                                String::new()
                            };
                            format!("{}{}", target.qualified_name(), conf_str)
                        })
                        .collect();
                    output.push_str(&format!(
                        "        {} {}\n",
                        Colorizer::dim("→ calls:"),
                        call_names.join(", ")
                    ));
                }

                // Show what calls this function
                let callers = graph.calls_to(&func_id);
                if !callers.is_empty() {
                    let caller_names: Vec<_> = callers
                        .iter()
                        .take(5)
                        .map(|(source, _)| source.qualified_name())
                        .collect();
                    output.push_str(&format!(
                        "        {} {}\n",
                        Colorizer::dim("← called by:"),
                        caller_names.join(", ")
                    ));
                }
            }
        }

        output
    }

    /// Render a class with call graph info for methods.
    fn render_class_with_calls(
        &self,
        class_name: &str,
        class_tag: &Option<&RankedTag>,
        methods: &[&RankedTag],
        fields: &[&RankedTag],
        detail: DetailLevel,
        badges: &HashMap<String, Vec<Badge>>,
        call_graph: Option<&CallGraph>,
    ) -> String {
        let mut output = String::from("    ");

        // Class header
        output.push_str("class ");
        output.push_str(&Colorizer::class_name(class_name));

        // Class-level badges
        let badge_key = format!(
            "{}::{}",
            class_tag.map(|t| t.tag.rel_fname.as_ref()).unwrap_or(""),
            class_name
        );
        if let Some(class_badges) = badges.get(&badge_key) {
            if !class_badges.is_empty() {
                output.push(' ');
                output.push_str(&Colorizer::badge_group(class_badges));
            }
        }

        output.push_str(":\n");

        // Render fields
        if !fields.is_empty() && detail >= DetailLevel::Medium {
            output.push_str("      fields: ");
            let field_names: Vec<_> = fields
                .iter()
                .map(|f| {
                    if let Some(field_info) = f.tag.fields.as_ref().and_then(|fs| fs.first()) {
                        field_info.render(detail)
                    } else {
                        f.tag.name.to_string()
                    }
                })
                .collect();
            output.push_str(&field_names.join(", "));
            output.push('\n');
        }

        // Render methods with call info
        if !methods.is_empty() {
            output.push_str("      methods:\n");
            for m in methods {
                let method_name = self.render_method_inline(&m.tag, detail);
                output.push_str(&format!("        {}\n", method_name));

                // Add call relationships for methods
                if let Some(graph) = call_graph {
                    let func_id = FunctionId::new(
                        m.tag.rel_fname.clone(),
                        m.tag.name.clone(),
                        m.tag.line,
                    ).with_parent(class_name);

                    // Show what this method calls
                    let calls = graph.calls_from(&func_id);
                    if !calls.is_empty() {
                        let call_names: Vec<_> = calls
                            .iter()
                            .take(3)
                            .map(|(target, _)| target.qualified_name())
                            .collect();
                        output.push_str(&format!(
                            "          {} {}\n",
                            Colorizer::dim("→"),
                            call_names.join(", ")
                        ));
                    }
                }
            }
        }

        output
    }

    // ============ Internal helpers ============

    /// Group tags by file path, preserving rank order within each file.
    fn group_by_file<'a>(&self, tags: &'a [RankedTag]) -> Vec<(Arc<str>, Vec<&'a RankedTag>)> {
        let mut grouped: HashMap<Arc<str>, Vec<&'a RankedTag>> = HashMap::new();

        for tag in tags {
            grouped
                .entry(tag.tag.rel_fname.clone())
                .or_default()
                .push(tag);
        }

        // Sort files by highest rank in each file (most important files first)
        let mut files: Vec<_> = grouped.into_iter().collect();
        files.sort_by(|a, b| {
            let max_a = a.1.iter().map(|t| t.rank).fold(0.0_f64, f64::max);
            let max_b = b.1.iter().map(|t| t.rank).fold(0.0_f64, f64::max);
            max_b.partial_cmp(&max_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        files
    }

    /// Render file header with path and badges.
    fn render_file_header(&self, file_path: &str, badges: &HashMap<String, Vec<Badge>>) -> String {
        let mut header = format!("  {}", Colorizer::file_path(file_path));

        // Add file-level badges
        if let Some(file_badges) = badges.get(file_path) {
            if !file_badges.is_empty() {
                header.push_str(": ");
                header.push_str(&Colorizer::badge_group(file_badges));
            }
        }

        header
    }

    /// Render temporal coupling information (files that change together).
    fn render_temporal_coupling(&self, mates: &[(String, f64)]) -> String {
        let mut output = String::from("    ");
        output.push_str(&Colorizer::coupling_label());
        output.push(' ');

        let entries: Vec<_> = mates
            .iter()
            .take(3) // Show top 3 most coupled files
            .map(|(file, score)| Colorizer::coupling_entry(file, *score))
            .collect();

        output.push_str(&entries.join(", "));
        output
    }

    /// Organize symbols into classes, functions, constants.
    fn organize_symbols<'a>(&self, tags: &[&'a RankedTag]) -> OrganizedSymbols<'a> {
        let mut classes: HashMap<Arc<str>, (Option<&'a RankedTag>, Vec<&'a RankedTag>, Vec<&'a RankedTag>)> = HashMap::new();
        let mut functions = Vec::new();
        let mut constants = Vec::new();

        for ranked_tag in tags {
            let tag = &ranked_tag.tag;

            // Only process definitions
            if !tag.is_def() {
                continue;
            }

            match tag.node_type.as_ref() {
                "class" | "class_definition" | "struct" | "struct_item" => {
                    // Class definition
                    classes.entry(tag.name.clone()).or_default().0 = Some(*ranked_tag);
                }
                "method" | "method_definition" => {
                    // Method inside a class
                    if let Some(parent) = &tag.parent_name {
                        classes.entry(parent.clone()).or_default().1.push(*ranked_tag);
                    } else {
                        // Orphan method, treat as function
                        functions.push(*ranked_tag);
                    }
                }
                "field" | "field_definition" => {
                    // Field inside a class
                    if let Some(parent) = &tag.parent_name {
                        classes.entry(parent.clone()).or_default().2.push(*ranked_tag);
                    }
                }
                "function" | "function_definition" => {
                    // Standalone function
                    functions.push(*ranked_tag);
                }
                "constant" | "const_item" | "variable" => {
                    // Check if it looks like a constant (uppercase naming convention)
                    if tag.name.chars().all(|c| c.is_uppercase() || c == '_' || c.is_numeric()) {
                        constants.push(*ranked_tag);
                    } else {
                        // Regular variable, might be a function-level thing
                        // Skip or treat as function depending on context
                    }
                }
                _ => {
                    // Unknown node type, skip for now
                }
            }
        }

        // Convert HashMap to sorted Vec for deterministic output
        let mut class_list = Vec::new();
        for (class_name, (class_tag, methods, fields)) in classes {
            class_list.push((class_name, class_tag, methods, fields));
        }
        // Sort classes by name
        class_list.sort_by(|a, b| a.0.cmp(&b.0));

        OrganizedSymbols {
            classes: class_list,
            functions,
            constants,
        }
    }

    /// Render a class with its methods and fields.
    fn render_class(
        &self,
        class_name: &str,
        class_tag: &Option<&RankedTag>,
        methods: &[&RankedTag],
        fields: &[&RankedTag],
        detail: DetailLevel,
        badges: &HashMap<String, Vec<Badge>>,
    ) -> String {
        let mut output = String::from("    ");

        // Class header
        output.push_str("class ");
        output.push_str(&Colorizer::class_name(class_name));

        // Class-level badges (e.g., [api])
        let badge_key = format!("{}::{}",
            class_tag.map(|t| t.tag.rel_fname.as_ref()).unwrap_or(""),
            class_name
        );
        if let Some(class_badges) = badges.get(&badge_key) {
            if !class_badges.is_empty() {
                output.push(' ');
                output.push_str(&Colorizer::badge_group(class_badges));
            }
        }

        output.push_str(":\n");

        // Render fields if detail level is high enough
        if !fields.is_empty() && detail >= DetailLevel::Medium {
            output.push_str("      fields: ");
            let field_names: Vec<_> = fields
                .iter()
                .map(|f| {
                    if let Some(field_info) = f.tag.fields.as_ref().and_then(|fs| fs.first()) {
                        field_info.render(detail)
                    } else {
                        f.tag.name.to_string()
                    }
                })
                .collect();
            output.push_str(&field_names.join(", "));
            output.push('\n');
        }

        // Render methods
        if !methods.is_empty() {
            output.push_str("      def: ");
            let method_strs: Vec<_> = methods
                .iter()
                .map(|m| self.render_method_inline(&m.tag, detail))
                .collect();
            output.push_str(&method_strs.join("\n           "));
            output.push('\n');
        }

        output
    }

    /// Render a method inline (one line per method).
    fn render_method_inline(&self, tag: &Tag, detail: DetailLevel) -> String {
        let mut output = Colorizer::function_name(&tag.name);

        // Add signature if available
        if let Some(sig) = &tag.signature {
            output.push_str(&sig.render(detail));
        } else if detail >= DetailLevel::Medium {
            output.push_str("(...)");
        }

        output
    }

    /// Render standalone functions.
    fn render_functions(
        &self,
        functions: &[&RankedTag],
        detail: DetailLevel,
        badges: &HashMap<String, Vec<Badge>>,
    ) -> String {
        let mut output = String::from("    def: ");

        let func_strs: Vec<_> = functions
            .iter()
            .map(|f| {
                let mut s = Colorizer::function_name(&f.tag.name);
                if let Some(sig) = &f.tag.signature {
                    s.push_str(&sig.render(detail));
                } else if detail >= DetailLevel::Medium {
                    s.push_str("(...)");
                }

                // Add symbol-level badges if present
                let badge_key = format!("{}::{}", f.tag.rel_fname, f.tag.name);
                if let Some(symbol_badges) = badges.get(&badge_key) {
                    if !symbol_badges.is_empty() {
                        s.push(' ');
                        s.push_str(&Colorizer::badge_group(symbol_badges));
                    }
                }

                s
            })
            .collect();

        output.push_str(&func_strs.join("\n         "));
        output.push('\n');
        output
    }

    /// Render constants.
    fn render_constants(&self, constants: &[&RankedTag], badges: &HashMap<String, Vec<Badge>>) -> String {
        let mut output = String::from("    const: ");

        let const_strs: Vec<_> = constants
            .iter()
            .map(|c| {
                let mut s = Colorizer::constant_name(&c.tag.name);

                // Add symbol-level badges if present
                let badge_key = format!("{}::{}", c.tag.rel_fname, c.tag.name);
                if let Some(symbol_badges) = badges.get(&badge_key) {
                    if !symbol_badges.is_empty() {
                        s.push(' ');
                        s.push_str(&Colorizer::badge_group(symbol_badges));
                    }
                }

                s
            })
            .collect();

        output.push_str(&const_strs.join(", "));
        output.push('\n');
        output
    }
}

/// Organized symbols by type for rendering.
struct OrganizedSymbols<'a> {
    /// Classes with (name, class_tag, methods, fields)
    classes: Vec<(Arc<str>, Option<&'a RankedTag>, Vec<&'a RankedTag>, Vec<&'a RankedTag>)>,
    /// Standalone functions
    functions: Vec<&'a RankedTag>,
    /// Constants
    constants: Vec<&'a RankedTag>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SignatureInfo, TagKind};
    use std::sync::Arc;

    fn make_tag(
        file: &str,
        name: &str,
        node_type: &str,
        parent: Option<&str>,
        signature: Option<SignatureInfo>,
    ) -> Tag {
        Tag {
            rel_fname: file.into(),
            fname: format!("/{}", file).into(),
            line: 1,
            name: name.into(),
            kind: TagKind::Def,
            node_type: node_type.into(),
            parent_name: parent.map(|p| p.into()),
            parent_line: None,
            signature,
            fields: None,
        metadata: None,
        }
    }

    #[test]
    fn test_group_by_file() {
        let renderer = DirectoryRenderer::with_char_estimator();

        let tags = vec![
            RankedTag::new(0.9, make_tag("a.rs", "foo", "function", None, None)),
            RankedTag::new(0.8, make_tag("b.rs", "bar", "function", None, None)),
            RankedTag::new(0.7, make_tag("a.rs", "baz", "function", None, None)),
        ];

        let grouped = renderer.group_by_file(&tags);

        // Should have 2 files
        assert_eq!(grouped.len(), 2);

        // a.rs should come first (higher max rank)
        assert_eq!(grouped[0].0.as_ref(), "a.rs");
        assert_eq!(grouped[0].1.len(), 2);
    }

    #[test]
    fn test_organize_symbols() {
        let renderer = DirectoryRenderer::with_char_estimator();

        let tags = vec![
            RankedTag::new(0.9, make_tag("test.rs", "MyClass", "class", None, None)),
            RankedTag::new(0.8, make_tag("test.rs", "method1", "method", Some("MyClass"), None)),
            RankedTag::new(0.7, make_tag("test.rs", "standalone", "function", None, None)),
            RankedTag::new(0.6, make_tag("test.rs", "MAX_SIZE", "constant", None, None)),
        ];

        let tag_refs: Vec<_> = tags.iter().collect();
        let organized = renderer.organize_symbols(&tag_refs);

        assert_eq!(organized.classes.len(), 1);
        assert_eq!(organized.functions.len(), 1);
        assert_eq!(organized.constants.len(), 1);

        // Check class has method
        assert_eq!(organized.classes[0].2.len(), 1);
    }

    #[test]
    fn test_render_empty() {
        let renderer = DirectoryRenderer::with_char_estimator();
        let badges = HashMap::new();
        let temporal = HashMap::new();

        let output = renderer.render(&[], DetailLevel::Medium, &badges, &temporal);

        // Should produce empty or minimal output
        assert!(output.trim().is_empty());
    }

    #[test]
    fn test_estimate_tokens() {
        let renderer = DirectoryRenderer::with_char_estimator();
        let text = "Hello world!"; // 12 chars → ~3 tokens

        let tokens = renderer.estimate_tokens(text);

        assert!(tokens > 0);
        assert!(tokens <= 12); // Should be reasonable
    }

    #[test]
    fn test_render_with_signature() {
        let renderer = DirectoryRenderer::with_char_estimator();

        let sig = SignatureInfo {
            parameters: vec![("x".into(), Some("int".into()))],
            return_type: Some("bool".into()),
            decorators: vec![],
            raw: None,
        };

        let tags = vec![
            RankedTag::new(0.9, make_tag("test.py", "check", "function", None, Some(sig))),
        ];

        let badges = HashMap::new();
        let temporal = HashMap::new();

        let output = renderer.render(&tags, DetailLevel::High, &badges, &temporal);

        // Should contain function name
        assert!(output.contains("check"));
        // Should contain signature elements at high detail
        assert!(output.contains("int"));
        assert!(output.contains("bool"));
    }
}
