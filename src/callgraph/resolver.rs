//! Call resolver: orchestrates multiple resolution strategies.
//!
//! The resolver is the brain of call graph construction:
//! 1. Builds context from tags (indexes, type maps)
//! 2. Runs each strategy on each call reference
//! 3. Picks highest-confidence resolution
//! 4. Builds the final CallGraph

use std::collections::HashMap;
use std::sync::Arc;
use crate::types::{Tag, TagKind};
use super::graph::{CallGraph, CallEdge, FunctionId};
use super::strategies::{ResolutionStrategy, ResolutionContext, Candidate};

/// Configuration for the resolver
#[derive(Debug, Clone)]
pub struct ResolverConfig {
    /// Minimum confidence threshold to accept a resolution
    pub min_confidence: f64,
    /// Whether to include unresolved calls as dangling nodes
    pub include_unresolved: bool,
    /// Language hint for strategy filtering (e.g., "python")
    pub language: Option<String>,
}

impl Default for ResolverConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.3,
            include_unresolved: false,
            language: None,
        }
    }
}

/// The main resolver that combines strategies to build call graphs.
pub struct CallResolver {
    strategies: Vec<Box<dyn ResolutionStrategy>>,
    config: ResolverConfig,
}

impl CallResolver {
    pub fn new() -> Self {
        Self {
            strategies: vec![],
            config: ResolverConfig::default(),
        }
    }

    pub fn with_config(mut self, config: ResolverConfig) -> Self {
        self.config = config;
        self
    }

    /// Add a resolution strategy (order matters for tie-breaking)
    pub fn add_strategy(&mut self, strategy: Box<dyn ResolutionStrategy>) {
        self.strategies.push(strategy);
    }

    /// Convenience builder pattern
    pub fn with_strategy(mut self, strategy: Box<dyn ResolutionStrategy>) -> Self {
        self.add_strategy(strategy);
        self
    }

    /// Build a complete call graph from extracted tags.
    ///
    /// Process:
    /// 1. Build resolution context (indexes, type maps)
    /// 2. Add all definitions as nodes
    /// 3. For each call reference, run strategies
    /// 4. Pick best resolution, add edge
    pub fn build_graph(&self, tags: &[Tag]) -> CallGraph {
        let context = ResolutionContext::new(tags);
        let mut graph = CallGraph::new();

        // Add all function definitions as nodes
        for tag in tags {
            if tag.kind.is_definition() {
                // Only add functions/methods, not classes or variables
                let node_type = tag.node_type.as_ref();
                if node_type.contains("function") || node_type.contains("method") {
                    let id = FunctionId::new(
                        tag.rel_fname.clone(),
                        tag.name.clone(),
                        tag.line,
                    ).with_parent_opt(tag.parent_name.clone());
                    graph.add_function(id);
                }
            }
        }

        // Process each call reference
        for tag in tags {
            if !tag.kind.is_reference() {
                continue;
            }

            // Find the enclosing function (caller)
            let caller = self.find_enclosing_function(tag, tags);
            let Some(caller) = caller else {
                continue; // Call not inside a function
            };

            // Run all strategies, collect candidates
            let mut all_candidates: Vec<(Candidate, &str)> = vec![];

            for strategy in &self.strategies {
                // Skip strategies that don't support this language
                if let Some(ref lang) = self.config.language {
                    if !strategy.supports_language(lang) {
                        continue;
                    }
                }

                let candidates = strategy.resolve(tag, &context);
                for c in candidates {
                    all_candidates.push((c, strategy.name()));
                }
            }

            // Pick the highest confidence candidate
            all_candidates.sort_by(|a, b| {
                b.0.confidence.partial_cmp(&a.0.confidence).unwrap()
            });

            if let Some((best, strategy_name)) = all_candidates.first() {
                if best.confidence >= self.config.min_confidence {
                    let edge = CallEdge::new(
                        best.confidence,
                        *strategy_name,
                        tag.line,
                    );
                    let edge = if let Some(ref hint) = best.type_hint {
                        edge.with_type_hint(hint.clone())
                    } else {
                        edge
                    };

                    graph.add_call(caller, best.target.clone(), edge);
                }
            } else if self.config.include_unresolved {
                // Add unresolved call as a dangling reference
                let unresolved = FunctionId::new(
                    Arc::<str>::from("?"), // Unknown file
                    tag.name.clone(),
                    0,
                );
                let edge = CallEdge::new(0.0, "unresolved", tag.line);
                graph.add_call(caller, unresolved, edge);
            }
        }

        graph
    }

    /// Find the function definition that encloses a given tag.
    /// Uses line number proximity within the same file.
    fn find_enclosing_function(&self, tag: &Tag, all_tags: &[Tag]) -> Option<FunctionId> {
        // Find all function definitions in the same file
        let mut functions: Vec<&Tag> = all_tags
            .iter()
            .filter(|t| {
                t.rel_fname == tag.rel_fname
                    && t.kind.is_definition()
                    && (t.node_type.contains("function") || t.node_type.contains("method"))
            })
            .collect();

        // Sort by line number descending (so we can find closest before)
        functions.sort_by_key(|t| std::cmp::Reverse(t.line));

        // Find the first function defined before this tag's line
        for func in functions {
            if func.line <= tag.line {
                return Some(FunctionId::new(
                    func.rel_fname.clone(),
                    func.name.clone(),
                    func.line,
                ).with_parent_opt(func.parent_name.clone()));
            }
        }

        None
    }

    /// Get statistics about resolution success
    pub fn stats(&self, tags: &[Tag]) -> ResolutionStats {
        let context = ResolutionContext::new(tags);
        let mut stats = ResolutionStats::default();

        for tag in tags {
            if !tag.kind.is_reference() {
                continue;
            }

            stats.total_calls += 1;

            let mut resolved = false;
            for strategy in &self.strategies {
                let candidates = strategy.resolve(tag, &context);
                if !candidates.is_empty() {
                    *stats.by_strategy.entry(strategy.name().to_string()).or_insert(0) += 1;
                    resolved = true;
                    break;
                }
            }

            if !resolved {
                stats.unresolved += 1;
            }
        }

        stats
    }
}

impl Default for CallResolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about call resolution
#[derive(Debug, Default)]
pub struct ResolutionStats {
    pub total_calls: usize,
    pub unresolved: usize,
    pub by_strategy: HashMap<String, usize>,
}

impl ResolutionStats {
    pub fn resolution_rate(&self) -> f64 {
        if self.total_calls == 0 {
            return 1.0;
        }
        (self.total_calls - self.unresolved) as f64 / self.total_calls as f64
    }
}

/// Builder for creating a fully-configured resolver with default strategies.
pub struct ResolverBuilder {
    config: ResolverConfig,
    same_file: bool,
    type_hints: bool,
    imports: bool,
    name_match: bool,
}

impl ResolverBuilder {
    pub fn new() -> Self {
        Self {
            config: ResolverConfig::default(),
            same_file: true,
            type_hints: true,
            imports: true,
            name_match: true,
        }
    }

    pub fn config(mut self, config: ResolverConfig) -> Self {
        self.config = config;
        self
    }

    pub fn same_file(mut self, enabled: bool) -> Self {
        self.same_file = enabled;
        self
    }

    pub fn type_hints(mut self, enabled: bool) -> Self {
        self.type_hints = enabled;
        self
    }

    pub fn imports(mut self, enabled: bool) -> Self {
        self.imports = enabled;
        self
    }

    pub fn name_match(mut self, enabled: bool) -> Self {
        self.name_match = enabled;
        self
    }

    pub fn build(self) -> CallResolver {
        use super::strategies::*;

        let mut resolver = CallResolver::new().with_config(self.config);

        // Add strategies in order of confidence (highest first)
        if self.same_file {
            resolver.add_strategy(Box::new(SameFileStrategy::new()));
        }
        if self.type_hints {
            resolver.add_strategy(Box::new(TypeHintStrategy::new()));
        }
        if self.imports {
            resolver.add_strategy(Box::new(ImportStrategy::new()));
        }
        if self.name_match {
            resolver.add_strategy(Box::new(NameMatchStrategy::new()));
        }

        resolver
    }
}

impl Default for ResolverBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_def(file: &str, name: &str, line: u32) -> Tag {
        Tag {
            rel_fname: Arc::from(file),
            fname: Arc::from(file),
            line,
            name: Arc::from(name),
            kind: TagKind::Def,
            node_type: Arc::from("function"),
            parent_name: None,
            parent_line: None,
            signature: None,
            fields: None,
            metadata: None,
        }
    }

    fn make_call(file: &str, name: &str, line: u32) -> Tag {
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
            metadata: None,
        }
    }

    #[test]
    fn test_build_graph() {
        let tags = vec![
            make_def("test.py", "main", 1),
            make_def("test.py", "helper", 10),
            make_call("test.py", "helper", 5), // main calls helper at line 5
        ];

        let resolver = ResolverBuilder::new().build();
        let graph = resolver.build_graph(&tags);

        assert_eq!(graph.function_count(), 2);
        assert_eq!(graph.call_count(), 1);
    }

    #[test]
    fn test_resolution_stats() {
        let tags = vec![
            make_def("test.py", "main", 1),
            make_def("test.py", "helper", 10),
            make_call("test.py", "helper", 5),
            make_call("test.py", "unknown", 7), // Unresolved
        ];

        let resolver = ResolverBuilder::new()
            .name_match(false) // Disable name matching to test unresolved
            .build();
        let stats = resolver.stats(&tags);

        assert_eq!(stats.total_calls, 2);
        assert!(stats.resolution_rate() > 0.0);
    }
}
