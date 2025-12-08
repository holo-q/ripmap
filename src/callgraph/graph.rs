//! Core call graph data structures.
//!
//! The graph is strategy-agnostic - it just stores resolved edges.
//! Resolution strategies populate it; PageRank consumes it.

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Unique identifier for a function/method in the codebase.
///
/// Uses (file, name, line) tuple for disambiguation.
/// Line is needed because Python allows multiple functions with same name
/// in different scopes within one file.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FunctionId {
    /// Relative file path
    pub file: Arc<str>,
    /// Function/method name
    pub name: Arc<str>,
    /// Line number (for disambiguation)
    pub line: u32,
    /// Parent class/module if applicable
    pub parent: Option<Arc<str>>,
}

impl FunctionId {
    pub fn new(file: impl Into<Arc<str>>, name: impl Into<Arc<str>>, line: u32) -> Self {
        Self {
            file: file.into(),
            name: name.into(),
            line,
            parent: None,
        }
    }

    pub fn with_parent(mut self, parent: impl Into<Arc<str>>) -> Self {
        self.parent = Some(parent.into());
        self
    }

    pub fn with_parent_opt(mut self, parent: Option<Arc<str>>) -> Self {
        self.parent = parent;
        self
    }

    /// Qualified name for display: "Class.method" or just "function"
    pub fn qualified_name(&self) -> String {
        match &self.parent {
            Some(p) => format!("{}.{}", p, self.name),
            None => self.name.to_string(),
        }
    }
}

/// An edge in the call graph representing a function call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallEdge {
    /// How confident we are in this resolution (0.0 - 1.0)
    pub confidence: f64,
    /// Which strategy resolved this call
    pub strategy: String,
    /// Line number where the call occurs
    pub call_site: u32,
    /// Type information if available (e.g., "self: MyClass")
    pub type_hint: Option<String>,
}

impl CallEdge {
    pub fn new(confidence: f64, strategy: impl Into<String>, call_site: u32) -> Self {
        Self {
            confidence,
            strategy: strategy.into(),
            call_site,
            type_hint: None,
        }
    }

    pub fn with_type_hint(mut self, hint: impl Into<String>) -> Self {
        self.type_hint = Some(hint.into());
        self
    }
}

/// The call graph: functions as nodes, calls as edges.
///
/// Uses petgraph for efficient graph algorithms (PageRank, traversal).
/// Multiple edges between same pair allowed (same function called multiple times).
#[derive(Debug)]
pub struct CallGraph {
    /// The underlying directed graph
    graph: DiGraph<FunctionId, CallEdge>,
    /// Fast lookup: FunctionId -> NodeIndex
    index: HashMap<FunctionId, NodeIndex>,
}

impl CallGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            index: HashMap::new(),
        }
    }

    /// Add a function to the graph, returns its node index.
    /// Idempotent - returns existing index if already present.
    pub fn add_function(&mut self, id: FunctionId) -> NodeIndex {
        if let Some(&idx) = self.index.get(&id) {
            return idx;
        }
        let idx = self.graph.add_node(id.clone());
        self.index.insert(id, idx);
        idx
    }

    /// Add a call edge between two functions.
    /// Both functions are auto-added if not present.
    pub fn add_call(&mut self, caller: FunctionId, callee: FunctionId, edge: CallEdge) {
        let caller_idx = self.add_function(caller);
        let callee_idx = self.add_function(callee);
        self.graph.add_edge(caller_idx, callee_idx, edge);
    }

    /// Get node index for a function (if exists)
    pub fn get_index(&self, id: &FunctionId) -> Option<NodeIndex> {
        self.index.get(id).copied()
    }

    /// Get function by index
    pub fn get_function(&self, idx: NodeIndex) -> Option<&FunctionId> {
        self.graph.node_weight(idx)
    }

    /// All functions (nodes) in the graph
    pub fn functions(&self) -> impl Iterator<Item = &FunctionId> {
        self.graph.node_weights()
    }

    /// Number of functions
    pub fn function_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of call edges
    pub fn call_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Get all calls FROM a function (outgoing edges)
    pub fn calls_from(&self, id: &FunctionId) -> Vec<(&FunctionId, &CallEdge)> {
        let Some(idx) = self.index.get(id) else {
            return vec![];
        };
        self.graph
            .edges(*idx)
            .map(|e| {
                let target = self.graph.node_weight(e.target()).unwrap();
                (target, e.weight())
            })
            .collect()
    }

    /// Get all calls TO a function (incoming edges) - "called by"
    pub fn calls_to(&self, id: &FunctionId) -> Vec<(&FunctionId, &CallEdge)> {
        let Some(idx) = self.index.get(id) else {
            return vec![];
        };
        self.graph
            .edges_directed(*idx, petgraph::Direction::Incoming)
            .map(|e| {
                let source = self.graph.node_weight(e.source()).unwrap();
                (source, e.weight())
            })
            .collect()
    }

    /// Access underlying petgraph for algorithms
    pub fn inner(&self) -> &DiGraph<FunctionId, CallEdge> {
        &self.graph
    }

    /// Find functions by name (may match multiple across files)
    pub fn find_by_name(&self, name: &str) -> Vec<&FunctionId> {
        self.graph
            .node_weights()
            .filter(|f| f.name.as_ref() == name)
            .collect()
    }

    /// Find functions in a specific file
    pub fn functions_in_file(&self, file: &str) -> Vec<&FunctionId> {
        self.graph
            .node_weights()
            .filter(|f| f.file.as_ref() == file)
            .collect()
    }

    /// Merge another graph into this one.
    /// Used when building incrementally from multiple files.
    pub fn merge(&mut self, other: CallGraph) {
        for func in other.graph.node_weights() {
            self.add_function(func.clone());
        }
        for edge in other.graph.edge_references() {
            let source = other.graph.node_weight(edge.source()).unwrap();
            let target = other.graph.node_weight(edge.target()).unwrap();
            self.add_call(source.clone(), target.clone(), edge.weight().clone());
        }
    }

    /// Export edges in the format expected by FocusResolver.expand_via_graph
    /// Returns: Vec<(from_file, from_sym, to_file, to_sym)>
    ///
    /// This allows the call graph to be used for focus expansion:
    /// when user focuses on "foo", we can BFS through call relationships
    /// to find related functions (callers and callees).
    pub fn as_symbol_edges(&self) -> Vec<(Arc<str>, Arc<str>, Arc<str>, Arc<str>)> {
        self.graph
            .edge_indices()
            .filter_map(|e| {
                let (src, tgt) = self.graph.edge_endpoints(e)?;
                let from = self.graph.node_weight(src)?;
                let to = self.graph.node_weight(tgt)?;
                Some((
                    from.file.clone(),
                    from.name.clone(),
                    to.file.clone(),
                    to.name.clone(),
                ))
            })
            .collect()
    }

    /// Compute caller weights: how many unique callers each function has.
    /// Returns map of (file, symbol) -> caller_count.
    ///
    /// Used for boost calculation: heavily-called functions are more important.
    /// This is the "API surface" signal - functions called from many places
    /// are likely public interfaces or critical utilities.
    pub fn caller_weights(&self) -> HashMap<(Arc<str>, Arc<str>), usize> {
        let mut weights = HashMap::new();

        for id in self.graph.node_weights() {
            let key = (id.file.clone(), id.name.clone());
            let caller_count = self.calls_to(id).len();
            if caller_count > 0 {
                weights.insert(key, caller_count);
            }
        }

        weights
    }

    /// Get all function IDs in the graph
    pub fn all_functions(&self) -> impl Iterator<Item = &FunctionId> {
        self.graph.node_weights()
    }
}

impl Default for CallGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_function_idempotent() {
        let mut graph = CallGraph::new();
        let f1 = FunctionId::new("test.py", "foo", 10);
        let idx1 = graph.add_function(f1.clone());
        let idx2 = graph.add_function(f1);
        assert_eq!(idx1, idx2);
        assert_eq!(graph.function_count(), 1);
    }

    #[test]
    fn test_add_call() {
        let mut graph = CallGraph::new();
        let caller = FunctionId::new("test.py", "main", 1);
        let callee = FunctionId::new("test.py", "helper", 10);

        graph.add_call(
            caller.clone(),
            callee.clone(),
            CallEdge::new(1.0, "test", 5),
        );

        assert_eq!(graph.function_count(), 2);
        assert_eq!(graph.call_count(), 1);

        let calls = graph.calls_from(&caller);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0.name.as_ref(), "helper");

        let callers = graph.calls_to(&callee);
        assert_eq!(callers.len(), 1);
        assert_eq!(callers[0].0.name.as_ref(), "main");
    }

    #[test]
    fn test_qualified_name() {
        let method = FunctionId::new("test.py", "process", 10).with_parent("MyClass");
        assert_eq!(method.qualified_name(), "MyClass.process");

        let func = FunctionId::new("test.py", "helper", 20);
        assert_eq!(func.qualified_name(), "helper");
    }
}
