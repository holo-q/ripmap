//! PageRank algorithm for file importance ranking.
//!
//! This module implements PageRank calculation for ranking files by their
//! interconnectedness within the repository. It builds a directed graph where:
//! - Nodes represent files
//! - Edges represent references (file A references a symbol defined in file B)
//! - Edge weights are based on reference counts
//!
//! The PageRank incorporates depth-aware personalization:
//! - Root/shallow files get higher base weight
//! - Vendor/third-party code is heavily penalized
//! - Chat files receive additional boost
//!
//! This depth-aware approach ensures important root files rank high while still
//! allowing deeply nested files to rank well if they're heavily interconnected.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;

use crate::types::{RankingConfig, Tag};

/// PageRank-based file importance calculator.
///
/// Builds a graph of file references and computes importance scores using
/// the PageRank algorithm with depth-aware personalization.
///
/// The graph structure captures how files reference each other:
/// - Definitions create "def" tags in files
/// - References create edges from referencing file to defining file
/// - Multiple references to the same symbol strengthen the edge
///
/// Personalization is depth-aware to bias toward root files while allowing
/// graph structure to override for truly important deep files.
pub struct PageRanker {
    config: RankingConfig,
}

impl PageRanker {
    /// Create a new PageRanker with the given configuration.
    pub fn new(config: RankingConfig) -> Self {
        Self { config }
    }

    /// Compute PageRank scores for all files.
    ///
    /// The algorithm:
    /// 1. Build graph with files as nodes
    /// 2. Add edges based on references (ref file -> def file)
    /// 3. Compute depth-aware personalization weights
    /// 4. Run PageRank with personalization (power iteration)
    /// 5. Return rank scores as dict[rel_fname -> score]
    ///
    /// # Arguments
    /// * `tags_by_file` - Map from absolute file path to its list of tags
    /// * `chat_fnames` - List of chat file absolute paths (for boost)
    ///
    /// # Returns
    /// Map from relative filename to PageRank score (0.0-1.0)
    pub fn compute_ranks(
        &self,
        tags_by_file: &HashMap<String, Vec<Tag>>,
        chat_fnames: &[String],
    ) -> HashMap<String, f64> {
        // Build symbol index: maps symbol names to files that define them
        let defines = self.build_defines_index(tags_by_file);

        // Build the reference graph
        let (graph, node_map, index_map) = self.build_graph(tags_by_file, &defines);

        if graph.node_count() == 0 {
            return HashMap::new();
        }

        // Build depth-aware personalization vector
        let chat_rel_fnames: HashSet<String> = chat_fnames
            .iter()
            .map(|f| self.extract_rel_fname(f))
            .collect();

        let personalization = self.build_personalization(&node_map, &chat_rel_fnames);

        // Run PageRank power iteration
        let ranks = self.pagerank(&graph, &personalization, &index_map);

        // Convert from NodeIndex back to filenames
        let mut result = HashMap::new();
        for (node_idx, rank) in ranks {
            if let Some(rel_fname) = index_map.get(&node_idx) {
                result.insert(rel_fname.clone(), rank);
            }
        }

        result
    }

    /// Build index of symbol definitions: symbol_name -> set of files that define it.
    ///
    /// This enables efficient lookup when building edges: for each reference,
    /// we need to find which file(s) define that symbol.
    fn build_defines_index(
        &self,
        tags_by_file: &HashMap<String, Vec<Tag>>,
    ) -> HashMap<Arc<str>, HashSet<String>> {
        let mut defines: HashMap<Arc<str>, HashSet<String>> = HashMap::new();

        for (fname, tags) in tags_by_file {
            let rel_fname = self.extract_rel_fname(fname);
            for tag in tags {
                if tag.is_def() {
                    defines
                        .entry(Arc::clone(&tag.name))
                        .or_insert_with(HashSet::new)
                        .insert(rel_fname.clone());
                }
            }
        }

        defines
    }

    /// Build the reference graph.
    ///
    /// Returns:
    /// - The petgraph DiGraph
    /// - Map from rel_fname to NodeIndex
    /// - Map from NodeIndex to rel_fname (inverse)
    ///
    /// Graph construction:
    /// - Add all files as nodes
    /// - For each reference tag, add edge from ref_file -> def_file
    /// - Allow multi-edges (multiple refs between same file pair strengthen connection)
    fn build_graph(
        &self,
        tags_by_file: &HashMap<String, Vec<Tag>>,
        defines: &HashMap<Arc<str>, HashSet<String>>,
    ) -> (DiGraph<(), ()>, HashMap<String, NodeIndex>, HashMap<NodeIndex, String>) {
        let mut graph = DiGraph::new();
        let mut node_map: HashMap<String, NodeIndex> = HashMap::new();
        let mut index_map: HashMap<NodeIndex, String> = HashMap::new();

        // Add all files as nodes
        for fname in tags_by_file.keys() {
            let rel_fname = self.extract_rel_fname(fname);
            if !node_map.contains_key(&rel_fname) {
                let idx = graph.add_node(());
                node_map.insert(rel_fname.clone(), idx);
                index_map.insert(idx, rel_fname);
            }
        }

        // Add edges based on references
        // For each reference tag in file A that references symbol S defined in file B:
        // Add edge A -> B
        for (fname, tags) in tags_by_file {
            let ref_fname = self.extract_rel_fname(fname);
            let ref_node = match node_map.get(&ref_fname) {
                Some(n) => *n,
                None => continue,
            };

            for tag in tags {
                if tag.is_ref() {
                    // Find which file(s) define this symbol
                    if let Some(def_fnames) = defines.get(&tag.name) {
                        for def_fname in def_fnames {
                            // Don't create self-loops
                            if def_fname != &ref_fname {
                                if let Some(&def_node) = node_map.get(def_fname) {
                                    // Add edge: referencing file -> defining file
                                    graph.add_edge(ref_node, def_node, ());
                                }
                            }
                        }
                    }
                }
            }
        }

        (graph, node_map, index_map)
    }

    /// Build depth-aware personalization weights for PageRank.
    ///
    /// Personalization biases the random walk toward certain nodes using weights from config:
    /// - Root/shallow files: depth_weight_root (1.0)
    /// - Moderate depth: depth_weight_moderate (0.5)
    /// - Deep files: depth_weight_deep (0.1)
    /// - Vendor/third-party: depth_weight_vendor (0.01)
    /// - Chat files: multiply by pagerank_chat_multiplier (100x)
    ///
    /// Returns a map from rel_fname to personalization weight.
    fn build_personalization(
        &self,
        node_map: &HashMap<String, NodeIndex>,
        chat_fnames: &HashSet<String>,
    ) -> HashMap<NodeIndex, f64> {
        let mut personalization = HashMap::new();

        for (rel_fname, &node_idx) in node_map {
            let weight = self.personalization_weight(rel_fname, chat_fnames);
            personalization.insert(node_idx, weight);
        }

        personalization
    }

    /// Calculate personalization weight for a single file.
    ///
    /// Weight is based on:
    /// - File depth (number of '/' in path)
    /// - Whether it's vendor code
    /// - Whether it's a chat file (current context)
    fn personalization_weight(&self, rel_fname: &str, chat_fnames: &HashSet<String>) -> f64 {
        let depth = rel_fname.matches('/').count();

        // Check if vendor/third-party
        let is_vendor = self
            .config
            .vendor_patterns
            .iter()
            .any(|pattern| rel_fname.contains(pattern.as_str()));

        // Determine base weight by depth and vendor status
        let base_weight = if is_vendor {
            self.config.depth_weight_vendor
        } else if depth <= self.config.depth_threshold_shallow {
            self.config.depth_weight_root
        } else if depth <= self.config.depth_threshold_moderate {
            self.config.depth_weight_moderate
        } else {
            self.config.depth_weight_deep
        };

        // Apply chat file multiplier if applicable
        if chat_fnames.contains(rel_fname) {
            base_weight * self.config.pagerank_chat_multiplier
        } else {
            base_weight
        }
    }

    /// Run PageRank using power iteration.
    ///
    /// PageRank formula with personalization:
    /// ```text
    /// PR(v) = (1-α) * personalization[v] + α * Σ(PR(u) / out_degree[u])
    ///                                          for all u pointing to v
    /// ```
    ///
    /// Where:
    /// - α = damping factor (0.85)
    /// - personalization[v] = normalized depth-aware weight (teleportation distribution)
    ///
    /// The personalization vector determines where random teleportation lands.
    /// High-weight nodes (root files, chat files) get more teleportation probability.
    ///
    /// Iterates until convergence (max change < epsilon) or max iterations reached.
    fn pagerank(
        &self,
        graph: &DiGraph<(), ()>,
        personalization: &HashMap<NodeIndex, f64>,
        _index_map: &HashMap<NodeIndex, String>,
    ) -> HashMap<NodeIndex, f64> {
        let alpha = self.config.pagerank_alpha;
        let epsilon = 1e-8;
        let max_iterations = 100;

        let n = graph.node_count();
        if n == 0 {
            return HashMap::new();
        }

        // Normalize personalization vector to sum to 1.0
        // This represents the probability distribution for random teleportation
        let total_personalization: f64 = personalization.values().sum();
        let normalized_personalization: HashMap<NodeIndex, f64> = personalization
            .iter()
            .map(|(&idx, &weight)| (idx, weight / total_personalization))
            .collect();

        // Initialize ranks uniformly
        let init_rank = 1.0 / n as f64;
        let mut ranks: HashMap<NodeIndex, f64> = graph.node_indices().map(|idx| (idx, init_rank)).collect();
        let mut new_ranks = ranks.clone();

        // Power iteration
        for _iteration in 0..max_iterations {
            // Handle dangling nodes (nodes with no outgoing edges)
            // Their rank needs to be redistributed according to personalization
            let mut dangling_sum = 0.0;
            for node in graph.node_indices() {
                let out_degree = graph.neighbors_directed(node, Direction::Outgoing).count();
                if out_degree == 0 {
                    dangling_sum += ranks[&node];
                }
            }

            for node in graph.node_indices() {
                // Calculate incoming contribution from following edges
                let mut incoming_sum = 0.0;

                // Sum over all incoming edges
                for predecessor in graph.neighbors_directed(node, Direction::Incoming) {
                    let pred_rank = ranks[&predecessor];
                    let out_degree = graph.neighbors_directed(predecessor, Direction::Outgoing).count();

                    if out_degree > 0 {
                        // Each outgoing edge contributes equally (standard PageRank)
                        incoming_sum += pred_rank / out_degree as f64;
                    }
                }

                // Apply PageRank formula with personalization
                // (1-α) portion: teleport according to personalization distribution
                // α portion: follow edges from predecessors
                // Also handle dangling node mass redistribution
                let personalization_value = normalized_personalization.get(&node).copied().unwrap_or(1.0 / n as f64);
                new_ranks.insert(
                    node,
                    (1.0 - alpha) * personalization_value
                        + alpha * incoming_sum
                        + alpha * dangling_sum * personalization_value,  // Redistribute dangling mass
                );
            }

            // Check convergence
            let max_change = ranks
                .iter()
                .map(|(node, &old_rank)| (new_ranks[node] - old_rank).abs())
                .fold(0.0_f64, f64::max);

            if max_change < epsilon {
                break;
            }

            // Swap for next iteration
            std::mem::swap(&mut ranks, &mut new_ranks);
        }

        ranks
    }

    /// Extract relative filename from absolute path.
    ///
    /// This is a simplified version - in production, would use proper
    /// path resolution relative to repo root.
    fn extract_rel_fname(&self, abs_fname: &str) -> String {
        // Simple heuristic: strip common prefixes
        // In practice, would use proper path canonicalization
        abs_fname
            .strip_prefix("/")
            .unwrap_or(abs_fname)
            .to_string()
    }

    /// Compute PageRank scores on a call graph (function-level ranking).
    ///
    /// This uses the CallGraph's precise call relationships for more accurate
    /// function importance scoring than file-level ranking.
    ///
    /// # Arguments
    /// * `call_graph` - The resolved call graph from CallResolver
    /// * `focus_functions` - Optional set of function names to boost (like chat files)
    ///
    /// # Returns
    /// Map from FunctionId to PageRank score
    pub fn compute_function_ranks(
        &self,
        call_graph: &crate::callgraph::CallGraph,
    ) -> HashMap<crate::callgraph::FunctionId, f64> {
        use petgraph::visit::EdgeRef;

        let inner = call_graph.inner();
        let n = inner.node_count();

        if n == 0 {
            return HashMap::new();
        }

        let alpha = self.config.pagerank_alpha;
        let epsilon = 1e-8;
        let max_iterations = 100;

        // Build personalization based on file depth (functions inherit file depth)
        let mut personalization: HashMap<petgraph::graph::NodeIndex, f64> = HashMap::new();
        for node_idx in inner.node_indices() {
            if let Some(func) = inner.node_weight(node_idx) {
                let weight = self.personalization_weight(func.file.as_ref(), &HashSet::new());
                personalization.insert(node_idx, weight);
            }
        }

        // Normalize personalization
        let total: f64 = personalization.values().sum();
        if total > 0.0 {
            for v in personalization.values_mut() {
                *v /= total;
            }
        }

        // Initialize ranks uniformly
        let init_rank = 1.0 / n as f64;
        let mut ranks: HashMap<petgraph::graph::NodeIndex, f64> =
            inner.node_indices().map(|idx| (idx, init_rank)).collect();
        let mut new_ranks = ranks.clone();

        // Power iteration
        for _iteration in 0..max_iterations {
            // Handle dangling nodes
            let mut dangling_sum = 0.0;
            for node in inner.node_indices() {
                let out_degree = inner.edges(node).count();
                if out_degree == 0 {
                    dangling_sum += ranks[&node];
                }
            }

            for node in inner.node_indices() {
                let mut incoming_sum = 0.0;

                // Sum contributions from callers (incoming edges = functions that call us)
                for edge in inner.edges_directed(node, petgraph::Direction::Incoming) {
                    let caller = edge.source();
                    let caller_rank = ranks[&caller];
                    let out_degree = inner.edges(caller).count();

                    if out_degree > 0 {
                        // Weight by edge confidence for more accurate ranking
                        let confidence = edge.weight().confidence;
                        incoming_sum += (caller_rank * confidence) / out_degree as f64;
                    }
                }

                let p_value = personalization.get(&node).copied().unwrap_or(1.0 / n as f64);
                new_ranks.insert(
                    node,
                    (1.0 - alpha) * p_value
                        + alpha * incoming_sum
                        + alpha * dangling_sum * p_value,
                );
            }

            // Check convergence
            let max_change = ranks
                .iter()
                .map(|(node, &old_rank)| (new_ranks[node] - old_rank).abs())
                .fold(0.0_f64, f64::max);

            if max_change < epsilon {
                break;
            }

            std::mem::swap(&mut ranks, &mut new_ranks);
        }

        // Convert to FunctionId keys
        let mut result = HashMap::new();
        for (node_idx, rank) in ranks {
            if let Some(func_id) = inner.node_weight(node_idx) {
                result.insert(func_id.clone(), rank);
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TagKind;

    fn make_tag(rel_fname: &str, name: &str, kind: TagKind) -> Tag {
        Tag {
            rel_fname: Arc::from(rel_fname),
            fname: Arc::from(format!("/{}", rel_fname)),
            line: 1,
            name: Arc::from(name),
            kind,
            node_type: Arc::from("function"),
            parent_name: None,
            parent_line: None,
            signature: None,
            fields: None,
        metadata: None,
        }
    }

    #[test]
    fn test_simple_pagerank() {
        let config = RankingConfig::default();
        let ranker = PageRanker::new(config);

        // Create simple graph:
        // a.rs defines "foo"
        // b.rs references "foo" (b -> a)
        // c.rs references "foo" (c -> a)
        // a.rs should have highest rank (referenced by both)

        let mut tags_by_file = HashMap::new();
        tags_by_file.insert(
            "/a.rs".to_string(),
            vec![make_tag("a.rs", "foo", TagKind::Def)],
        );
        tags_by_file.insert(
            "/b.rs".to_string(),
            vec![make_tag("b.rs", "foo", TagKind::Ref)],
        );
        tags_by_file.insert(
            "/c.rs".to_string(),
            vec![make_tag("c.rs", "foo", TagKind::Ref)],
        );

        let chat_fnames = vec![];
        let ranks = ranker.compute_ranks(&tags_by_file, &chat_fnames);

        // a.rs should have highest rank
        assert!(ranks["a.rs"] > ranks["b.rs"]);
        assert!(ranks["a.rs"] > ranks["c.rs"]);
    }

    #[test]
    fn test_depth_aware_personalization() {
        let config = RankingConfig::default();
        let ranker = PageRanker::new(config.clone());

        let chat_fnames = HashSet::new();

        // Root file (0 slashes)
        let weight_root = ranker.personalization_weight("main.rs", &chat_fnames);
        assert_eq!(weight_root, config.depth_weight_root);

        // Shallow file (1 slash)
        let weight_shallow = ranker.personalization_weight("src/lib.rs", &chat_fnames);
        assert_eq!(weight_shallow, config.depth_weight_root);

        // Deep file (5 slashes)
        let weight_deep = ranker.personalization_weight("src/a/b/c/d/e.rs", &chat_fnames);
        assert_eq!(weight_deep, config.depth_weight_deep);

        // Vendor file
        let weight_vendor = ranker.personalization_weight("vendor/lib.rs", &chat_fnames);
        assert_eq!(weight_vendor, config.depth_weight_vendor);
    }

    #[test]
    fn test_chat_file_boost() {
        let config = RankingConfig::default();
        let ranker = PageRanker::new(config.clone());

        let mut chat_fnames = HashSet::new();
        chat_fnames.insert("main.rs".to_string());

        // Chat file should get multiplier
        let weight_chat = ranker.personalization_weight("main.rs", &chat_fnames);
        assert_eq!(
            weight_chat,
            config.depth_weight_root * config.pagerank_chat_multiplier
        );

        // Non-chat file should not
        let weight_normal = ranker.personalization_weight("other.rs", &chat_fnames);
        assert_eq!(weight_normal, config.depth_weight_root);
    }

    #[test]
    fn test_vendor_patterns() {
        let config = RankingConfig::default();
        let ranker = PageRanker::new(config.clone());

        let chat_fnames = HashSet::new();

        // Test various vendor patterns
        assert_eq!(
            ranker.personalization_weight("node_modules/lib.js", &chat_fnames),
            config.depth_weight_vendor
        );
        assert_eq!(
            ranker.personalization_weight("src/vendor/lib.rs", &chat_fnames),
            config.depth_weight_vendor
        );
        assert_eq!(
            ranker.personalization_weight("third_party/lib.c", &chat_fnames),
            config.depth_weight_vendor
        );
    }

    #[test]
    fn test_empty_graph() {
        let config = RankingConfig::default();
        let ranker = PageRanker::new(config);

        let tags_by_file = HashMap::new();
        let chat_fnames = vec![];
        let ranks = ranker.compute_ranks(&tags_by_file, &chat_fnames);

        assert!(ranks.is_empty());
    }

    #[test]
    fn test_pagerank_convergence() {
        let config = RankingConfig::default();
        let ranker = PageRanker::new(config);

        // Create a chain: a -> b -> c
        let mut tags_by_file = HashMap::new();
        tags_by_file.insert(
            "/a.rs".to_string(),
            vec![make_tag("a.rs", "func_b", TagKind::Ref)],
        );
        tags_by_file.insert(
            "/b.rs".to_string(),
            vec![
                make_tag("b.rs", "func_b", TagKind::Def),
                make_tag("b.rs", "func_c", TagKind::Ref),
            ],
        );
        tags_by_file.insert(
            "/c.rs".to_string(),
            vec![make_tag("c.rs", "func_c", TagKind::Def)],
        );

        let chat_fnames = vec![];
        let ranks = ranker.compute_ranks(&tags_by_file, &chat_fnames);

        // All ranks should sum to approximately 1.0 (standard PageRank property)
        // Our implementation follows the standard formula which preserves this invariant
        let total: f64 = ranks.values().sum();
        assert!((total - 1.0).abs() < 0.01, "Total rank should be close to 1.0, got {}", total);

        // c should have highest rank (pointed to by b)
        // b should have second (pointed to by a)
        // a should have lowest (points but not pointed to)
        assert!(ranks["c.rs"] >= ranks["b.rs"], "c.rs rank {} should be >= b.rs rank {}", ranks["c.rs"], ranks["b.rs"]);
        assert!(ranks["b.rs"] >= ranks["a.rs"], "b.rs rank {} should be >= a.rs rank {}", ranks["b.rs"], ranks["a.rs"]);
    }

    #[test]
    fn test_function_level_pagerank() {
        use crate::callgraph::{CallGraph, CallEdge, FunctionId};

        let config = RankingConfig::default();
        let ranker = PageRanker::new(config);

        // Build a simple call graph:
        // main() -> helper() -> util()
        // main() -> util()
        // util() should rank highest (called by both)
        let mut graph = CallGraph::new();

        let main = FunctionId::new("test.rs", "main", 1);
        let helper = FunctionId::new("test.rs", "helper", 10);
        let util = FunctionId::new("test.rs", "util", 20);

        graph.add_call(
            main.clone(),
            helper.clone(),
            CallEdge::new(0.9, "same_file", 5),
        );
        graph.add_call(
            main.clone(),
            util.clone(),
            CallEdge::new(0.9, "same_file", 6),
        );
        graph.add_call(
            helper.clone(),
            util.clone(),
            CallEdge::new(0.9, "same_file", 15),
        );

        let ranks = ranker.compute_function_ranks(&graph);

        // util should have highest rank (most called)
        assert!(
            ranks[&util] >= ranks[&helper],
            "util rank {} should be >= helper rank {}",
            ranks[&util], ranks[&helper]
        );
        assert!(
            ranks[&util] >= ranks[&main],
            "util rank {} should be >= main rank {}",
            ranks[&util], ranks[&main]
        );
    }
}
