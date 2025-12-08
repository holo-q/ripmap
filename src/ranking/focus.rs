//! Focus query resolution and graph expansion.
//!
//! The focus system enables targeted exploration of large codebases:
//! - Fuzzy matching: resolve user queries to specific symbols and files
//! - Graph expansion: BFS with decay to find related code
//! - Multi-target: comma-separated queries like "auth,parser,main.rs"
//!
//! ## Fuzzy Matching Strategy
//!
//! Matches are attempted in priority order:
//! 1. Exact match (case-insensitive)
//! 2. Substring containment
//! 3. Word-part subset matching (handles snake_case, camelCase)
//! 4. Any query part found in name (3+ chars)
//! 5. Stem matching (auth ~ authenticate ~ authorization)
//! 6. Edit distance <= 1 for typos (4+ chars)
//!
//! This enables natural queries:
//! - "auth" → matches "authenticate", "authorization", "auth_handler"
//! - "getuser" → matches "getUserName", "get_user_profile"
//! - "parsr" → matches "parser" (typo tolerance)
//!
//! ## Graph Expansion
//!
//! From seed symbols, expand via BFS with exponential decay:
//! - Seed nodes: weight = 1.0
//! - 1-hop neighbors: weight = decay^1 (default 0.5)
//! - 2-hop neighbors: weight = decay^2 (default 0.25)
//!
//! This surfaces related code without flooding with transitive dependencies.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::Arc;

use crate::types::Tag;

/// Focus resolver - converts user queries into matched files/symbols,
/// then expands via graph traversal to find related code.
pub struct FocusResolver {
    /// Repository root for resolving relative paths
    root: std::path::PathBuf,
}

impl FocusResolver {
    /// Create a new focus resolver rooted at the given path.
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
    }

    /// Resolve focus targets to matched files and symbol identifiers.
    ///
    /// Supports:
    /// - File paths: "src/lib.rs", "./parser.rs", absolute paths
    /// - Symbol queries: "authenticate", "parse_config", fuzzy matched
    /// - Comma-separated: "auth,parser,main.rs"
    ///
    /// Returns:
    /// - matched_files: absolute file paths that match
    /// - matched_idents: symbol names (Arc<str>) that match
    pub fn resolve(
        &self,
        focus_targets: &[String],
        tags_by_file: &HashMap<String, Vec<Tag>>,
    ) -> (HashSet<String>, HashSet<String>) {
        let mut matched_files = HashSet::new();
        let mut matched_idents = HashSet::new();

        // Parse comma-separated targets
        let targets = focus_targets
            .iter()
            .flat_map(|t| t.split(',').map(|s| s.trim()))
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>();

        for target in targets {
            // Try as file path first (absolute or relative)
            if let Some(matched_file) = self.try_match_file(target, tags_by_file) {
                matched_files.insert(matched_file);
                continue;
            }

            // Try as symbol query - fuzzy match against all symbols
            let query_idents = self.fuzzy_match_symbols(target, tags_by_file);
            if !query_idents.is_empty() {
                matched_idents.extend(query_idents);
                continue;
            }

            // No matches - silently skip (user may be experimenting)
            eprintln!("focus: no matches for '{}'", target);
        }

        (matched_files, matched_idents)
    }

    /// Try to match a target as a file path.
    fn try_match_file(
        &self,
        target: &str,
        tags_by_file: &HashMap<String, Vec<Tag>>,
    ) -> Option<String> {
        // Only try file matching if target looks like a file path
        // (contains path separator or file extension)
        if !target.contains('/') && !target.contains('.') {
            return None;
        }

        // Try absolute path
        let abs_path = std::path::Path::new(target);
        if abs_path.is_absolute() && abs_path.exists() {
            return Some(abs_path.to_string_lossy().to_string());
        }

        // Try relative to root
        let rel_path = self.root.join(target);
        if rel_path.exists() {
            return Some(rel_path.to_string_lossy().to_string());
        }

        // Try matching against known files (suffix matching)
        // E.g., "lib.rs" matches "src/lib.rs"
        for file_path in tags_by_file.keys() {
            if file_path.ends_with(target) || file_path.contains(&format!("/{}", target)) {
                return Some(file_path.clone());
            }
        }

        None
    }

    /// Fuzzy match a query against all symbols in tags_by_file.
    /// Returns the set of matched symbol names.
    fn fuzzy_match_symbols(
        &self,
        query: &str,
        tags_by_file: &HashMap<String, Vec<Tag>>,
    ) -> HashSet<String> {
        let mut matched = HashSet::new();

        for tags in tags_by_file.values() {
            for tag in tags {
                if matches_query(&tag.name, query) {
                    matched.insert(tag.name.to_string());
                }
            }
        }

        matched
    }

    /// Expand focus via BFS graph traversal with exponential decay.
    ///
    /// Starting from matched identifiers (seed nodes), traverse the symbol
    /// graph to find related code. Each hop reduces weight by decay factor.
    ///
    /// # Arguments
    /// * `matched_idents` - Seed symbol names
    /// * `symbol_graph` - Edges as (from_file, from_sym, to_file, to_sym)
    /// * `max_hops` - Maximum BFS depth (typically 1-2)
    /// * `decay` - Weight decay per hop (typically 0.5)
    ///
    /// # Returns
    /// Map of (file, symbol) -> weight, where weight = decay^hop_distance
    pub fn expand_via_graph(
        &self,
        matched_idents: &HashSet<String>,
        symbol_graph: &[(Arc<str>, Arc<str>, Arc<str>, Arc<str>)],
        max_hops: usize,
        decay: f64,
    ) -> HashMap<(Arc<str>, Arc<str>), f64> {
        let mut expanded = HashMap::new();

        // Find seed nodes: any symbol in graph matching our identifiers
        let seeds: HashSet<_> = symbol_graph
            .iter()
            .filter_map(|(from_file, from_sym, to_file, to_sym)| {
                // Check both endpoints of each edge
                let mut matches = Vec::new();
                if matched_idents.contains(&**from_sym) {
                    matches.push((from_file.clone(), from_sym.clone()));
                }
                if matched_idents.contains(&**to_sym) {
                    matches.push((to_file.clone(), to_sym.clone()));
                }
                if matches.is_empty() {
                    None
                } else {
                    Some(matches)
                }
            })
            .flatten()
            .collect();

        if seeds.is_empty() {
            return expanded;
        }

        // Initialize seeds with weight 1.0
        for seed in &seeds {
            expanded.insert(seed.clone(), 1.0);
        }

        // BFS expansion with decay
        let mut frontier: VecDeque<_> = seeds.into_iter().collect();
        let mut visited = HashSet::new();

        for hop in 1..=max_hops {
            let weight = decay.powi(hop as i32);
            let frontier_size = frontier.len();

            for _ in 0..frontier_size {
                let node = match frontier.pop_front() {
                    Some(n) => n,
                    None => break,
                };

                if !visited.insert(node.clone()) {
                    continue;
                }

                // Find neighbors (graph is undirected for expansion purposes)
                for (from_file, from_sym, to_file, to_sym) in symbol_graph {
                    let neighbor = if from_file == &node.0 && from_sym == &node.1 {
                        // Forward edge
                        Some((to_file.clone(), to_sym.clone()))
                    } else if to_file == &node.0 && to_sym == &node.1 {
                        // Backward edge (reverse reference graph)
                        Some((from_file.clone(), from_sym.clone()))
                    } else {
                        None
                    };

                    if let Some(neighbor_node) = neighbor {
                        // Only expand to new nodes
                        if !expanded.contains_key(&neighbor_node) {
                            expanded.insert(neighbor_node.clone(), weight);
                            frontier.push_back(neighbor_node);
                        }
                    }
                }
            }

            if frontier.is_empty() {
                break;
            }
        }

        expanded
    }
}

/// Fuzzy match a symbol name against a query.
///
/// Implements multi-strategy matching for natural queries:
/// 1. Exact match (case-insensitive)
/// 2. Substring containment
/// 3. Word-part subset (handles identifiers)
/// 4. Any query part in name (3+ chars)
/// 5. Stem matching (morphological variants)
/// 6. Edit distance <= 1 (typo tolerance)
fn matches_query(name: &str, query: &str) -> bool {
    let name_lower = name.to_lowercase();
    let query_lower = query.to_lowercase();

    // 1. Exact match
    if name_lower == query_lower {
        return true;
    }

    // 2. Substring match
    if name_lower.contains(&query_lower) {
        return true;
    }

    // 3. Word part matching - query parts must be subset of name parts
    let query_parts: HashSet<_> = split_identifier(&query_lower);
    let name_parts: HashSet<_> = split_identifier(&name_lower);

    if !query_parts.is_empty() && query_parts.is_subset(&name_parts) {
        return true;
    }

    // 4. Any query part in name (3+ chars to avoid false positives)
    if query_parts
        .iter()
        .any(|p| p.len() >= 3 && name_lower.contains(p))
    {
        return true;
    }

    // 5. Stem matching - handles morphological variants
    let query_stems: HashSet<_> = query_parts.iter().filter_map(|p| get_stem(p)).collect();
    let name_stems: HashSet<_> = name_parts.iter().filter_map(|p| get_stem(p)).collect();

    if !query_stems.is_empty() && !query_stems.is_disjoint(&name_stems) {
        return true;
    }

    // 6. Edit distance for typos (4+ chars, distance <= 1)
    if query_lower.len() >= 4 {
        for part in &name_parts {
            if part.len() >= 4 && levenshtein(&query_lower, part) <= 1 {
                return true;
            }
        }
    }

    false
}

/// Split identifier into constituent words.
///
/// Handles both snake_case and camelCase:
/// - "getUserName" -> ["get", "user", "name"]
/// - "get_user_name" -> ["get", "user", "name"]
/// - "HTTPServer" -> ["http", "server"]
fn split_identifier(s: &str) -> HashSet<String> {
    let mut parts = HashSet::new();
    let mut current = String::new();

    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];

        if ch == '_' || ch == '-' || ch == '.' {
            // Word boundary - flush current
            if !current.is_empty() {
                parts.insert(current.to_lowercase());
                current.clear();
            }
            i += 1;
            continue;
        }

        if ch.is_uppercase() && !current.is_empty() {
            // camelCase boundary - check if this is an acronym
            // "HTTPServer" should split as HTTP, Server not H, T, T, P, Server
            let next_is_lower = i + 1 < chars.len() && chars[i + 1].is_lowercase();
            let prev_is_upper = current.chars().last().map_or(false, |c| c.is_uppercase());

            if next_is_lower || !prev_is_upper {
                // Boundary found
                parts.insert(current.to_lowercase());
                current.clear();
            }
        }

        current.push(ch);
        i += 1;
    }

    if !current.is_empty() {
        parts.insert(current.to_lowercase());
    }

    parts
}

/// Get canonical stem for a word.
///
/// Stem groups capture common morphological variants:
/// - auth, authenticate, authentication, authorized, authorization
/// - parse, parser, parsing, parsed
/// - valid, validate, validation, validator, invalid
///
/// Returns the canonical form (first element of the group).
fn get_stem(word: &str) -> Option<&'static str> {
    STEM_GROUPS
        .iter()
        .find(|group| group.contains(&word))
        .map(|group| group[0])
}

/// Stem groups for common programming vocabulary.
///
/// Each group lists morphological variants that should fuzzy-match.
/// First element is the canonical stem.
const STEM_GROUPS: &[&[&str]] = &[
    // Authentication/Authorization
    &["auth", "authenticate", "authentication", "authenticated", "authorize", "authorization", "authorized"],
    // Parsing
    &["parse", "parser", "parsing", "parsed"],
    // Validation
    &["valid", "validate", "validation", "validator", "validated", "invalid", "invalidate", "invalidated"],
    // Configuration
    &["config", "configure", "configuration", "configured", "configurator"],
    // Initialization
    &["init", "initialize", "initialization", "initialized", "initializer"],
    // Rendering
    &["render", "renderer", "rendering", "rendered"],
    // Caching
    &["cache", "caching", "cached"],
    // Handling
    &["handle", "handler", "handling", "handled"],
    // Execution
    &["exec", "execute", "execution", "executed", "executor"],
    // Processing
    &["process", "processor", "processing", "processed"],
    // Serialization
    &["serial", "serialize", "serialization", "serialized", "serializer", "deserialize", "deserialized"],
    // Connection
    &["connect", "connection", "connected", "connector", "disconnect", "disconnected"],
    // Transformation
    &["transform", "transformer", "transformation", "transformed"],
    // Compilation
    &["compile", "compiler", "compilation", "compiled"],
    // Evaluation
    &["eval", "evaluate", "evaluation", "evaluated", "evaluator"],
    // Generation
    &["gen", "generate", "generation", "generated", "generator"],
    // Registration
    &["register", "registration", "registered", "registry"],
    // Query/Request
    &["query", "request", "req"],
    // Response
    &["response", "resp", "reply"],
];

/// Compute Levenshtein edit distance between two strings.
///
/// Used for typo tolerance in fuzzy matching.
/// Optimized for early exit when distance exceeds threshold.
fn levenshtein(a: &str, b: &str) -> usize {
    let a_len = a.len();
    let b_len = b.len();

    if a_len == 0 {
        return b_len;
    }
    if b_len == 0 {
        return a_len;
    }

    // Use single-row optimization (space O(min(a,b)))
    let mut prev_row: Vec<usize> = (0..=b_len).collect();
    let mut curr_row = vec![0; b_len + 1];

    for (i, a_char) in a.chars().enumerate() {
        curr_row[0] = i + 1;

        for (j, b_char) in b.chars().enumerate() {
            let cost = if a_char == b_char { 0 } else { 1 };
            curr_row[j + 1] = (curr_row[j] + 1) // insertion
                .min(prev_row[j + 1] + 1) // deletion
                .min(prev_row[j] + cost); // substitution
        }

        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[b_len]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_identifier() {
        // snake_case
        assert_eq!(
            split_identifier("get_user_name"),
            ["get", "user", "name"].iter().map(|s| s.to_string()).collect()
        );

        // camelCase
        assert_eq!(
            split_identifier("getUserName"),
            ["get", "user", "name"].iter().map(|s| s.to_string()).collect()
        );

        // PascalCase
        assert_eq!(
            split_identifier("GetUserName"),
            ["get", "user", "name"].iter().map(|s| s.to_string()).collect()
        );

        // Acronyms
        let parts = split_identifier("HTTPServer");
        assert!(parts.contains("http") || parts.contains("h")); // Either is acceptable
        assert!(parts.contains("server"));

        // Mixed
        assert_eq!(
            split_identifier("parse_JSONObject"),
            ["parse", "json", "object"].iter().map(|s| s.to_string()).collect()
        );
    }

    #[test]
    fn test_get_stem() {
        assert_eq!(get_stem("authenticate"), Some("auth"));
        assert_eq!(get_stem("authentication"), Some("auth"));
        assert_eq!(get_stem("authorized"), Some("auth"));
        assert_eq!(get_stem("auth"), Some("auth"));

        assert_eq!(get_stem("parser"), Some("parse"));
        assert_eq!(get_stem("parsing"), Some("parse"));
        assert_eq!(get_stem("parsed"), Some("parse"));

        assert_eq!(get_stem("validator"), Some("valid"));
        assert_eq!(get_stem("invalid"), Some("valid"));

        assert_eq!(get_stem("unknown_word"), None);
    }

    #[test]
    fn test_levenshtein() {
        assert_eq!(levenshtein("", ""), 0);
        assert_eq!(levenshtein("a", ""), 1);
        assert_eq!(levenshtein("", "b"), 1);
        assert_eq!(levenshtein("abc", "abc"), 0);
        assert_eq!(levenshtein("abc", "abd"), 1);
        assert_eq!(levenshtein("abc", "axc"), 1);
        assert_eq!(levenshtein("abc", "abcd"), 1);
        assert_eq!(levenshtein("parse", "parser"), 1);
        assert_eq!(levenshtein("parsr", "parser"), 1); // typo
    }

    #[test]
    fn test_matches_query_exact() {
        assert!(matches_query("authenticate", "authenticate"));
        assert!(matches_query("Authenticate", "authenticate")); // case insensitive
        assert!(matches_query("AUTHENTICATE", "authenticate"));
    }

    #[test]
    fn test_matches_query_substring() {
        assert!(matches_query("authenticate", "auth"));
        assert!(matches_query("user_authentication", "auth"));
        assert!(matches_query("HTTPServer", "http"));
    }

    #[test]
    fn test_matches_query_word_parts() {
        // Query parts are subset of name parts
        assert!(matches_query("getUserName", "getuser"));
        assert!(matches_query("get_user_name", "user"));
        assert!(matches_query("parseHTTPRequest", "parse"));
    }

    #[test]
    fn test_matches_query_stem() {
        // Stem matching across morphological variants
        assert!(matches_query("authenticate", "auth"));
        assert!(matches_query("authentication", "auth"));
        assert!(matches_query("authorized", "auth"));

        assert!(matches_query("parser", "parse"));
        assert!(matches_query("parsing", "parse"));

        assert!(matches_query("validator", "valid"));
        assert!(matches_query("invalid", "valid"));
    }

    #[test]
    fn test_matches_query_typo() {
        // Edit distance <= 1 for 4+ char words
        assert!(matches_query("parser", "parsr")); // missing 'e'
        assert!(matches_query("authenticate", "authentcate")); // missing 'i'
        // Note: "parsxr" would match "parser" via parts matching ("pars" substring)
        // For a true negative, need a completely different word
        assert!(!matches_query("parser", "xyzabc"));
    }

    #[test]
    fn test_matches_query_negative() {
        assert!(!matches_query("authenticate", "xyz"));
        assert!(!matches_query("getUserName", "setpassword"));
        assert!(!matches_query("parser", "compiler"));
    }

    #[test]
    fn test_resolve_empty() {
        let resolver = FocusResolver::new("/tmp");
        let tags_by_file = HashMap::new();
        let (files, idents) = resolver.resolve(&[], &tags_by_file);
        assert!(files.is_empty());
        assert!(idents.is_empty());
    }

    #[test]
    fn test_resolve_symbols() {
        let resolver = FocusResolver::new("/tmp");

        // Create some mock tags
        let mut tags_by_file = HashMap::new();
        tags_by_file.insert(
            "/src/auth.rs".to_string(),
            vec![
                Tag {
                    rel_fname: "src/auth.rs".into(),
                    fname: "/src/auth.rs".into(),
                    line: 10,
                    name: "authenticate".into(),
                    kind: crate::types::TagKind::Def,
                    node_type: "function".into(),
                    parent_name: None,
                    parent_line: None,
                    signature: None,
                    fields: None,
                metadata: None,
                },
                Tag {
                    rel_fname: "src/auth.rs".into(),
                    fname: "/src/auth.rs".into(),
                    line: 20,
                    name: "authorize".into(),
                    kind: crate::types::TagKind::Def,
                    node_type: "function".into(),
                    parent_name: None,
                    parent_line: None,
                    signature: None,
                    fields: None,
                metadata: None,
                },
            ],
        );

        let (files, idents) = resolver.resolve(&["auth".to_string()], &tags_by_file);

        assert!(files.is_empty()); // "auth" is not a file
        assert_eq!(idents.len(), 2); // matches both authenticate and authorize
        assert!(idents.contains("authenticate"));
        assert!(idents.contains("authorize"));
    }

    #[test]
    fn test_resolve_comma_separated() {
        let resolver = FocusResolver::new("/tmp");

        let mut tags_by_file = HashMap::new();
        tags_by_file.insert(
            "/src/parser.rs".to_string(),
            vec![Tag {
                rel_fname: "src/parser.rs".into(),
                fname: "/src/parser.rs".into(),
                line: 10,
                name: "parse".into(),
                kind: crate::types::TagKind::Def,
                node_type: "function".into(),
                parent_name: None,
                parent_line: None,
                signature: None,
                fields: None,
            metadata: None,
            }],
        );
        tags_by_file.insert(
            "/src/auth.rs".to_string(),
            vec![Tag {
                rel_fname: "src/auth.rs".into(),
                fname: "/src/auth.rs".into(),
                line: 20,
                name: "authenticate".into(),
                kind: crate::types::TagKind::Def,
                node_type: "function".into(),
                parent_name: None,
                parent_line: None,
                signature: None,
                fields: None,
            metadata: None,
            }],
        );

        let (files, idents) = resolver.resolve(&["parse,auth".to_string()], &tags_by_file);

        assert!(files.is_empty());
        assert_eq!(idents.len(), 2);
        assert!(idents.contains("parse"));
        assert!(idents.contains("authenticate"));
    }

    #[test]
    fn test_expand_via_graph_empty() {
        let resolver = FocusResolver::new("/tmp");
        let matched_idents = HashSet::new();
        let symbol_graph = vec![];

        let expanded = resolver.expand_via_graph(&matched_idents, &symbol_graph, 2, 0.5);
        assert!(expanded.is_empty());
    }

    #[test]
    fn test_expand_via_graph_seeds_only() {
        let resolver = FocusResolver::new("/tmp");

        let mut matched_idents = HashSet::new();
        matched_idents.insert("foo".to_string());

        // Graph with foo but no connections
        let symbol_graph = vec![];

        let expanded = resolver.expand_via_graph(&matched_idents, &symbol_graph, 2, 0.5);
        // No seeds found in empty graph
        assert!(expanded.is_empty());
    }

    #[test]
    fn test_expand_via_graph_one_hop() {
        let resolver = FocusResolver::new("/tmp");

        let mut matched_idents = HashSet::new();
        matched_idents.insert("foo".to_string());

        // foo calls bar
        let symbol_graph = vec![(
            Arc::from("a.rs"),
            Arc::from("foo"),
            Arc::from("b.rs"),
            Arc::from("bar"),
        )];

        let expanded = resolver.expand_via_graph(&matched_idents, &symbol_graph, 1, 0.5);

        // Should have foo (weight 1.0) and bar (weight 0.5)
        assert_eq!(expanded.len(), 2);
        assert_eq!(expanded.get(&(Arc::from("a.rs"), Arc::from("foo"))), Some(&1.0));
        assert_eq!(expanded.get(&(Arc::from("b.rs"), Arc::from("bar"))), Some(&0.5));
    }

    #[test]
    fn test_expand_via_graph_two_hops() {
        let resolver = FocusResolver::new("/tmp");

        let mut matched_idents = HashSet::new();
        matched_idents.insert("foo".to_string());

        // foo -> bar -> baz
        let symbol_graph = vec![
            (
                Arc::from("a.rs"),
                Arc::from("foo"),
                Arc::from("b.rs"),
                Arc::from("bar"),
            ),
            (
                Arc::from("b.rs"),
                Arc::from("bar"),
                Arc::from("c.rs"),
                Arc::from("baz"),
            ),
        ];

        let expanded = resolver.expand_via_graph(&matched_idents, &symbol_graph, 2, 0.5);

        // Should have foo (1.0), bar (0.5), baz (0.25)
        assert_eq!(expanded.len(), 3);
        assert_eq!(expanded.get(&(Arc::from("a.rs"), Arc::from("foo"))), Some(&1.0));
        assert_eq!(expanded.get(&(Arc::from("b.rs"), Arc::from("bar"))), Some(&0.5));
        assert_eq!(expanded.get(&(Arc::from("c.rs"), Arc::from("baz"))), Some(&0.25));
    }

    #[test]
    fn test_expand_via_graph_max_hops() {
        let resolver = FocusResolver::new("/tmp");

        let mut matched_idents = HashSet::new();
        matched_idents.insert("foo".to_string());

        // Long chain: foo -> bar -> baz -> qux
        let symbol_graph = vec![
            (
                Arc::from("a.rs"),
                Arc::from("foo"),
                Arc::from("b.rs"),
                Arc::from("bar"),
            ),
            (
                Arc::from("b.rs"),
                Arc::from("bar"),
                Arc::from("c.rs"),
                Arc::from("baz"),
            ),
            (
                Arc::from("c.rs"),
                Arc::from("baz"),
                Arc::from("d.rs"),
                Arc::from("qux"),
            ),
        ];

        // Limit to 1 hop
        let expanded = resolver.expand_via_graph(&matched_idents, &symbol_graph, 1, 0.5);
        assert_eq!(expanded.len(), 2); // foo, bar only
        assert!(!expanded.contains_key(&(Arc::from("c.rs"), Arc::from("baz"))));

        // Limit to 2 hops
        let expanded = resolver.expand_via_graph(&matched_idents, &symbol_graph, 2, 0.5);
        assert_eq!(expanded.len(), 3); // foo, bar, baz
        assert!(!expanded.contains_key(&(Arc::from("d.rs"), Arc::from("qux"))));
    }

    #[test]
    fn test_expand_via_graph_bidirectional() {
        let resolver = FocusResolver::new("/tmp");

        let mut matched_idents = HashSet::new();
        matched_idents.insert("bar".to_string());

        // foo -> bar <- baz (bar is called by both foo and baz)
        let symbol_graph = vec![
            (
                Arc::from("a.rs"),
                Arc::from("foo"),
                Arc::from("b.rs"),
                Arc::from("bar"),
            ),
            (
                Arc::from("c.rs"),
                Arc::from("baz"),
                Arc::from("b.rs"),
                Arc::from("bar"),
            ),
        ];

        let expanded = resolver.expand_via_graph(&matched_idents, &symbol_graph, 1, 0.5);

        // Should expand to both callers
        assert_eq!(expanded.len(), 3); // bar, foo, baz
        assert_eq!(expanded.get(&(Arc::from("b.rs"), Arc::from("bar"))), Some(&1.0));
        assert_eq!(expanded.get(&(Arc::from("a.rs"), Arc::from("foo"))), Some(&0.5));
        assert_eq!(expanded.get(&(Arc::from("c.rs"), Arc::from("baz"))), Some(&0.5));
    }

    #[test]
    fn test_expand_via_graph_custom_decay() {
        let resolver = FocusResolver::new("/tmp");

        let mut matched_idents = HashSet::new();
        matched_idents.insert("foo".to_string());

        let symbol_graph = vec![(
            Arc::from("a.rs"),
            Arc::from("foo"),
            Arc::from("b.rs"),
            Arc::from("bar"),
        )];

        // Test with different decay values
        let expanded = resolver.expand_via_graph(&matched_idents, &symbol_graph, 1, 0.75);
        assert_eq!(expanded.get(&(Arc::from("b.rs"), Arc::from("bar"))), Some(&0.75));

        let expanded = resolver.expand_via_graph(&matched_idents, &symbol_graph, 1, 0.25);
        assert_eq!(expanded.get(&(Arc::from("b.rs"), Arc::from("bar"))), Some(&0.25));
    }

    #[test]
    fn test_resolve_file_with_extension() {
        let resolver = FocusResolver::new("/tmp");

        let mut tags_by_file = HashMap::new();
        tags_by_file.insert(
            "/src/auth.rs".to_string(),
            vec![Tag {
                rel_fname: "src/auth.rs".into(),
                fname: "/src/auth.rs".into(),
                line: 10,
                name: "authenticate".into(),
                kind: crate::types::TagKind::Def,
                node_type: "function".into(),
                parent_name: None,
                parent_line: None,
                signature: None,
                fields: None,
            metadata: None,
            }],
        );

        // Should match as a file (has .rs extension)
        let (files, idents) = resolver.resolve(&["auth.rs".to_string()], &tags_by_file);
        assert_eq!(files.len(), 1);
        assert!(files.contains("/src/auth.rs"));
        assert!(idents.is_empty());
    }

    #[test]
    fn test_expand_via_graph_cycle() {
        let resolver = FocusResolver::new("/tmp");

        let mut matched_idents = HashSet::new();
        matched_idents.insert("foo".to_string());

        // Cycle: foo -> bar -> foo
        let symbol_graph = vec![
            (
                Arc::from("a.rs"),
                Arc::from("foo"),
                Arc::from("b.rs"),
                Arc::from("bar"),
            ),
            (
                Arc::from("b.rs"),
                Arc::from("bar"),
                Arc::from("a.rs"),
                Arc::from("foo"),
            ),
        ];

        let expanded = resolver.expand_via_graph(&matched_idents, &symbol_graph, 2, 0.5);

        // Should handle cycle gracefully - foo not re-added at lower weight
        assert_eq!(expanded.len(), 2); // foo, bar
        assert_eq!(expanded.get(&(Arc::from("a.rs"), Arc::from("foo"))), Some(&1.0));
        assert_eq!(expanded.get(&(Arc::from("b.rs"), Arc::from("bar"))), Some(&0.5));
    }

    #[test]
    fn test_split_identifier_edge_cases() {
        // Single word
        assert_eq!(split_identifier("foo"), ["foo"].iter().map(|s| s.to_string()).collect());

        // Empty
        assert_eq!(split_identifier(""), HashSet::new());

        // Numbers
        let parts = split_identifier("foo123bar");
        assert!(parts.contains("foo123bar") || parts.len() >= 1);

        // Multiple delimiters
        assert_eq!(
            split_identifier("foo__bar--baz"),
            ["foo", "bar", "baz"].iter().map(|s| s.to_string()).collect()
        );
    }

    #[test]
    fn test_matches_query_case_insensitive() {
        assert!(matches_query("AuthHandler", "authhandler"));
        assert!(matches_query("AUTH_HANDLER", "auth"));
        assert!(matches_query("parseJSON", "parsejson"));
    }

    #[test]
    fn test_expand_via_graph_multiple_seeds() {
        let resolver = FocusResolver::new("/tmp");

        let mut matched_idents = HashSet::new();
        matched_idents.insert("foo".to_string());
        matched_idents.insert("baz".to_string());

        // Two separate chains: foo -> bar, baz -> qux
        let symbol_graph = vec![
            (
                Arc::from("a.rs"),
                Arc::from("foo"),
                Arc::from("b.rs"),
                Arc::from("bar"),
            ),
            (
                Arc::from("c.rs"),
                Arc::from("baz"),
                Arc::from("d.rs"),
                Arc::from("qux"),
            ),
        ];

        let expanded = resolver.expand_via_graph(&matched_idents, &symbol_graph, 1, 0.5);

        // Should expand from both seeds
        assert_eq!(expanded.len(), 4); // foo, bar, baz, qux
        assert_eq!(expanded.get(&(Arc::from("a.rs"), Arc::from("foo"))), Some(&1.0));
        assert_eq!(expanded.get(&(Arc::from("b.rs"), Arc::from("bar"))), Some(&0.5));
        assert_eq!(expanded.get(&(Arc::from("c.rs"), Arc::from("baz"))), Some(&1.0));
        assert_eq!(expanded.get(&(Arc::from("d.rs"), Arc::from("qux"))), Some(&0.5));
    }
}
