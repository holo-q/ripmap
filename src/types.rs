//! Core types for ripmap - the ultra-fast codebase cartographer.
//!
//! This module mirrors the Python grepmap types but optimized for Rust's
//! zero-cost abstractions. Key design decisions:
//! - `Cow<str>` for zero-copy string handling from memory-mapped files
//! - `Arc` for shared ownership of interned strings
//! - Frozen/immutable by default (like Python's frozen dataclasses)

use std::sync::Arc;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Serde serialization helpers for Arc<str> fields
mod arc_str_serde {
    use super::*;

    pub fn serialize<S>(arc: &Arc<str>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(arc.as_ref())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Arc<str>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(s.into())
    }

    pub fn serialize_opt<S>(arc: &Option<Arc<str>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match arc {
            Some(s) => serializer.serialize_some(s.as_ref()),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize_opt<'de, D>(deserializer: D) -> Result<Option<Arc<str>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt: Option<String> = Option::deserialize(deserializer)?;
        Ok(opt.map(|s| s.into()))
    }
}

/// Detail level for rendering - controls how much information to show.
/// Higher levels = more tokens but more context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum DetailLevel {
    /// Names only: "connect, disconnect, validate"
    Low = 1,
    /// Names + simplified types: "connect(host, port) -> bool"
    Medium = 2,
    /// Full signatures: "connect(host: &str, port: u16) -> Result<Connection>"
    High = 3,
}

impl DetailLevel {
    pub fn value(self) -> u8 {
        self as u8
    }
}

/// Function/method signature information.
/// Extracted from AST to enable detail-level rendering.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SignatureInfo {
    /// Parameters as (name, optional_type) pairs
    pub parameters: Vec<(String, Option<String>)>,
    /// Return type annotation if present
    pub return_type: Option<String>,
    /// Decorators like @property, @staticmethod
    pub decorators: Vec<String>,
    /// Raw signature text for markdown previews
    pub raw: Option<String>,
}

impl SignatureInfo {
    pub fn new() -> Self {
        Self {
            parameters: Vec::new(),
            return_type: None,
            decorators: Vec::new(),
            raw: None,
        }
    }

    /// Render at a given detail level
    pub fn render(&self, detail: DetailLevel) -> String {
        match detail {
            DetailLevel::Low => {
                // Just parameter names, no types
                let names: Vec<_> = self.parameters.iter().map(|(n, _)| n.as_str()).collect();
                format!("({})", names.join(", "))
            }
            DetailLevel::Medium => {
                // Names with simplified types (strip generics)
                let parts: Vec<_> = self
                    .parameters
                    .iter()
                    .map(|(name, typ)| {
                        if let Some(t) = typ {
                            let simple = simplify_type(t);
                            format!("{}: {}", name, simple)
                        } else {
                            name.clone()
                        }
                    })
                    .collect();
                let ret = self.return_type.as_ref().map(|t| format!(" -> {}", simplify_type(t))).unwrap_or_default();
                format!("({}){}", parts.join(", "), ret)
            }
            DetailLevel::High => {
                // Full signature
                let parts: Vec<_> = self
                    .parameters
                    .iter()
                    .map(|(name, typ)| {
                        if let Some(t) = typ {
                            format!("{}: {}", name, t)
                        } else {
                            name.clone()
                        }
                    })
                    .collect();
                let ret = self.return_type.as_ref().map(|t| format!(" -> {}", t)).unwrap_or_default();
                format!("({}){}", parts.join(", "), ret)
            }
        }
    }
}

impl Default for SignatureInfo {
    fn default() -> Self {
        Self::new()
    }
}

/// Simplify a type for MEDIUM detail level.
/// Strips generics and long paths, keeps core type name.
fn simplify_type(t: &str) -> &str {
    // Strip generic params: "Dict[str, int]" -> "Dict"
    if let Some(pos) = t.find('[') {
        return &t[..pos];
    }
    if let Some(pos) = t.find('<') {
        return &t[..pos];
    }
    // Strip module path: "std::collections::HashMap" -> "HashMap"
    if let Some(pos) = t.rfind("::") {
        return &t[pos + 2..];
    }
    if let Some(pos) = t.rfind('.') {
        return &t[pos + 1..];
    }
    t
}

/// Class/struct field information.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FieldInfo {
    pub name: String,
    pub type_annotation: Option<String>,
    pub default_value: Option<String>,
}

impl FieldInfo {
    pub fn render(&self, detail: DetailLevel) -> String {
        match detail {
            DetailLevel::Low => self.name.clone(),
            DetailLevel::Medium => {
                if let Some(t) = &self.type_annotation {
                    format!("{}: {}", self.name, simplify_type(t))
                } else {
                    self.name.clone()
                }
            }
            DetailLevel::High => {
                let typ = self.type_annotation.as_deref().unwrap_or("?");
                if let Some(default) = &self.default_value {
                    format!("{}: {} = {}", self.name, typ, default)
                } else {
                    format!("{}: {}", self.name, typ)
                }
            }
        }
    }
}

/// The fundamental unit of code structure - a symbol tag.
/// Represents either a definition ("def") or reference ("ref").
///
/// This is the atom from which all ranking and rendering is built.
/// Frozen/immutable to enable safe sharing across threads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tag {
    /// Relative path for display (e.g., "src/lib.rs")
    #[serde(with = "arc_str_serde")]
    pub rel_fname: Arc<str>,
    /// Absolute path for I/O operations
    #[serde(with = "arc_str_serde")]
    pub fname: Arc<str>,
    /// Line number (1-indexed)
    pub line: u32,
    /// Symbol name (function, class, variable name)
    #[serde(with = "arc_str_serde")]
    pub name: Arc<str>,
    /// "def" for definition, "ref" for reference
    pub kind: TagKind,
    /// AST node type: "function", "class", "method", "call", etc.
    #[serde(with = "arc_str_serde")]
    pub node_type: Arc<str>,
    /// Enclosing scope's name (class or function containing this symbol)
    #[serde(serialize_with = "arc_str_serde::serialize_opt", deserialize_with = "arc_str_serde::deserialize_opt")]
    pub parent_name: Option<Arc<str>>,
    /// Line of the enclosing scope
    pub parent_line: Option<u32>,
    /// Function signature info (for functions/methods)
    pub signature: Option<SignatureInfo>,
    /// Class fields (for classes/structs)
    pub fields: Option<Vec<FieldInfo>>,
    /// Additional metadata from tree-sitter captures (type hints, receivers, etc.)
    /// Keys depend on language and query: "receiver", "var_type", "import_module", etc.
    pub metadata: Option<std::collections::HashMap<String, String>>,
}

/// Tag kind - definition or reference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TagKind {
    /// Symbol definition (function def, class def, variable assignment)
    Def,
    /// Symbol reference (function call, variable use)
    Ref,
}

impl TagKind {
    /// Check if this is any kind of definition
    pub fn is_definition(&self) -> bool {
        matches!(self, TagKind::Def)
    }

    /// Check if this is any kind of reference
    pub fn is_reference(&self) -> bool {
        matches!(self, TagKind::Ref)
    }
}

impl Tag {
    /// Check if this is a definition tag
    pub fn is_def(&self) -> bool {
        matches!(self.kind, TagKind::Def)
    }

    /// Check if this is a reference tag
    pub fn is_ref(&self) -> bool {
        matches!(self.kind, TagKind::Ref)
    }
}

/// A tag with its computed importance rank.
/// The rank is a combination of PageRank score and contextual boosts.
#[derive(Debug, Clone)]
pub struct RankedTag {
    /// Combined importance score (PageRank × boosts)
    pub rank: f64,
    /// The underlying tag
    pub tag: Tag,
}

impl RankedTag {
    pub fn new(rank: f64, tag: Tag) -> Self {
        Self { rank, tag }
    }
}

/// Ordering by rank (descending - highest rank first)
impl PartialEq for RankedTag {
    fn eq(&self, other: &Self) -> bool {
        self.rank == other.rank
    }
}

impl Eq for RankedTag {}

impl PartialOrd for RankedTag {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RankedTag {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse order - higher rank comes first
        other.rank.partial_cmp(&self.rank).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Configuration for the ranking system.
/// All values are tunable at runtime for experimentation.
#[derive(Debug, Clone)]
pub struct RankingConfig {
    // PageRank settings
    pub pagerank_alpha: f64,
    pub pagerank_chat_multiplier: f64,

    // Depth weights for personalization
    pub depth_weight_root: f64,
    pub depth_weight_moderate: f64,
    pub depth_weight_deep: f64,
    pub depth_weight_vendor: f64,
    pub depth_threshold_shallow: usize,
    pub depth_threshold_moderate: usize,

    // Boost multipliers
    pub boost_mentioned_ident: f64,
    pub boost_mentioned_file: f64,
    pub boost_chat_file: f64,
    pub boost_temporal_coupling: f64,
    pub boost_focus_expansion: f64,
    pub boost_caller_weight: f64,

    // Call graph / focus expansion settings
    pub focus_expansion_max_hops: usize,
    pub focus_expansion_decay: f64,

    // Test↔source coupling settings
    // Codex optimization identified this as a missing architectural feature:
    // "path-aware test↔crate coupling edges" for surfacing related test files
    pub boost_test_coupling: f64,
    pub test_coupling_min_confidence: f64,

    // Hub damping: penalizes "hub" nodes with excessive incoming edges.
    // Codex identified "degree-normalized hub damping" as needed to prevent
    // utility functions (log(), print()) from dominating rankings.
    // Values:
    //   0.0 = no damping (heavily-called functions get full boost)
    //   1.0 = full damping (caller count is neutralized)
    //   >1.0 = penalty (heavily-called functions are penalized as noise)
    pub hub_damping: f64,

    // Git settings
    pub git_recency_decay_days: f64,
    pub git_recency_max_boost: f64,
    pub git_churn_threshold: usize,
    pub git_churn_max_boost: f64,
    pub git_author_boost: f64,
    pub git_badge_recent_days: u32,
    pub git_badge_churn_commits: usize,

    // Lifecycle phase thresholds
    pub phase_crystal_min_age_days: u32,
    pub phase_crystal_min_quiet_days: u32,
    pub phase_rotting_min_age_days: u32,
    pub phase_rotting_max_quiet_days: u32,
    pub phase_rotting_churn_multiplier: f64,
    pub phase_emergent_max_age_days: u32,

    // Vendor patterns
    pub vendor_patterns: Vec<String>,
}

impl Default for RankingConfig {
    fn default() -> Self {
        Self {
            // PageRank
            pagerank_alpha: 0.85,
            pagerank_chat_multiplier: 100.0,

            // Depth weights
            depth_weight_root: 1.0,
            depth_weight_moderate: 0.5,
            depth_weight_deep: 0.1,
            depth_weight_vendor: 0.01,
            depth_threshold_shallow: 2,
            depth_threshold_moderate: 4,

            // Boosts
            boost_mentioned_ident: 10.0,
            boost_mentioned_file: 5.0,
            boost_chat_file: 20.0,
            boost_temporal_coupling: 3.0,
            boost_focus_expansion: 5.0,
            boost_caller_weight: 2.0,  // Files with heavily-called functions

            // Call graph / focus expansion
            focus_expansion_max_hops: 2,  // BFS depth through call relationships
            focus_expansion_decay: 0.5,   // Weight decay per hop (0.5 = halve each hop)

            // Test↔source coupling
            boost_test_coupling: 5.0,      // Test files boost their source files
            test_coupling_min_confidence: 0.5, // Minimum pattern match confidence

            // Hub damping: balance between "called = important" and "utility = noise"
            // 0.0 = no damping (heavily-called functions are boosted)
            // 1.0 = neutralize caller boost entirely
            // Future: Codex may tune this based on repo characteristics
            hub_damping: 0.0,

            // Git
            git_recency_decay_days: 30.0,
            git_recency_max_boost: 10.0,
            git_churn_threshold: 5,
            git_churn_max_boost: 6.0,
            git_author_boost: 1.5,
            git_badge_recent_days: 7,
            git_badge_churn_commits: 10,

            // Lifecycle phases
            phase_crystal_min_age_days: 180,
            phase_crystal_min_quiet_days: 30,
            phase_rotting_min_age_days: 90,
            phase_rotting_max_quiet_days: 14,
            phase_rotting_churn_multiplier: 1.5,
            phase_emergent_max_age_days: 30,

            // Vendor
            vendor_patterns: vec![
                "node_modules".into(),
                "vendor".into(),
                "third_party".into(),
                "__pycache__".into(),
                "site-packages".into(),
                ".git".into(),
                "target".into(),
            ],
        }
    }
}

/// File lifecycle phase - indicates maturity and stability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FilePhase {
    /// Old and stable - settled code, safe to rely on
    Crystal,
    /// Old but recently churning - tech debt surfacing
    Rotting,
    /// New file - still finding its shape
    Emergent,
    /// Normal development - actively being worked on
    Evolving,
}

impl FilePhase {
    pub fn badge(&self) -> &'static str {
        match self {
            FilePhase::Crystal => "crystal",
            FilePhase::Rotting => "rotting",
            FilePhase::Emergent => "emergent",
            FilePhase::Evolving => "evolving",
        }
    }
}

/// Intent classification for focus-aware ranking.
/// Different intents get different ranking recipes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Intent {
    /// Debugging - favor callers, recent changes
    Debug,
    /// Exploring - neutral graph structure
    Explore,
    /// Extending - favor API surfaces
    Extend,
    /// Refactoring - favor high-churn code
    Refactor,
}

impl Intent {
    /// Get the ranking recipe for this intent
    pub fn recipe(&self) -> IntentRecipe {
        match self {
            Intent::Debug => IntentRecipe {
                recency_weight: 1.5,
                churn_weight: 0.8,
                reverse_edge_bias: 2.0,
            },
            Intent::Explore => IntentRecipe {
                recency_weight: 1.0,
                churn_weight: 1.0,
                reverse_edge_bias: 1.0,
            },
            Intent::Extend => IntentRecipe {
                recency_weight: 1.2,
                churn_weight: 0.5,
                reverse_edge_bias: 0.7,
            },
            Intent::Refactor => IntentRecipe {
                recency_weight: 0.8,
                churn_weight: 2.0,
                reverse_edge_bias: 1.0,
            },
        }
    }
}

/// Ranking weights for a given intent
#[derive(Debug, Clone, Copy)]
pub struct IntentRecipe {
    pub recency_weight: f64,
    pub churn_weight: f64,
    pub reverse_edge_bias: f64,
}

/// Symbol identifier - (file, symbol_name) tuple for graph nodes
pub type SymbolId = (Arc<str>, Arc<str>);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detail_levels() {
        assert!(DetailLevel::Low < DetailLevel::Medium);
        assert!(DetailLevel::Medium < DetailLevel::High);
    }

    #[test]
    fn test_signature_render() {
        let sig = SignatureInfo {
            parameters: vec![
                ("host".into(), Some("str".into())),
                ("port".into(), Some("int".into())),
            ],
            return_type: Some("bool".into()),
            decorators: vec![],
            raw: None,
        };

        assert_eq!(sig.render(DetailLevel::Low), "(host, port)");
        assert_eq!(sig.render(DetailLevel::Medium), "(host: str, port: int) -> bool");
        assert_eq!(sig.render(DetailLevel::High), "(host: str, port: int) -> bool");
    }

    #[test]
    fn test_simplify_type() {
        assert_eq!(simplify_type("Dict[str, int]"), "Dict");
        assert_eq!(simplify_type("Vec<String>"), "Vec");
        assert_eq!(simplify_type("std::collections::HashMap"), "HashMap");
        assert_eq!(simplify_type("collections.OrderedDict"), "OrderedDict");
        assert_eq!(simplify_type("int"), "int");
    }

    #[test]
    fn test_ranked_tag_ordering() {
        let tag1 = Tag {
            rel_fname: "a.rs".into(),
            fname: "/a.rs".into(),
            line: 1,
            name: "foo".into(),
            kind: TagKind::Def,
            node_type: "function".into(),
            parent_name: None,
            parent_line: None,
            signature: None,
            fields: None,
            metadata: None,
        };
        let tag2 = tag1.clone();

        let ranked1 = RankedTag::new(0.5, tag1);
        let ranked2 = RankedTag::new(0.8, tag2);

        // Higher rank should come first
        assert!(ranked2 < ranked1);
    }

    #[test]
    fn test_tag_kind_helpers() {
        assert!(TagKind::Def.is_definition());
        assert!(!TagKind::Def.is_reference());
        assert!(TagKind::Ref.is_reference());
        assert!(!TagKind::Ref.is_definition());
    }
}
