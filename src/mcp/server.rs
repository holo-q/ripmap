//! MCP server implementation for ripmap.
//!
//! Provides the `grep_map` tool via MCP protocol over stdio.
//! This enables AI assistants to invoke ripmap for codebase cartography.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use rmcp::{
    ServerHandler,
    handler::server::{router::tool::ToolRouter, tool::Parameters},
    model::{ErrorCode, ErrorData as McpError, *},
    tool, tool_handler, tool_router,
};
use serde::{Deserialize, Serialize};

use crate::discovery::find_source_files;
use crate::extraction::{Parser, extract_tags};
use crate::ranking::{BoostCalculator, PageRanker};
use crate::rendering::DirectoryRenderer;
use crate::types::{DetailLevel, RankingConfig};

/// Ripmap MCP server - exposes codebase cartography as an MCP tool.
///
/// This server wraps the full ripmap pipeline and exposes it via the
/// Model Context Protocol for use by AI assistants.
#[derive(Debug, Clone)]
pub struct RipmapServer {
    tool_router: ToolRouter<RipmapServer>,
}

/// Request parameters for the grep_map tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[allow(dead_code)] // Future: implement other_files, exclude_unranked, force_refresh
pub struct GrepMapRequest {
    /// Absolute path to the project root directory.
    #[schemars(description = "Absolute path to the project root directory")]
    pub project_root: String,

    /// Files you're actively working on (relative paths). Get highest ranking boost.
    #[schemars(
        description = "Files you're actively working on (relative paths). Get highest ranking boost."
    )]
    pub chat_files: Option<Vec<String>>,

    /// Additional files to consider. If omitted, scans entire project.
    #[schemars(description = "Additional files to consider. If omitted, scans entire project.")]
    pub other_files: Option<Vec<String>>,

    /// Token budget for the map output (default: 8192). Increase for more detail.
    #[schemars(description = "Token budget for the map output (default: 8192)")]
    pub token_limit: Option<usize>,

    /// Hide files with PageRank of 0 (peripheral files).
    #[schemars(description = "Hide files with PageRank of 0 (peripheral files)")]
    pub exclude_unranked: Option<bool>,

    /// Bypass cache and reparse everything.
    #[schemars(description = "Bypass cache and reparse everything")]
    pub force_refresh: Option<bool>,

    /// Files mentioned in conversation (mid-level ranking boost).
    #[schemars(description = "Files mentioned in conversation (mid-level ranking boost)")]
    pub mentioned_files: Option<Vec<String>>,

    /// Identifiers to boost (function/class names you're looking for).
    #[schemars(description = "Identifiers to boost (function/class names)")]
    pub mentioned_idents: Option<Vec<String>>,
}

/// Response from the grep_map tool.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct GrepMapResponse {
    /// The generated codebase map.
    pub map: String,
    /// Report with statistics about the mapping.
    pub report: GrepMapReport,
}

/// Statistics report from grep_map.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct GrepMapReport {
    /// Number of files excluded from output.
    pub excluded: usize,
    /// Number of definition matches found.
    pub definition_matches: usize,
    /// Number of reference matches found.
    pub reference_matches: usize,
    /// Total files considered in analysis.
    pub total_files_considered: usize,
}

#[tool_router]
impl RipmapServer {
    /// Create a new ripmap MCP server.
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    /// Generate a topology-aware structural map using PageRank over the dependency graph.
    ///
    /// **What this provides:**
    /// NOT alphabetical file lists. YES graph-theoretic importance analysis.
    /// - Parses all code with tree-sitter (functions, classes, imports, references)
    /// - Builds dependency graph: files as nodes, symbol references as edges
    /// - Runs PageRank with depth-aware personalization (root=1.0x, vendor=0.01x)
    /// - Binary-searches token budget to maximize information density
    ///
    /// **Topology preservation:**
    /// Output maintains directory hierarchy and class structure:
    /// - Directory nesting shows architectural layers
    /// - Classes display fields/properties/methods grouped hierarchically
    /// - Multi-line signatures collapsed to one line with full type info
    ///
    /// **Causality model:**
    /// High-ranked files are *dependencies* of many others (causal anchors).
    /// If session.py has high PageRank, it's because many files import from it.
    /// This is transitive importance, not file size or alphabetical proximity.
    #[tool(
        name = "grep_map",
        description = "Generate a topology-aware structural map using PageRank over the dependency graph. Surfaces the load-bearing structure of a codebase by parsing all code, building a dependency graph, and ranking files by structural importance."
    )]
    async fn grep_map(
        &self,
        Parameters(request): Parameters<GrepMapRequest>,
    ) -> Result<CallToolResult, McpError> {
        // Validate project root
        let root = PathBuf::from(&request.project_root);
        if !root.is_dir() {
            return Err(McpError {
                code: ErrorCode(-32602),
                message: Cow::from(format!(
                    "Project root directory not found: {}",
                    request.project_root
                )),
                data: None,
            });
        }

        // Canonicalize root for consistent paths
        let root = root.canonicalize().map_err(|e| McpError {
            code: ErrorCode(-32603),
            message: Cow::from(format!("Failed to resolve root path: {}", e)),
            data: None,
        })?;

        let token_limit = request.token_limit.unwrap_or(8192);
        let chat_files = request.chat_files.unwrap_or_default();
        let mentioned_files: HashSet<String> = request
            .mentioned_files
            .unwrap_or_default()
            .into_iter()
            .collect();
        let mentioned_idents: HashSet<String> = request
            .mentioned_idents
            .unwrap_or_default()
            .into_iter()
            .collect();

        // Stage 1: File Discovery
        let files = find_source_files(&root, false).map_err(|e| McpError {
            code: ErrorCode(-32603),
            message: Cow::from(format!("File discovery failed: {}", e)),
            data: None,
        })?;

        if files.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No source files found. Check your path and .gitignore settings.",
            )]));
        }

        // Stage 2: Tag Extraction
        let parser = Parser::new();
        let mut tags_by_file: HashMap<String, Vec<crate::Tag>> = HashMap::new();
        let mut _total_tags = 0; // Future: include in detailed stats
        let mut definition_matches = 0;
        let mut reference_matches = 0;

        for file in &files {
            let rel_fname = file
                .strip_prefix(&root)
                .unwrap_or(file)
                .to_string_lossy()
                .to_string();

            match extract_tags(file, &rel_fname, &parser) {
                Ok(tags) => {
                    for tag in &tags {
                        if tag.is_def() {
                            definition_matches += 1;
                        } else {
                            reference_matches += 1;
                        }
                    }
                    _total_tags += tags.len();
                    if !tags.is_empty() {
                        tags_by_file.insert(rel_fname, tags);
                    }
                }
                Err(_) => continue,
            }
        }

        // Stage 3 & 4: PageRank
        let config = RankingConfig::default();
        let page_ranker = PageRanker::new(config.clone());

        // Resolve chat files to relative paths
        let chat_fnames: Vec<String> = chat_files
            .iter()
            .filter_map(|f| {
                let path = root.join(f);
                if path.exists() { Some(f.clone()) } else { None }
            })
            .collect();

        let file_ranks = page_ranker.compute_ranks(&tags_by_file, &chat_fnames);

        // Stage 5: Apply Boosts
        let boost_calculator = BoostCalculator::new(config);
        let chat_fnames_set: HashSet<String> = chat_fnames.into_iter().collect();

        let ranked_tags = boost_calculator.apply_boosts(
            &tags_by_file,
            &file_ranks,
            None,
            &chat_fnames_set,
            &mentioned_files,
            &mentioned_idents,
            &HashSet::new(),
            None,
            None,
            None,
        );

        // Stage 6: Rendering
        let token_counter = |s: &str| s.len() / 4;
        let renderer = DirectoryRenderer::new(Box::new(token_counter));

        let detail = if token_limit >= 16384 {
            DetailLevel::High
        } else if token_limit >= 4096 {
            DetailLevel::Medium
        } else {
            DetailLevel::Low
        };

        let output = renderer.render(&ranked_tags, detail, &HashMap::new(), &HashMap::new());

        // Build response
        let header = format!(
            "# Codebase Map: {} | {} symbols | ~{} tokens\n\n",
            if ranked_tags.len() > 100 {
                "dense"
            } else {
                "sparse"
            },
            ranked_tags.len(),
            token_counter(&output)
        );

        let map_content = format!("{}{}", header, output);

        let report = GrepMapReport {
            excluded: files.len() - tags_by_file.len(),
            definition_matches,
            reference_matches,
            total_files_considered: files.len(),
        };

        // Return as JSON for structured response
        let response = GrepMapResponse {
            map: map_content,
            report,
        };

        let json = serde_json::to_string_pretty(&response).map_err(|e| McpError {
            code: ErrorCode(-32603),
            message: Cow::from(format!("JSON serialization failed: {}", e)),
            data: None,
        })?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }
}

impl Default for RipmapServer {
    fn default() -> Self {
        Self::new()
    }
}

#[tool_handler]
impl ServerHandler for RipmapServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "ripmap".into(),
                version: env!("CARGO_PKG_VERSION").into(),
            },
            instructions: Some(
                "Ultra-fast codebase cartography using PageRank. \
                 Use grep_map to generate topology-aware structural maps \
                 that surface load-bearing code structure."
                    .to_string(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_creation() {
        let server = RipmapServer::new();
        let info = server.get_info();
        assert_eq!(info.server_info.name, "ripmap");
    }
}
