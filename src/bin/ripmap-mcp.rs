//! ripmap MCP server binary.
//!
//! Runs the ripmap codebase cartography tool as an MCP server over stdio.
//! This enables AI assistants like Claude to invoke ripmap for understanding
//! codebase structure.
//!
//! # Usage
//!
//! ```bash
//! ripmap-mcp
//! ```
//!
//! The server communicates via JSON-RPC over stdio and provides the
//! `grep_map` tool for generating topology-aware structural maps.

use anyhow::Result;
use rmcp::{transport::stdio, ServiceExt};
use ripmap::mcp::RipmapServer;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize and run the MCP server over stdio
    let service = RipmapServer::new().serve(stdio()).await?;

    // Wait for the service to complete (runs until client disconnects)
    service.waiting().await?;

    Ok(())
}
