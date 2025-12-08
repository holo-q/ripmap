//! MCP (Model Context Protocol) server for ripmap.
//!
//! Exposes ripmap's codebase cartography as an MCP tool that can be
//! invoked by AI assistants like Claude. The server runs over stdio
//! and provides:
//!
//! - `grep_map`: Generate topology-aware structural maps using PageRank
//!
//! # Architecture
//!
//! The MCP server wraps the core ripmap pipeline:
//! ```text
//! MCP Request → ripmap pipeline → MCP Response
//!     ↓              ↓                ↓
//! JSON-RPC      discover/extract   JSON-RPC
//! over stdio    rank/render        over stdio
//! ```
//!
//! # Usage
//!
//! Run the MCP server:
//! ```bash
//! ripmap-mcp
//! ```
//!
//! Or as a subcommand:
//! ```bash
//! ripmap mcp
//! ```

mod server;

pub use server::RipmapServer;
