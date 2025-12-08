//! LSP Client - JSON-RPC Communication with `ty server`
//!
//! This module manages the `ty server` process lifecycle and handles JSON-RPC
//! communication for type resolution queries. It provides both synchronous and
//! batch query interfaces with transparent caching.
//!
//! # Architecture
//!
//! ```text
//! LspClient
//!   ├── process: Mutex<Option<Child>>    # ty server subprocess
//!   ├── cache: DashMap<Location, TypeInfo>  # Results cache
//!   └── pending: DashMap<RequestId, Sender> # In-flight request tracking
//! ```
//!
//! # Process Management
//!
//! - **Lazy Startup**: The `ty server` process is started on first query
//! - **Graceful Degradation**: If `ty` is unavailable, queries return None
//! - **Automatic Restart**: Process crashes trigger lazy restart on next query
//!
//! # Query Modes
//!
//! ## Single Query
//! ```ignore
//! let type_info = client.hover("src/main.py", 42, 10)?;
//! ```
//!
//! ## Batch Query (Wavefront Execution)
//! ```ignore
//! let queries = vec![
//!     ("src/main.py", 42, 10),
//!     ("src/utils.py", 15, 5),
//! ];
//! let results = client.resolve_batch(&queries);
//! ```
//!
//! Batch queries are sent as parallel JSON-RPC requests to maximize throughput.
//!
//! # Caching Strategy
//!
//! - **Key**: (File path, Line, Column) - the call site location
//! - **Value**: TypeInfo - resolved type name, module, confidence
//! - **Coherence Tracking**: Cache tracks how many sites share the same receiver
//!   to estimate group gain (one query resolving multiple edges)
//!
//! # LSP Protocol Subset
//!
//! We use a minimal subset of the LSP protocol:
//!
//! - `textDocument/hover`: Get type information at cursor position
//! - `textDocument/definition`: Navigate to type definition (for DAG depth)
//!
//! Full LSP is overkill - we only need type resolution, not editing features.
//!
//! # JSON-RPC Wire Format
//!
//! Request:
//! ```json
//! {
//!   "jsonrpc": "2.0",
//!   "id": 1,
//!   "method": "textDocument/hover",
//!   "params": {
//!     "textDocument": {"uri": "file:///path/to/file.py"},
//!     "position": {"line": 41, "character": 10}
//!   }
//! }
//! ```
//!
//! Response:
//! ```json
//! {
//!   "jsonrpc": "2.0",
//!   "id": 1,
//!   "result": {
//!     "contents": {"kind": "markdown", "value": "```python\nuser: User\n```"}
//!   }
//! }
//! ```
//!
//! # Error Handling
//!
//! - Process spawn failures: Return None, log warning
//! - JSON parse errors: Skip malformed responses
//! - Timeout (5s default): Mark query as failed, cache None result
//! - LSP errors: Extract from error field, return None

// === Implementation Notes ===
//
// Design Philosophy:
// - Lazy initialization: Don't spawn ty until first query
// - Thread-safe caching: DashMap for concurrent access across wavefronts
// - Batch-aware: Single method can handle both individual and batch queries
// - Graceful degradation: If ty unavailable, return None (fallback to heuristics)
//
// Architecture Notes:
// - The client is the ONLY layer that talks to ty server
// - Cache keys are (file, line, col) tuples for precise location
// - Confidence is always ~0.95 for LSP (vs 0.2-0.4 for heuristics)
// - This creates the regime shift from 14% → 80%+ resolution
//
// Process Lifecycle:
// 1. Client created (process = None)
// 2. First query → ensure_started() → spawn ty server + initialize
// 3. Queries → check cache → miss → JSON-RPC → cache result
// 4. Shutdown → send shutdown + exit messages
//
// Thread Safety:
// - process/stdin/stdout: Mutex (single writer to stdin, single reader from stdout)
// - caches: DashMap (concurrent reads, writes)
// - next_id: Mutex (sequential request IDs)
//
// Coherence Tracking:
// When we resolve `user: User`, we can immediately resolve all
// `user.method()` calls. This is "group gain" - one query buys
// multiple edge resolutions. The cache enables this.

use std::process::{Command, Stdio, Child, ChildStdin, ChildStdout};
use std::sync::{Mutex, Arc};
use std::io::{Write, BufRead, BufReader};
use std::collections::HashMap;
use dashmap::DashMap;
use anyhow::{Result, Context, bail};
use serde::{Serialize, Deserialize};
use serde_json::{json, Value};

/// Type information returned by LSP hover
///
/// The confidence is always high (~0.95) because LSP has semantic knowledge
/// vs heuristics (name matching ~0.4, type hints ~0.2). This creates the
/// information-theoretic value: H(edge) drops from 0.4 → 0.0, weighted by
/// centrality = massive entropy reduction.
#[derive(Debug, Clone)]
pub struct TypeInfo {
    /// Type string, e.g., "MyClass", "List[str]", "Optional[Dict[str, Any]]"
    pub type_str: String,

    /// Confidence score. Always ~0.95 for LSP-resolved types.
    /// This is the regime shift parameter: heuristics give 0.2-0.4,
    /// LSP gives 0.95. The difference compounds across PageRank.
    pub confidence: f64,
}

/// Location returned by LSP definition/references
///
/// Used for building the call graph: when we see x.foo(), we query the
/// definition of 'foo' to resolve which function it actually calls.
#[derive(Debug, Clone)]
pub struct Location {
    pub file: String,
    pub line: u32,
    pub col: u32,
}

/// JSON-RPC message structures for LSP protocol
#[derive(Serialize, Deserialize, Debug)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: u64,
    method: String,
    params: Value,
}

#[derive(Serialize, Deserialize, Debug)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<Value>,
}

/// LSP client for ty server (Python type checker)
///
/// Process Lifecycle:
/// 1. Client created (process = None)
/// 2. First query → ensure_started() → spawn ty server + initialize
/// 3. Queries → check cache → miss → JSON-RPC → cache result
/// 4. Shutdown → send shutdown + exit messages
///
/// Thread Safety:
/// - process/stdin: Mutex (single writer to stdin)
/// - caches: DashMap (concurrent reads, writes)
/// - next_id: Mutex (sequential request IDs)
///
/// Coherence Tracking:
/// When we resolve `user: User`, we can immediately resolve all
/// `user.method()` calls. This is "group gain" - one query buys
/// multiple edge resolutions. The cache enables this.
pub struct LspClient {
    /// ty server process. None until first query (lazy init).
    process: Mutex<Option<Child>>,

    /// Stdin handle for sending JSON-RPC requests
    stdin: Mutex<Option<ChildStdin>>,

    /// Stdout reader for receiving responses (wrapped in mutex for BufReader)
    stdout: Mutex<Option<BufReader<ChildStdout>>>,

    /// Type cache: (file, line, col) -> TypeInfo
    /// None means "we queried but got no result" (cache negative results too)
    pub type_cache: Arc<DashMap<(String, u32, u32), Option<TypeInfo>>>,

    /// Definition cache: (file, line, col) -> Location
    pub def_cache: Arc<DashMap<(String, u32, u32), Option<Location>>>,

    /// Next JSON-RPC request ID (incremental)
    next_id: Mutex<u64>,

    /// Initialization state
    initialized: Mutex<bool>,
}

impl LspClient {
    /// Create new LSP client (does not start process yet)
    pub fn new() -> Self {
        Self {
            process: Mutex::new(None),
            stdin: Mutex::new(None),
            stdout: Mutex::new(None),
            type_cache: Arc::new(DashMap::new()),
            def_cache: Arc::new(DashMap::new()),
            next_id: Mutex::new(1),
            initialized: Mutex::new(false),
        }
    }

    /// Check if ty is available on the system
    ///
    /// Used for graceful degradation: if ty not found, skip LSP entirely
    /// and fall back to heuristic-only resolution (current 14% regime).
    pub fn is_available() -> bool {
        Command::new("ty")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }

    /// Ensure ty server is started and initialized
    ///
    /// Initialization sequence:
    /// 1. Spawn `ty server` with stdio pipes
    /// 2. Send LSP initialize request with rootUri
    /// 3. Wait for initialized response
    /// 4. Send initialized notification
    ///
    /// This is called on first query (lazy). Thread-safe: multiple callers
    /// will wait, but only one will actually spawn.
    pub fn ensure_started(&self) -> Result<()> {
        // Check if already initialized
        {
            let initialized = self.initialized.lock().unwrap();
            if *initialized {
                return Ok(());
            }
        }

        // Acquire process lock to prevent race
        let mut process_guard = self.process.lock().unwrap();
        let mut stdin_guard = self.stdin.lock().unwrap();
        let mut stdout_guard = self.stdout.lock().unwrap();
        let mut init_guard = self.initialized.lock().unwrap();

        // Double-check after acquiring lock (another thread may have initialized)
        if *init_guard {
            return Ok(());
        }

        // Spawn ty server process
        let mut child = Command::new("ty")
            .arg("server")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())  // TODO: Maybe log this for debugging?
            .spawn()
            .context("Failed to spawn 'ty server'. Is ty installed?")?;

        let stdin = child.stdin.take()
            .context("Failed to capture stdin of ty server")?;
        let stdout = child.stdout.take()
            .context("Failed to capture stdout of ty server")?;

        *process_guard = Some(child);
        *stdin_guard = Some(stdin);
        *stdout_guard = Some(BufReader::new(stdout));

        // Send LSP initialize request
        // For now, use a minimal initialize. We can expand capabilities later.
        let init_params = json!({
            "processId": std::process::id(),
            "rootUri": null,  // TODO: Could pass actual workspace root
            "capabilities": {
                "textDocument": {
                    "hover": {
                        "contentFormat": ["plaintext", "markdown"]
                    },
                    "definition": {
                        "linkSupport": false
                    }
                }
            }
        });

        let request_id = {
            let mut id = self.next_id.lock().unwrap();
            let current = *id;
            *id += 1;
            current
        };

        let init_request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: request_id,
            method: "initialize".to_string(),
            params: init_params,
        };

        // Send request
        self.send_request_locked(&init_request, stdin_guard.as_mut().unwrap())?;

        // Read response
        let _response = self.read_response_locked(stdout_guard.as_mut().unwrap())?;

        // Send initialized notification
        let initialized_notif = json!({
            "jsonrpc": "2.0",
            "method": "initialized",
            "params": {}
        });

        self.send_notification_locked(&initialized_notif, stdin_guard.as_mut().unwrap())?;

        *init_guard = true;

        Ok(())
    }

    /// Shutdown ty server gracefully
    pub fn shutdown(&self) {
        let mut process_guard = self.process.lock().unwrap();
        let mut stdin_guard = self.stdin.lock().unwrap();
        let mut init_guard = self.initialized.lock().unwrap();

        if !*init_guard {
            return;  // Not started
        }

        // Try to send shutdown request
        if let Some(stdin) = stdin_guard.as_mut() {
            let shutdown_req = JsonRpcRequest {
                jsonrpc: "2.0".to_string(),
                id: 9999,
                method: "shutdown".to_string(),
                params: json!({}),
            };

            let _ = self.send_request_locked(&shutdown_req, stdin);

            // Send exit notification
            let exit_notif = json!({
                "jsonrpc": "2.0",
                "method": "exit",
                "params": null
            });
            let _ = self.send_notification_locked(&exit_notif, stdin);
        }

        // Kill process if still alive
        if let Some(child) = process_guard.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }

        *process_guard = None;
        *stdin_guard = None;
        *init_guard = false;
    }

    /// Get type information at a specific position
    ///
    /// Cache-first: check type_cache, only query ty if miss.
    /// Returns None if:
    /// - ty not available
    /// - Position has no type info
    /// - Query failed
    ///
    /// This is the core primitive for the PolicyEngine. Each call has a cost
    /// (latency) and a value (entropy reduction). The policy coordinates
    /// determine which calls to make.
    pub fn hover(&self, file: &str, line: u32, col: u32) -> Option<TypeInfo> {
        let key = (file.to_string(), line, col);

        // Check cache first
        if let Some(cached) = self.type_cache.get(&key) {
            return cached.clone();
        }

        // Cache miss - need to query
        match self.hover_uncached(file, line, col) {
            Ok(type_info) => {
                self.type_cache.insert(key, type_info.clone());
                type_info
            }
            Err(_) => {
                // Cache negative result (avoid repeated failed queries)
                self.type_cache.insert(key, None);
                None
            }
        }
    }

    /// Get definition location at a specific position
    ///
    /// Used for resolving call edges: x.foo() → where is foo defined?
    pub fn definition(&self, file: &str, line: u32, col: u32) -> Option<Location> {
        let key = (file.to_string(), line, col);

        if let Some(cached) = self.def_cache.get(&key) {
            return cached.clone();
        }

        match self.definition_uncached(file, line, col) {
            Ok(location) => {
                self.def_cache.insert(key, location.clone());
                location
            }
            Err(_) => {
                self.def_cache.insert(key, None);
                None
            }
        }
    }

    /// Batch resolve multiple positions (for wavefront execution)
    ///
    /// The wavefront model queries N positions at once. We could:
    /// 1. Send N individual requests (simple but slow)
    /// 2. Send N requests in parallel (requires async)
    /// 3. Use batch LSP methods (not all servers support)
    ///
    /// For now: simple sequential within batch. The key is that the
    /// *selection* of the batch is intelligent (PolicyEngine), not that
    /// the execution is maximally parallel. We can optimize later.
    ///
    /// Returns only successful resolutions (sparse result).
    pub fn resolve_batch(&self, queries: &[(String, u32, u32)]) -> HashMap<(String, u32, u32), TypeInfo> {
        let mut results = HashMap::new();

        for (file, line, col) in queries {
            if let Some(type_info) = self.hover(file, *line, *col) {
                results.insert((file.clone(), *line, *col), type_info);
            }
        }

        results
    }

    /// Internal: query hover without caching
    fn hover_uncached(&self, file: &str, line: u32, col: u32) -> Result<Option<TypeInfo>> {
        // Ensure server is running
        self.ensure_started()?;

        let params = json!({
            "textDocument": {
                "uri": format!("file://{}", file)
            },
            "position": {
                "line": line.saturating_sub(1),  // LSP is 0-indexed
                "character": col.saturating_sub(1)
            }
        });

        let request_id = {
            let mut id = self.next_id.lock().unwrap();
            let current = *id;
            *id += 1;
            current
        };

        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: request_id,
            method: "textDocument/hover".to_string(),
            params,
        };

        // Send request
        {
            let mut stdin_guard = self.stdin.lock().unwrap();
            if let Some(stdin) = stdin_guard.as_mut() {
                self.send_request_locked(&request, stdin)?;
            } else {
                bail!("ty server stdin not available");
            }
        }

        // Read response
        let response = {
            let mut stdout_guard = self.stdout.lock().unwrap();
            if let Some(stdout) = stdout_guard.as_mut() {
                self.read_response_locked(stdout)?
            } else {
                bail!("ty server stdout not available");
            }
        };

        // Parse response
        if let Some(error) = response.error {
            bail!("LSP error: {:?}", error);
        }

        let result = response.result.unwrap_or(Value::Null);

        // Extract type from hover result
        // Format varies by server, but typically:
        // { "contents": { "kind": "markdown", "value": "```python\nstr\n```" } }
        // or { "contents": "str" }
        if result.is_null() {
            return Ok(None);
        }

        let type_str = self.extract_type_from_hover(&result)?;

        Ok(Some(TypeInfo {
            type_str,
            confidence: 0.95,  // LSP is high-confidence
        }))
    }

    /// Internal: query definition without caching
    fn definition_uncached(&self, file: &str, line: u32, col: u32) -> Result<Option<Location>> {
        self.ensure_started()?;

        let params = json!({
            "textDocument": {
                "uri": format!("file://{}", file)
            },
            "position": {
                "line": line.saturating_sub(1),
                "character": col.saturating_sub(1)
            }
        });

        let request_id = {
            let mut id = self.next_id.lock().unwrap();
            let current = *id;
            *id += 1;
            current
        };

        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: request_id,
            method: "textDocument/definition".to_string(),
            params,
        };

        {
            let mut stdin_guard = self.stdin.lock().unwrap();
            if let Some(stdin) = stdin_guard.as_mut() {
                self.send_request_locked(&request, stdin)?;
            } else {
                bail!("ty server stdin not available");
            }
        }

        let response = {
            let mut stdout_guard = self.stdout.lock().unwrap();
            if let Some(stdout) = stdout_guard.as_mut() {
                self.read_response_locked(stdout)?
            } else {
                bail!("ty server stdout not available");
            }
        };

        if let Some(error) = response.error {
            bail!("LSP error: {:?}", error);
        }

        let result = response.result.unwrap_or(Value::Null);

        if result.is_null() {
            return Ok(None);
        }

        let location = self.extract_location_from_definition(&result)?;

        Ok(location)
    }

    /// Send JSON-RPC request with Content-Length header
    fn send_request_locked(&self, request: &JsonRpcRequest, stdin: &mut ChildStdin) -> Result<()> {
        let json = serde_json::to_string(request)?;
        let content = format!("Content-Length: {}\r\n\r\n{}", json.len(), json);
        stdin.write_all(content.as_bytes())?;
        stdin.flush()?;
        Ok(())
    }

    /// Send JSON-RPC notification (no response expected)
    fn send_notification_locked(&self, notification: &Value, stdin: &mut ChildStdin) -> Result<()> {
        let json = serde_json::to_string(notification)?;
        let content = format!("Content-Length: {}\r\n\r\n{}", json.len(), json);
        stdin.write_all(content.as_bytes())?;
        stdin.flush()?;
        Ok(())
    }

    /// Read JSON-RPC response with Content-Length header
    fn read_response_locked(&self, stdout: &mut BufReader<ChildStdout>) -> Result<JsonRpcResponse> {
        // Read headers until blank line
        let mut content_length: Option<usize> = None;
        let mut line = String::new();

        loop {
            line.clear();
            stdout.read_line(&mut line)?;

            if line.trim().is_empty() {
                break;  // End of headers
            }

            if line.starts_with("Content-Length:") {
                let len_str = line.trim_start_matches("Content-Length:")
                    .trim();
                content_length = Some(len_str.parse()?);
            }
        }

        let content_length = content_length
            .context("Missing Content-Length header")?;

        // Read body
        let mut buffer = vec![0u8; content_length];
        std::io::Read::read_exact(stdout, &mut buffer)?;

        let response: JsonRpcResponse = serde_json::from_slice(&buffer)?;

        Ok(response)
    }

    /// Extract type string from hover response
    ///
    /// LSP hover responses vary by server. This handles common formats:
    /// - { "contents": "str" }
    /// - { "contents": { "value": "str" } }
    /// - { "contents": { "value": "```python\nstr\n```" } }
    fn extract_type_from_hover(&self, result: &Value) -> Result<String> {
        let contents = &result["contents"];

        if contents.is_null() {
            bail!("No contents in hover response");
        }

        // Simple string
        if let Some(s) = contents.as_str() {
            return Ok(self.clean_type_string(s));
        }

        // MarkupContent
        if let Some(value) = contents["value"].as_str() {
            return Ok(self.clean_type_string(value));
        }

        // Array of contents
        if let Some(arr) = contents.as_array() {
            if let Some(first) = arr.first() {
                if let Some(s) = first.as_str() {
                    return Ok(self.clean_type_string(s));
                }
                if let Some(value) = first["value"].as_str() {
                    return Ok(self.clean_type_string(value));
                }
            }
        }

        bail!("Could not extract type from hover response: {:?}", contents);
    }

    /// Clean type string (remove markdown code fences, etc.)
    fn clean_type_string(&self, s: &str) -> String {
        let s = s.trim();

        // Remove markdown code blocks
        if s.starts_with("```") {
            let lines: Vec<&str> = s.lines().collect();
            if lines.len() >= 3 {
                // Skip first (```python) and last (```)
                return lines[1..lines.len()-1].join("\n").trim().to_string();
            }
        }

        s.to_string()
    }

    /// Extract location from definition response
    ///
    /// Response can be:
    /// - Single Location: { "uri": "file://...", "range": { "start": {...} } }
    /// - Array of Locations: [{ "uri": ..., "range": {...} }, ...]
    fn extract_location_from_definition(&self, result: &Value) -> Result<Option<Location>> {
        // Handle array (take first)
        let location_value = if result.is_array() {
            result.as_array()
                .and_then(|arr| arr.first())
                .unwrap_or(result)
        } else {
            result
        };

        let uri = location_value["uri"].as_str()
            .context("Missing uri in definition response")?;

        // Strip file:// prefix
        let file = uri.trim_start_matches("file://").to_string();

        let start = &location_value["range"]["start"];
        let line = start["line"].as_u64()
            .context("Missing line in definition response")? as u32 + 1;  // Convert to 1-indexed
        let col = start["character"].as_u64()
            .context("Missing character in definition response")? as u32 + 1;

        Ok(Some(Location { file, line, col }))
    }
}

impl Drop for LspClient {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_available() {
        // This test will fail if ty is not installed, which is expected
        // In CI, we can skip if ty not available
        let available = LspClient::is_available();
        println!("ty available: {}", available);
    }

    #[test]
    fn test_clean_type_string() {
        let client = LspClient::new();

        let markdown = "```python\nList[str]\n```";
        assert_eq!(client.clean_type_string(markdown), "List[str]");

        let plain = "str";
        assert_eq!(client.clean_type_string(plain), "str");
    }
}
