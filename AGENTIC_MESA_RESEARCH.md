# Agentic Mesa Mode Research

## Executive Summary

This document researches how to enable tool-using/agentic mode for the mesa agents (Claude, Gemini, Codex) in ripmap's training system. Currently the system uses single-shot prompting, but we want agents to be able to:

1. **Read files** (scratchpad, previous episodes, other promptgrams)
2. **Iterate on their reasoning** (multi-turn exploration)
3. **Run lightweight experiments** (potentially test parameter hypotheses)

## Current Implementation Analysis

### Location
- **Source file**: `/home/nuck/holoq/repositories/ripmap/src/training/reasoning.rs`
- **Functions**: `call_claude()`, `call_gemini()`, `call_codex()`

### Current Invocation Modes

#### Claude (`call_claude`)
```rust
claude --print -p "prompt" [--model MODEL]
```
- `--print`: Non-interactive single-shot mode
- `-p`: Pass prompt as argument
- No tools available
- Returns stdout directly

#### Gemini (`call_gemini`)
```rust
gemini -o text -y "prompt" [-m MODEL]
```
- `-o text`: Text output format
- `-y`: YOLO mode (auto-approve all actions)
- Returns stdout directly

#### Codex (`call_codex`)
```rust
codex exec --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox [-m MODEL] -o OUTPUT_FILE -
```
- `exec`: Non-interactive execution
- `-`: Read prompt from stdin
- `-o FILE`: Write final output to file
- `--dangerously-bypass-approvals-and-sandbox`: Auto-execute without prompts
- Returns file contents

---

## Tool-Enabled Modes: Research Findings

### 1. Claude CLI - Agentic Mode

#### Available Flags
```bash
claude [OPTIONS] [prompt]
```

**Key flags for agentic mode:**
- **NO `--print` flag**: Remove this to enable interactive/agentic mode
- `--tools <tools...>`: Specify available tools (e.g., "Bash,Edit,Read,Write,Grep,Glob")
  - Use `"default"` for all tools
  - Use `""` to disable all tools
  - **ONLY works with `--print` mode** (contradiction!)
- `--permission-mode <mode>`: Control approval flow
  - `acceptEdits`: Auto-accept file edits
  - `bypassPermissions`: Skip all permission checks
  - `dontAsk`: Don't ask for confirmations
  - `plan`: Planning mode
- `--add-dir <directories...>`: Additional accessible directories
- `--output-format <format>`: "text" (default), "json", "stream-json"
- `--json-schema <schema>`: Structured output validation (JSON Schema)
- `--dangerously-skip-permissions`: Bypass all permission checks (use in sandboxes only)

**Critical Discovery**: The `--tools` flag ONLY works with `--print` mode, but `--print` is single-shot! This means:
- **Option A**: Use `--print` with `--tools` for single-shot tool use (agent gets one chance to read files before responding)
- **Option B**: Drop `--print` entirely and capture session output differently (interactive mode)

**Session Resumption:**
```bash
claude --resume SESSION_ID [prompt]
claude --continue  # Resume most recent
claude --fork-session  # Create new session ID when resuming
```

#### Proposed Approach for Claude

**Strategy 1: Single-Shot with Tools** (minimal change)
```bash
claude --print \
  --tools "Read,Bash,Grep" \
  --permission-mode bypassPermissions \
  --add-dir /path/to/scratchpad \
  --output-format json \
  -p "PROMPT"
```

Pros:
- Minimal code changes
- Predictable output capture
- Agent can read files in one turn

Cons:
- Still fundamentally single-shot
- No iterative refinement
- Agent must plan all reads upfront

**Strategy 2: Multi-Turn Interactive Session** (major change)
```bash
# Initial call (no --print)
claude \
  --session-id UUID \
  --tools "Read,Bash,Grep" \
  --permission-mode bypassPermissions \
  --add-dir /path/to/scratchpad \
  --output-format stream-json \
  "INITIAL_PROMPT"

# Capture conversation
# Extract final JSON from message history
```

Pros:
- True agentic behavior (can iterate)
- Can refine reasoning based on file contents
- More powerful exploration

Cons:
- Complex output parsing (need to track message stream)
- Non-deterministic execution time
- Harder to extract final JSON
- Need session management

**Recommended**: Start with Strategy 1 (single-shot with tools) as it's a smaller leap from current architecture.

---

### 2. Gemini CLI - Agentic Mode

#### Available Flags
```bash
gemini [OPTIONS] [query...]
```

**Key flags for agentic mode:**
- **Default behavior is already agentic!** Gemini CLI is tool-enabled by default
- `--approval-mode <mode>`: Control approval flow
  - `default`: Prompt for approval
  - `auto_edit`: Auto-approve edit tools only
  - `yolo`: Auto-approve all tools (currently using `-y` which is equivalent)
- `--allowed-tools <tools...>`: Tools that run without confirmation
- `--extensions <extensions...>`: Enable specific extensions
- `-o, --output-format <format>`: "text", "json", "stream-json"
- `--include-directories <dirs...>`: Additional accessible directories
- `--sandbox`: Run in sandbox mode

**Session Resumption:**
```bash
gemini --resume latest       # Most recent
gemini --resume 5            # By index
gemini --list-sessions       # Show available
gemini --delete-session N    # Clean up
```

#### Current vs Agentic Mode

**Current invocation:**
```bash
gemini -o text -y "prompt"
```
- `-y` (yolo mode) already auto-approves tools
- `-o text` returns plain text
- This is ALREADY agentic! Just needs proper output capture

**Proposed Agentic Invocation:**
```bash
gemini \
  -o json \
  --approval-mode yolo \
  --include-directories /path/to/scratchpad,/path/to/episodes \
  "PROMPT"
```

Changes:
- Switch to `-o json` for structured output
- Explicitly use `--approval-mode yolo` (clearer than `-y`)
- Add `--include-directories` for scratchpad access

**Critical Insight**: Gemini is ALREADY running in agentic mode with `-y`! We just need to:
1. Change output format to JSON
2. Add directory access
3. Handle multi-turn conversations (optional)

---

### 3. Codex CLI - Agentic Mode

#### Available Flags
```bash
codex exec [OPTIONS] [PROMPT]
```

**Key flags for agentic mode:**
- **Already agentic by default!** `exec` runs the agent with tools
- `-a, --ask-for-approval <policy>`: Approval policy
  - `untrusted`: Only ask for untrusted commands (default-ish)
  - `on-failure`: Only ask if command fails
  - `on-request`: Model decides when to ask
  - `never`: Never ask (dangerous)
- `--full-auto`: Convenience alias (`-a on-request --sandbox workspace-write`)
- `--dangerously-bypass-approvals-and-sandbox`: Skip all approvals (currently using)
- `-s, --sandbox <mode>`: Sandbox policy
  - `read-only`: Read-only filesystem
  - `workspace-write`: Write to workspace only
  - `danger-full-access`: Full filesystem access
- `--add-dir <dirs...>`: Additional writable directories
- `-C, --cd <dir>`: Working directory
- `-o, --output-last-message <file>`: Write final message to file
- `--json`: Output events as JSONL
- `--output-schema <file>`: JSON Schema for final response

**Session Resumption:**
```bash
codex exec resume SESSION_ID [prompt]
codex exec resume --last [prompt]
```

#### Current vs Agentic Mode

**Current invocation:**
```bash
codex exec \
  --skip-git-repo-check \
  --dangerously-bypass-approvals-and-sandbox \
  -o OUTPUT_FILE \
  -
# (prompt via stdin)
```
- Already bypasses approvals
- Already writes output to file
- **Already agentic!**

**Proposed Enhanced Invocation:**
```bash
codex exec \
  --skip-git-repo-check \
  --full-auto \
  --add-dir /path/to/scratchpad \
  -C /path/to/workspace \
  -o OUTPUT_FILE \
  --output-schema schema.json \
  -
# (prompt via stdin)
```

Changes:
- Use `--full-auto` instead of `--dangerously-bypass-approvals-and-sandbox` (safer)
- Add `--add-dir` for scratchpad access
- Set working directory with `-C`
- Optionally enforce output schema

---

## Output Capture Strategies

### Challenge: Extracting JSON after Multi-Turn Exploration

All three CLIs can operate in agentic mode, but we need to reliably extract the final JSON response after the agent has potentially:
- Read multiple files
- Executed bash commands
- Iterated on its reasoning

### Strategy A: Single-Shot with Tool Access (Recommended for MVP)

**Approach**: Enable tools but keep single-turn execution.

**Claude**:
```bash
claude --print --tools "Read,Grep" --permission-mode bypassPermissions -p "PROMPT"
```
- Agent gets ONE turn to use tools, then must respond
- Output is still predictable (stdout)
- Parse JSON from response (same as current implementation)

**Gemini**:
```bash
gemini -o json --approval-mode yolo --include-directories /scratchpad "PROMPT"
```
- Already works this way!
- Just change output format to JSON

**Codex**:
```bash
codex exec --full-auto --add-dir /scratchpad -o out.json -
```
- Already works this way!
- Just add directory access

### Strategy B: Multi-Turn with Message Extraction

**Approach**: Allow full multi-turn exploration, extract final message.

**Claude**:
```bash
claude --output-format stream-json --session-id UUID ... [prompt]
```
- Parse JSONL stream
- Extract last message with role="assistant"
- Search for JSON block in content

**Gemini**:
```bash
gemini -o stream-json --resume latest [prompt]
```
- Similar JSONL parsing
- Track conversation state

**Codex**:
```bash
codex exec --json -o final.txt -
```
- `--json` produces JSONL event stream on stdout
- `-o final.txt` captures final message text
- Read from file, parse JSON

**Implementation Note**: Strategy B requires:
1. Session management (create/resume/cleanup)
2. JSONL parsing
3. Conversation state tracking
4. Timeout handling (agents might explore indefinitely)

---

## File/Tool Access Patterns

### What Files Do Agents Need?

From the training system architecture:

1. **Scratchpad** (`scratchpad.json`):
   - Previous episodes with reasoning
   - Discovered parameter interactions
   - Success/failure patterns
   - Structural proposals

2. **Previous Episodes** (individual episode files):
   - Detailed failure analysis from past runs
   - Parameter settings that were tried
   - NDCG trajectories

3. **Current Promptgram** (for L2/outer loop):
   - Sections of the prompt being optimized
   - Previous prompt versions
   - Diffs between versions

4. **Ground Truth Data** (optional):
   - Git cases showing expected rankings
   - Commit contexts
   - Repository metadata

### Recommended Directory Structure

```
/run_directory/
  scratchpad.json          # Main scratchpad
  episodes/
    episode_001.json
    episode_002.json
    ...
  promptgrams/
    inner_v001.toml
    inner_v002.toml
  ground_truth/
    cases.json
```

### Access Configuration

**Claude**:
```bash
--add-dir /run_directory
```

**Gemini**:
```bash
--include-directories /run_directory
```

**Codex**:
```bash
--add-dir /run_directory -C /run_directory
```

---

## Security & Sandboxing Considerations

### Risk Assessment

**Current System**:
- Runs in training environment (not production)
- No external network access needed
- Limited blast radius (only affects training runs)

**With Agentic Mode**:
- Agents can read arbitrary files in allowed directories
- Agents can execute bash commands (Codex especially)
- Agents can write files (less concerning for reasoning tasks)

### Mitigation Strategies

1. **Directory Restrictions**:
   - Only allow access to run-specific directories
   - Use `--add-dir` / `--include-directories` to whitelist paths
   - Never allow access to home directory or system paths

2. **Tool Restrictions** (Claude):
   - Whitelist only: `Read`, `Grep`, `Glob`
   - Avoid: `Bash`, `Edit`, `Write`, `WebFetch`
   - Command: `--tools "Read,Grep,Glob"`

3. **Sandbox Modes** (Codex):
   - Use `--sandbox workspace-write` instead of `danger-full-access`
   - Use `--full-auto` instead of `--dangerously-bypass-approvals-and-sandbox`

4. **Timeouts**:
   - Set maximum execution time per agent call
   - Kill runaway processes
   - Rust: `std::process::Command::timeout()`

5. **Resource Limits**:
   - Limit file sizes agent can read
   - Cap number of files accessed per run
   - Monitor token usage (Claude) for cost control

### Recommended Sandbox Configuration

**Claude** (safest):
```bash
claude --print \
  --tools "Read,Grep" \
  --permission-mode dontAsk \
  --add-dir /run_dir
```
- No shell access
- No file writes
- Only reading/searching

**Gemini** (moderate):
```bash
gemini -o json \
  --approval-mode auto_edit \
  --sandbox \
  --include-directories /run_dir
```
- Sandboxed execution
- Auto-approve only edits (not bash)

**Codex** (needs care):
```bash
codex exec \
  --full-auto \
  --sandbox workspace-write \
  --add-dir /run_dir \
  -C /run_dir
```
- Sandboxed writes
- Request-based approvals
- Limited to workspace

---

## Implementation Roadmap

### Phase 1: Single-Shot with Read Access (MVP)

**Goal**: Let agents read scratchpad/episodes before reasoning, but keep single-turn execution.

**Changes**:
1. Modify `call_claude()`:
   ```rust
   Command::new("claude")
       .args(["--print", "--tools", "Read,Grep,Glob"])
       .args(["--permission-mode", "bypassPermissions"])
       .args(["--add-dir", run_directory])
       .args(["-p", prompt])
   ```

2. Modify `call_gemini()`:
   ```rust
   Command::new("gemini")
       .args(["-o", "json"])  // Change from "text"
       .args(["--approval-mode", "yolo"])
       .args(["--include-directories", run_directory])
       .arg(prompt)
   ```

3. Modify `call_codex()`:
   ```rust
   Command::new("codex")
       .args(["exec", "--skip-git-repo-check"])
       .args(["--full-auto"])  // Replace --dangerously-bypass...
       .args(["--add-dir", run_directory])
       .args(["-C", run_directory])
       .args(["-o", output_file])
       .arg("-")
   ```

4. Update prompts to tell agents:
   - "You have access to Read, Grep tools"
   - "Previous episodes in: /run_dir/episodes/"
   - "Current scratchpad: /run_dir/scratchpad.json"

**Testing**:
- Verify agents can read files
- Confirm JSON extraction still works
- Check output format consistency

### Phase 2: Multi-Turn Exploration (Advanced)

**Goal**: Allow agents to iterate on reasoning, reading multiple files.

**Changes**:
1. Add session management:
   ```rust
   fn create_session(agent: Agent, run_dir: &Path) -> String {
       // Generate UUID, store session config
   }

   fn resume_session(session_id: &str, prompt: &str) -> Result<String, String> {
       // Resume and inject new prompt
   }
   ```

2. Add JSONL parsing:
   ```rust
   fn extract_final_json(stream: &str) -> Result<serde_json::Value, String> {
       // Parse JSONL, find last assistant message, extract JSON
   }
   ```

3. Add timeout handling:
   ```rust
   Command::new(...)
       .timeout(Duration::from_secs(300))  // 5 min max
   ```

4. Update each agent caller:
   - Claude: `--output-format stream-json`, `--session-id UUID`
   - Gemini: `-o stream-json`, `--resume`
   - Codex: `--json` flag for event stream

**Testing**:
- Verify agents explore multiple files
- Confirm final JSON extraction
- Test timeout behavior
- Check session cleanup

### Phase 3: Lightweight Experiments (Research)

**Goal**: Let agents test hypotheses by running mini-evaluations.

**Ideas**:
- Provide a tool: `eval_params(params: ParameterPoint) -> NDCG`
- Agent can test "what if alpha=0.9?" before committing
- Requires safe sandboxing (don't want infinite recursion!)

**Challenges**:
- Computational cost (each eval is expensive)
- Tool implementation complexity
- Risk of agent getting stuck in local search

**Recommendation**: Defer to Phase 3+, focus on file access first.

---

## Recommended Next Steps

1. **Start with Phase 1** (single-shot with read access):
   - Low risk, high value
   - Minimal code changes
   - Agents can read scratchpad history

2. **Test with each agent**:
   - Create test prompts that require file reading
   - Verify JSON extraction works
   - Compare reasoning quality with/without file access

3. **Monitor carefully**:
   - Log all file accesses
   - Track agent execution time
   - Watch for unexpected behavior

4. **Iterate on prompts**:
   - Tell agents what files are available
   - Provide file path conventions
   - Show example tool usage

5. **Consider Phase 2** after Phase 1 proves valuable:
   - Only if single-shot tool use feels limiting
   - Multi-turn adds complexity, must justify benefit

---

## Appendix: CLI Version Info

- **Claude CLI**: v2.0.61 (Claude Code)
- **Gemini CLI**: v0.19.2
- **Codex CLI**: v0.63.0

(Researched on 2025-12-08)

---

## Questions for Further Research

1. **Token costs**: Do multi-turn tool-enabled conversations cost significantly more?
2. **Reliability**: Are JSON responses still consistent with tool use enabled?
3. **Prompt engineering**: How should we instruct agents to use tools effectively?
4. **Caching**: Can we cache file contents across agent calls? (Claude supports prompt caching)
5. **Parallel execution**: Can we run multiple agentic episodes concurrently?

---

## Conclusion

All three mesa CLIs support agentic/tool-using modes:

- **Claude**: Use `--tools` with `--print` for single-shot, or drop `--print` for multi-turn
- **Gemini**: Already agentic with `-y`, just needs directory access
- **Codex**: Already agentic with `exec`, just needs directory access

**Recommended starting point**: Phase 1 (single-shot with Read access) for all three agents. This gives 80% of the benefit with 20% of the complexity.

The key insight: **We don't need multi-turn iteration immediately**. Just letting agents READ the scratchpad and previous episodes before reasoning will be a huge improvement over blind single-shot prompting.
