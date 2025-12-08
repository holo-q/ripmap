# Implementation Notes

This document tracks key implementation decisions, bugs, and lessons learned during ripmap development.

## L2 Schema Corruption Analysis

### The Problem

During the L2 overnight run (l2_overnight_500ep), we discovered that starting from inner_v002.md onwards, the Output_schema section contained an empty `proposed_changes` example:

```json
"proposed_changes": {}
```

This caused the L1 inner optimizer (the model reading this promptgram) to output empty `proposed_changes` objects, breaking the optimization loop since no parameter adjustments were being proposed.

### Root Cause: Mismatch Between Seed Files

**The corruption was NOT caused by an L2 edit.** Instead, it was caused by a mismatch between two seed sources:

1. **Hand-written seed (v001.md)**: Located at `/home/nuck/holoq/repositories/ripmap/training-outer/prompts/inner/v001.md`, this file contained:
   ```json
   "proposed_changes": {{}}
   ```
   The double braces `{{}}` were likely intended as a template placeholder, making it clear this was meant to be filled in.

2. **Rust baseline function**: The `baseline_promptgram()` function in `src/training_outer/promptgram.rs` (line 431) contains:
   ```rust
   "proposed_changes": {}
   ```
   This is valid JSON for an empty object but provides no hint that it should be filled with actual content.

### How The Corruption Propagated

1. **Episode 1**: The system started with v001.md (with `{{}}`)
2. **Episode 2**: When creating inner_v002, the system used the Rust `baseline_promptgram()` function (likely through a fork operation) rather than loading from v001.md
3. **Result**: The Output_schema section inherited `{}` instead of `{{}}`
4. **Propagation**: All subsequent versions (v002 through v051+) inherited this empty example through the fork chain

### Evidence Trail

- v001.md: Contains `"proposed_changes": {{}}`
- inner_v002.md onwards: All contain `"proposed_changes": {}`
- No L2 proposals in outer_scratchpad.json show edits to the Output_schema section (it's marked as immutable in the code)
- The corruption appears in inner_v002.toml line 57, which is the first TOML-serialized promptgram

### What We Learned

1. **Template ambiguity is dangerous**: The difference between `{}` (valid empty JSON) and `{{}}` (template placeholder) is subtle but critical for model behavior
2. **Seed file synchronization**: When you have multiple seed sources (hand-written markdown AND Rust code), they MUST be kept in sync
3. **Immutability is good but not foolproof**: While Output_schema was marked immutable (preventing L2 from editing it), the corruption happened at initialization, bypassing this protection
4. **Examples matter enormously**: LLMs are highly influenced by the examples in their prompts. An empty example teaches them to output empty results

### The Fix

The proper fix requires:

1. Update `baseline_promptgram()` in `src/training_outer/promptgram.rs` to include a filled example:
   ```rust
   "proposed_changes": {
     "pagerank_alpha": ["-", 0.15, "localize to suppress hub dominance"],
     "focus_expansion_boost": ["+", 2.0, "amplify adjacent signals"]
   }
   ```

2. Alternatively, use the double-brace notation `{{}}` in Rust and document that this is a template placeholder that should be rendered with a filled example when the promptgram is instantiated

3. Add validation that checks promptgrams for empty critical fields before serialization

### Implications for L2 Training

This corruption meant that for the entire overnight run (episodes 2-51+):
- The inner optimizer was trained by example to output empty proposed_changes
- No parameter tuning actually occurred at L1
- The optimization loop was essentially running in "observation-only" mode
- Any improvements in NDCG were likely due to L2's changes to Heuristics/Policy sections affecting the model's reasoning, not actual parameter adjustments

This is a profound lesson: **the schema example is not just documentation—it's a training example that shapes model behavior**.

---

## Agentic Mesa Mode Research

### Context

The training system currently uses single-shot prompting for mesa agents (Claude, Gemini, Codex):
- `claude --print -p "prompt"` - no tools, single shot
- `gemini -o text -y "prompt"` - auto-approve enabled but single shot
- `codex exec -` - already somewhat agentic but no explicit file access

We want agents to be able to:
1. Read files (scratchpad, previous episodes, other promptgrams)
2. Iterate on their reasoning (multi-turn exploration)
3. Potentially run lightweight experiments

**Full research document**: See `AGENTIC_MESA_RESEARCH.md` for comprehensive findings.

### Key Findings Summary

All three CLIs already support agentic/tool-using modes:

**Claude CLI (v2.0.61)**:
- `--tools <list>`: Enable specific tools (Read, Grep, Glob, Bash, Edit, Write)
- **Critical caveat**: `--tools` only works WITH `--print` flag (single-shot mode)
- For multi-turn: Drop `--print`, use `--output-format stream-json`, parse JSONL
- `--permission-mode bypassPermissions`: Auto-approve tool use
- `--add-dir <paths>`: Grant access to specific directories

**Gemini CLI (v0.19.2)**:
- Already agentic by default with `-y` flag!
- Just needs: `-o json` (instead of text) + `--include-directories <paths>`
- `--approval-mode yolo`: Explicit version of `-y`
- `--resume latest`: Multi-turn session resumption

**Codex CLI (v0.63.0)**:
- Already agentic with `exec` subcommand!
- Just needs: `--add-dir <paths>` for scratchpad access
- `--full-auto`: Safer alternative to `--dangerously-bypass-approvals-and-sandbox`
- `-o <file>`: Captures final output message

### Recommended Implementation Path

**Phase 1: Single-Shot with Read Access (MVP)**

Let agents read files in one turn before responding. Minimal code changes, immediate value.

**Claude**:
```rust
Command::new("claude")
    .args(["--print", "--tools", "Read,Grep,Glob"])
    .args(["--permission-mode", "bypassPermissions"])
    .args(["--add-dir", run_directory])
    .args(["-p", prompt])
```

**Gemini**:
```rust
Command::new("gemini")
    .args(["-o", "json"])  // Change from "text"
    .args(["--approval-mode", "yolo"])
    .args(["--include-directories", run_directory])
    .arg(prompt)
```

**Codex**:
```rust
Command::new("codex")
    .args(["exec", "--skip-git-repo-check", "--full-auto"])
    .args(["--add-dir", run_directory])
    .args(["-C", run_directory])
    .args(["-o", output_file, "-"])
```

**Benefits**:
- Agents can read `scratchpad.json` and past episodes before reasoning
- Still predictable single-turn execution
- Easy to extract JSON from output (same parsing as current)
- Low risk, high value

**Phase 2: Multi-Turn Exploration (Advanced)**

Allow full iterative reasoning with multiple file reads. Requires:
- Session management (create/resume/cleanup)
- JSONL parsing to extract final message
- Timeout handling (5-10 min max)
- More complex but enables deeper agent exploration

### Security Considerations

**Directory Restrictions**:
- Only whitelist run-specific directories
- Never allow home directory or system paths
- Use `--add-dir` / `--include-directories` carefully

**Tool Restrictions** (Claude):
- Safest: `--tools "Read,Grep,Glob"` (no shell, no writes)
- Avoid enabling Bash/Edit/Write for reasoning tasks

**Sandboxing** (Codex):
- Use `--sandbox workspace-write` instead of `danger-full-access`
- Use `--full-auto` instead of `--dangerously-bypass-approvals-and-sandbox`

**Timeouts**:
- Set max execution time per agent call (e.g., 5 minutes)
- Kill runaway processes to prevent infinite loops

### File Access Patterns

Agents will need access to:

```
/run_directory/
  scratchpad.json          # Main scratchpad with all episodes
  episodes/
    episode_001.json       # Individual episode details
    episode_002.json
  promptgrams/             # For L2/outer loop
    inner_v001.toml
    inner_v002.toml
  ground_truth/            # Optional
    cases.json
```

Update prompts to tell agents:
- "You have Read, Grep, Glob tools available"
- "Previous episodes: /run_dir/episodes/"
- "Scratchpad: /run_dir/scratchpad.json"
- "Use tools to explore past failures before proposing changes"

### Next Steps

1. Implement Phase 1 for all three agents (start with one, test thoroughly)
2. Update prompt templates to mention tool availability
3. Test with real training runs, compare reasoning quality
4. Monitor file access patterns and agent behavior
5. Consider Phase 2 only if single-shot feels limiting

### Open Questions

1. **Token costs**: Do tool-enabled conversations cost significantly more?
2. **Reliability**: Does JSON extraction remain consistent with tool use?
3. **Prompt engineering**: How should we instruct agents to use tools effectively?
4. **Caching**: Can we cache file contents across agent calls? (Claude supports prompt caching)
5. **Parallel execution**: Can we run multiple agentic episodes concurrently?

### Key Insight

**We don't need multi-turn iteration immediately.** Just letting agents READ the scratchpad and previous episodes before reasoning will be a massive improvement over blind single-shot prompting. Start simple (Phase 1), iterate based on observed behavior.

The most surprising finding: Gemini and Codex are already running in agentic mode! We just need to add file access and adjust output formats. Only Claude requires significant changes to enable tool use.
