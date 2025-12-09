# ripmap

Codebase cartography for LLMs.

ripmap learns to navigate code. Not by keyword matching, but by understanding
structural significance. It uses tree-sitter parsing, PageRank on the symbol
graph, git-aware temporal signals, and LLM-driven optimization to surface
what matters.

## The Vision

ripmap is not a search tool. It's a **Cognitive Mirror**—a machine that learns
the algorithm for finding relevant code, not just the code itself.

```
Traditional Search: query → pattern match → results
ripmap:             query → learned navigation policy → ranked context
```

Under the hood, ripmap is a **Recurrent Graph Neural Network** with ~55 trainable
coordinates. These aren't hyperparameters—they're the **Physics Constants of Code
Navigation**:

- **PageRank damping** (α): How far should importance spread?
- **Strategy weights**: Trust name matching vs. type hints vs. imports?
- **Acceptance gates**: When is a candidate "good enough"?
- **Focus decay**: How quickly does relevance fade from the query?

The hypothesis: these constants are **universal**. A single set of 55 numbers
governs navigation in Python, Rust, TypeScript, Go—any codebase.

## Example

```console
$ ripmap .
# Ranking: high (dense) | 1168 symbols | ~10728 tokens
  src/ranking/pagerank.rs
    class PageRanker:
    def:
      compute_ranks(...)
      build_graph(...)
      pagerank(...)
      personalization_weight(...)

  src/extraction/treesitter.rs
    class TreeSitterParser:
    def:
      extract_tags(...)
      get_language(...)
      supports_language(...)
```

Focus on specific areas:

```console
$ ripmap --focus "cache" --tokens 2048
# Ranking: high (dense) | 312 symbols | ~1847 tokens
  src/cache/store.rs
    class TagCache:
    class CacheEntry:
    def:
      get(...)
      set(...)
      is_valid(...)
      clear(...)
```

Show call relationships:

```console
$ ripmap --calls --tokens 2048
  src/ranking/pagerank.rs
    class PageRanker:
    def:
      compute_ranks
        ← called by: run, main
      build_graph
        → calls: build_defines_index(90%), extract_rel_fname(90%)
        ← called by: compute_ranks
```

## Installation

```console
$ cargo install --path .
```

Or build from source:

```console
$ cargo build --release
$ ./target/release/ripmap --help
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              COGNITIVE MIRROR                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Shadow    │───▶│  PageRank   │───▶│   Policy    │───▶│    Final    │  │
│  │    Pass     │    │   (GNN)     │    │   Engine    │    │    Pass     │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│        │                  │                  │                  │          │
│        ▼                  ▼                  ▼                  ▼          │
│   High Recall        Structural         Gating &          High Precision   │
│   Name Match         Importance         Wavefront         LSP-Verified     │
│                                                                             │
│  ════════════════════════════════════════════════════════════════════════  │
│                         55 Trainable Coordinates                            │
│  ════════════════════════════════════════════════════════════════════════  │
│                                                                             │
│   Shadow Strategy: name_match_weight, heuristic_confidence, ...            │
│   Final Strategy:  type_hint_weight, import_weight, same_file_weight, ...  │
│   Policy:          acceptance_bias, selection_temp, marginal_utility, ...  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Bicameral Pipeline

ripmap uses a two-hemisphere architecture inspired by how humans search:

1. **Shadow Pass** (Right Brain): Cast a wide net. High recall via fuzzy name
   matching. Seeds the graph with candidates.

2. **PageRank** (Corpus Callosum): Propagate importance through the symbol graph.
   Central APIs and load-bearing functions rise to the top.

3. **Policy Engine** (Prefrontal Cortex): Gating decisions. Should we continue
   exploring? Is this candidate worth keeping?

4. **Final Pass** (Left Brain): Verify with precision. LSP-style type checking,
   import validation. Prune false positives.

### Dissolved Decision Trees

Traditional code has discrete branches:
```rust
match strategy {
    Strategy::Greedy => ...,
    Strategy::Exploratory => ...,
}
```

ripmap "dissolves" these into continuous coordinates:
```rust
let score = λ * greedy_signal + (1-λ) * exploration_bonus;
```

This enables **Liquid Logic**—the system can be "30% more greedy" rather than
flipping a binary switch. Every decision becomes a tunable dial.

## Training

ripmap is trained by **LLMs reasoning about why rankings fail**. This is
mesa-optimization where the inner optimizer is Claude, Gemini, or Codex.

```
┌────────────────────────────────────────────────────────────────────┐
│                    LLM Mesa-Optimization                           │
├────────────────────────────────────────────────────────────────────┤
│  Failures → Claude → "boost_chat too high drowns structural..."   │
│                 ↓                                                  │
│         {param: [direction, magnitude, rationale]}                 │
│                                                                    │
│  Power: Semantic. Knows WHY rankings fail, proposes fixes.        │
└────────────────────────────────────────────────────────────────────┘
```

The "gradient" emerges from reasoning in concept space:

```json
{
  "diagnosis": "High-PR distractors from unrelated modules flooding results",
  "proposed_changes": {
    "pagerank_alpha": ["decrease", "medium", "Reduce global spread"],
    "boost_focus_match": ["increase", "small", "Strengthen local signal"]
  }
}
```

### Two-Level Training Stack

- **L1 (Inner Loop)**: Claude proposes parameter changes based on NDCG failures
- **L2 (Outer Loop)**: Gemini evolves the prompt that steers L1's reasoning

```console
$ ./target/release/ripmap-train \
    --curated \
    --reason \
    --episodes 100 \
    --agent claude
```

Ground truth comes from **git history**—files that changed together in bugfix
commits have causal dependencies that ripmap learns to predict.

## Usage

```
ripmap [OPTIONS] [FILES]...

Arguments:
  [FILES]...  Files or directories to focus on

Options:
  -f, --focus <QUERY>    Semantic search across symbol names
  -t, --tokens <N>       Output token budget [default: 8192]
  -e, --ext <EXT>        Filter by file extension (repeatable)
  -r, --root <PATH>      Project root [default: .]
  -v, --verbose          Show progress messages
      --stats            Print performance statistics
      --calls            Show call graph relationships
      --git-weight       Boost recently changed files
      --join             Concatenate full file contents
      --no-color         Disable ANSI colors
  -h, --help             Print help
  -V, --version          Print version
```

### Common patterns

```console
$ ripmap .                          # Map entire codebase
$ ripmap src/lib.rs                 # Focus on specific file
$ ripmap --focus "auth parser"      # Semantic search
$ ripmap --tokens 2048              # Quick overview
$ ripmap --tokens 32768             # Deep dive
$ ripmap -e rs                      # Rust files only
$ ripmap --join -e rs               # Full file contents
```

## Configuration

ripmap auto-detects your project's include/exclude patterns from existing tooling.
No config needed for most projects.

### Auto-detected config files

| File | What's extracted |
|------|------------------|
| `ripmap.toml` | Native config (preferred) |
| `pyproject.toml` | `[tool.ripmap]` → `[tool.ty]` → `[tool.ruff]` → ... |
| `tsconfig.json` | `include` / `exclude` |
| `biome.json` | `files.include` / `files.ignore` |
| `Cargo.toml` | `[workspace]` members |

### Native config (ripmap.toml)

```toml
include = ["src/**", "lib/**"]
exclude = ["**/generated/**"]
extend-exclude = ["**/vendor/**"]
src = ["src", "lib"]
```

## Performance

Designed for 1000x speedup over interpreted alternatives:

- Parallel file parsing via rayon
- Memory-mapped I/O for large files
- String interning for symbol names
- Persistent tag cache with redb

| Stage | Time |
|-------|------|
| File discovery | ~50ms |
| Tag extraction (cached) | ~100ms |
| PageRank | ~10ms |
| **Total** | **~200ms** |

## Supported Languages

Python, Rust, JavaScript/TypeScript, Go, Java, C/C++, Ruby, PHP.

Additional languages via tree-sitter grammar files in `queries/`.

## MCP Server

ripmap includes an MCP (Model Context Protocol) server for IDE integration:

```console
$ ripmap-mcp
```

## Documentation

- [LLM Mesa-Optimization](docs/10_LLM_MESA_OPTIMIZATION.md) - The training vision
- [Dissolved Decision Trees](docs/8_DISSOLVED_DECISION_TREES.md) - Continuous coordinates

## Related

- [tree-sitter](https://tree-sitter.github.io/tree-sitter/) - Parsing
- [ripgrep](https://github.com/BurntSushi/ripgrep) - Inspiration for CLI design
- [aider](https://github.com/paul-gauthier/aider) - The repomap concept

## License

MIT
