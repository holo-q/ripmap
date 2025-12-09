# ripmap

Codebase cartography for LLMs.

ripmap surfaces structurally significant code. It uses tree-sitter parsing,
PageRank on the symbol graph, git-aware temporal signals, and LLM-driven
parameter optimization.

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

  src/extraction/treesitter.rs
    class TreeSitterParser:
    def:
      extract_tags(...)
      get_language(...)
```

```console
$ ripmap --focus "cache" --tokens 2048
$ ripmap --calls --tokens 2048
```

## Installation

```console
$ cargo install --path .
```

## How It Works

ripmap is a recurrent graph neural network with 55 trainable coordinates.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Shadow Pass  ──▶  PageRank  ──▶  Policy Engine  ──▶  Final Pass        │
│  (high recall)     (importance)   (gating)           (high precision)   │
├──────────────────────────────────────────────────────────────────────────┤
│  55 Trainable Coordinates                                                │
│  ├── Shadow Strategy: name_match_weight, heuristic_confidence, ...      │
│  ├── Final Strategy:  type_hint_weight, import_weight, ...              │
│  └── Policy:          acceptance_bias, selection_temp, ...              │
└──────────────────────────────────────────────────────────────────────────┘
```

The pipeline has two hemispheres. The Shadow Pass casts a wide net via fuzzy
name matching. PageRank propagates importance through the symbol graph. The
Policy Engine decides when to stop exploring. The Final Pass verifies candidates
with LSP-style precision.

The coordinates control the physics:

- **PageRank damping** (α): How far importance spreads through the graph
- **Strategy weights**: Whether to trust name matching, type hints, or imports
- **Acceptance gates**: Sigmoid thresholds for candidate quality
- **Focus decay**: How quickly relevance fades from the query epicenter
- **Interaction mixing** (λ): Interpolates between OR-logic and AND-logic

Traditional code has discrete branches. ripmap dissolves these into continuous
coordinates. The system can be "30% more greedy" rather than flipping a switch.

The hypothesis: these 55 numbers are universal constants. A single configuration
governs navigation in Python, Rust, TypeScript, Go—any codebase.

## Training

The optimizer is not gradient descent. The optimizer is Claude.

```console
$ ripmap-train --curated --reason --episodes 100 --agent claude
```

ripmap is trained by LLMs reasoning about why rankings fail. The LLM observes
NDCG scores, analyzes failure cases, and proposes parameter adjustments with
natural language rationale:

```json
{
  "diagnosis": "High-PR distractors from unrelated modules flooding results",
  "proposed_changes": {
    "pagerank_alpha": ["decrease", "medium", "Reduce global spread"],
    "boost_focus_match": ["increase", "small", "Strengthen local signal"]
  },
  "confidence": 0.7
}
```

This is mesa-optimization. The "gradient" emerges from reasoning in concept
space. The LLM can propose changes that no numerical gradient could express:
"the multiplicative combination can't express OR-logic—add an additive pathway."

Ground truth comes from git history. Files that changed together in bugfix
commits have causal dependencies. ripmap learns to predict these relationships.

### Two-Level Stack

- **L1 (Inner Loop)**: Claude proposes parameter changes based on ranking failures
- **L2 (Outer Loop)**: Gemini evolves the prompt that steers L1's reasoning

L2 observes L1's performance across runs and mutates the promptgram—adding
heuristics, adjusting policy, changing reasoning style. The prompt is a program.

## Usage

```
ripmap [OPTIONS] [FILES]...

Options:
  -f, --focus <QUERY>    Semantic search across symbol names
  -t, --tokens <N>       Output token budget [default: 8192]
  -e, --ext <EXT>        Filter by file extension
  -v, --verbose          Show progress
      --calls            Show call graph relationships
      --git-weight       Boost recently changed files
      --join             Concatenate full file contents
```

## Configuration

Auto-detects from `ripmap.toml`, `pyproject.toml`, `tsconfig.json`, `Cargo.toml`, etc.

```toml
# ripmap.toml
include = ["src/**", "lib/**"]
exclude = ["**/generated/**"]
```

## Performance

| Stage | Time |
|-------|------|
| File discovery | ~50ms |
| Tag extraction (cached) | ~100ms |
| PageRank | ~10ms |
| **Total** | **~200ms** |

## Languages

Python, Rust, JavaScript/TypeScript, Go, Java, C/C++, Ruby, PHP.

## Documentation

- [LLM Mesa-Optimization](docs/10_LLM_MESA_OPTIMIZATION.md)
- [Dissolved Decision Trees](docs/8_DISSOLVED_DECISION_TREES.md)

## License

MIT
