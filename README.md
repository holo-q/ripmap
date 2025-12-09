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

The coordinates control:
- PageRank damping (α): importance spread
- Strategy weights: name matching vs. type hints vs. imports
- Acceptance gates: candidate thresholds
- Focus decay: query relevance falloff

These are hypothesized to be universal across languages and codebases.

## Training

Parameters are optimized by LLMs reasoning about ranking failures.

```console
$ ripmap-train --curated --reason --episodes 100 --agent claude
```

The LLM observes NDCG scores, analyzes failures, and proposes adjustments:

```json
{
  "diagnosis": "High-PR distractors flooding results",
  "proposed_changes": {
    "pagerank_alpha": ["decrease", "medium", "Reduce global spread"],
    "boost_focus_match": ["increase", "small", "Strengthen local signal"]
  }
}
```

Ground truth comes from git history—files changed together in bugfix commits.

Two-level optimization:
- L1: Claude proposes parameter changes
- L2: Gemini evolves the prompt that steers L1

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
