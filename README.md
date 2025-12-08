# ripmap

Ultra-fast codebase cartography for LLMs.

ripmap reveals the load-bearing structure of your codebase. Not by size, but by
structural significance. It uses tree-sitter parsing, PageRank on the symbol
graph, and git-aware signals to surface what matters.

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

Map entire codebase:
```console
$ ripmap .
```

Focus on specific file:
```console
$ ripmap src/lib.rs
```

Semantic search:
```console
$ ripmap --focus "auth parser"
```

Quick overview (fewer tokens):
```console
$ ripmap --tokens 2048
```

Deep dive with signatures:
```console
$ ripmap --tokens 32768
```

Rust files only:
```console
$ ripmap -e rs
```

Full file contents for small projects:
```console
$ ripmap --join -e rs
```

## How it works

```
File Discovery → Tag Extraction → Graph Building → PageRank → Boosts → Rendering
      ↓              ↓                ↓              ↓          ↓          ↓
   ignore        tree-sitter      petgraph      iterative   contextual   ANSI
   crate          + .scm          DiGraph        power       signals     colors
```

1. **File Discovery**: Respects `.gitignore`, filters by extension
2. **Tag Extraction**: tree-sitter parses symbols (classes, functions, methods)
3. **Graph Building**: References between symbols become directed edges
4. **PageRank**: Iterative power method computes importance scores
5. **Contextual Boosts**: Focus query, git recency, call graph, temporal coupling
6. **Rendering**: Ranked output within token budget

### Why PageRank?

Functions that are called by many others rank higher. Entry points and central
APIs surface naturally. This mimics how humans navigate code: start at the
important nodes, follow references.

### Supported languages

- Python
- Rust
- JavaScript/TypeScript
- Go
- Java
- C/C++
- Ruby
- PHP

Additional languages via tree-sitter grammar files in `queries/`.

## Performance

Designed for 1000x speedup over interpreted alternatives:

- Parallel file parsing via rayon
- Memory-mapped I/O for large files
- String interning for symbol names
- Persistent tag cache with redb

Typical performance on medium codebases (~1000 files):

| Stage | Time |
|-------|------|
| File discovery | ~50ms |
| Tag extraction (cached) | ~100ms |
| PageRank | ~10ms |
| **Total** | **~200ms** |

First run builds cache; subsequent runs are near-instant.

## MCP Server

ripmap includes an MCP (Model Context Protocol) server for IDE integration:

```console
$ ripmap-mcp
```

## Architecture

```
src/
├── main.rs              # CLI entry point
├── lib.rs               # Public API
├── discovery/           # File finding
├── extraction/          # tree-sitter parsing
├── callgraph/           # Call relationship resolution
├── ranking/
│   ├── pagerank.rs      # Core ranking algorithm
│   ├── boosts.rs        # Contextual multipliers
│   ├── focus.rs         # Semantic search
│   └── git.rs           # Recency/churn signals
├── rendering/           # Terminal output
├── cache/               # Persistent tag storage
└── mcp/                 # MCP server
```

## Related

- [tree-sitter](https://tree-sitter.github.io/tree-sitter/) - Parsing
- [ripgrep](https://github.com/BurntSushi/ripgrep) - Inspiration for CLI design
- [aider](https://github.com/paul-gauthier/aider) - The repomap concept

## License

MIT
