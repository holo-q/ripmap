# PageRank Implementation Summary

## Overview
Successfully implemented PageRank algorithm for file importance ranking in ripmap (Rust rewrite of grepmap).

**Location**: `/home/nuck/holoq/repositories/ripmap/src/ranking/pagerank.rs`

## Implementation Details

### Core Algorithm
The implementation uses the standard PageRank algorithm with depth-aware personalization:

```text
PR(v) = (1-α) * personalization[v] + α * Σ(PR(u) / out_degree[u])
                                         for all u pointing to v
```

Where:
- **α = 0.85** (damping factor from config)
- **personalization[v]** = normalized depth-aware weight for node v
- Handles **dangling nodes** (no outgoing edges) by redistributing their mass

### Key Features

1. **Graph Construction**
   - Nodes = files (identified by relative path)
   - Edges = symbol references (file A references symbol defined in file B)
   - Uses petgraph DiGraph for efficient graph operations

2. **Depth-Aware Personalization**
   The personalization vector biases the random walk based on file characteristics:

   | File Type | Weight | Description |
   |-----------|--------|-------------|
   | Root files (depth ≤ 2) | 1.0 | Highest priority |
   | Moderate depth (3-4) | 0.5 | Medium priority |
   | Deep files (5+) | 0.1 | Lower priority |
   | Vendor code | 0.01 | Heavily penalized |
   | Chat files | 100× multiplier | Current context boost |

3. **Power Iteration Convergence**
   - Maximum 100 iterations
   - Convergence threshold: ε = 1e-8
   - Preserves PageRank invariant: total ranks sum to 1.0

4. **Dangling Node Handling**
   Files with no outgoing references (leaf nodes) have their rank redistributed according to the personalization distribution, preventing rank loss.

## Struct API

### `PageRanker`
```rust
pub struct PageRanker {
    config: RankingConfig,
}

impl PageRanker {
    pub fn new(config: RankingConfig) -> Self

    pub fn compute_ranks(
        &self,
        tags_by_file: &HashMap<String, Vec<Tag>>,
        chat_fnames: &[String],
    ) -> HashMap<String, f64>
}
```

### Private Methods
- `build_defines_index()` - Creates symbol→files lookup
- `build_graph()` - Constructs petgraph DiGraph from tags
- `build_personalization()` - Computes depth-aware weights
- `personalization_weight()` - Calculates individual file weight
- `pagerank()` - Power iteration algorithm
- `extract_rel_fname()` - Path normalization helper

## Testing

### Test Coverage
Six comprehensive tests verify the implementation:

1. **test_simple_pagerank** - Basic graph ranking (3 nodes)
2. **test_depth_aware_personalization** - Verifies depth-based weights
3. **test_chat_file_boost** - Confirms 100× chat multiplier
4. **test_vendor_patterns** - Validates vendor code penalty
5. **test_empty_graph** - Edge case handling
6. **test_pagerank_convergence** - Verifies sum=1.0 and correct ordering

### Example Run
```bash
$ cargo run --example test_pagerank

PageRank results:
  c.rs: 0.212766
  a.rs: 0.574468  ← highest (referenced by both b and c)
  b.rs: 0.212766

Total rank: 1.000000 ✓
```

## Performance Characteristics

- **Time Complexity**: O(I × (V + E)) where:
  - I = iterations until convergence (typically < 20)
  - V = number of files
  - E = number of references

- **Space Complexity**: O(V + E) for graph storage

- **Optimizations**:
  - Uses petgraph for efficient graph operations
  - Single-pass dangling node calculation
  - Early convergence detection

## Integration Points

The PageRanker integrates with the broader ripmap ranking pipeline:

1. **Input**: Receives `tags_by_file` from extraction pipeline
2. **Output**: Returns `HashMap<String, f64>` of file→rank scores
3. **Config**: All hyperparameters tunable via `RankingConfig`

## Differences from Python Implementation

### Similarities
- Identical algorithm and formula
- Same personalization strategy
- Equivalent depth thresholds and weights

### Improvements
1. **Type Safety**: Rust's type system prevents many runtime errors
2. **Explicit Ownership**: Clear data flow without reference counting overhead
3. **Zero-Copy**: Uses `Arc<str>` for interned strings
4. **Performance**: Native code + petgraph optimizations
5. **Dangling Node Handling**: Explicitly redistributes dangling mass

## Configuration

All parameters are defined in `RankingConfig` (src/types.rs):

```rust
pub struct RankingConfig {
    // PageRank settings
    pub pagerank_alpha: f64,                // 0.85
    pub pagerank_chat_multiplier: f64,      // 100.0

    // Depth weights
    pub depth_weight_root: f64,             // 1.0
    pub depth_weight_moderate: f64,         // 0.5
    pub depth_weight_deep: f64,             // 0.1
    pub depth_weight_vendor: f64,           // 0.01
    pub depth_threshold_shallow: usize,     // 2
    pub depth_threshold_moderate: usize,    // 4

    // Vendor patterns
    pub vendor_patterns: Vec<String>,       // ["node_modules", "vendor", ...]
}
```

## Example Usage

```rust
use ripmap::ranking::PageRanker;
use ripmap::types::RankingConfig;

let config = RankingConfig::default();
let ranker = PageRanker::new(config);

let ranks = ranker.compute_ranks(&tags_by_file, &chat_files);

// ranks: HashMap<String, f64>
// - Keys: relative file paths
// - Values: importance scores (sum to 1.0)
```

## Future Enhancements

Potential optimizations for 1000× speed target:

1. **Parallel Power Iteration**: Use rayon for parallel rank updates
2. **Sparse Matrix**: Leverage CSR format for large graphs
3. **Approximate PageRank**: Use random walk sampling for massive graphs
4. **Incremental Updates**: Recompute only affected subgraph on file changes

## References

- Original Python implementation: `grepmapper-py/grepmap/ranking/pagerank.py`
- PageRank formula: Standard formulation with personalization
- Graph library: petgraph 0.6
- Inspired by: Google's original PageRank algorithm
