# LSP-Native Ripmap Architecture

## The Regime Shift

ripmap currently operates in a **heuristic-limited regime**: 14% call graph resolution via name matching, type hints, and import analysis. This creates a sparse, uncertain graph where PageRank flows through "maybe" connections.

With LSP integration (`ty server` for Python), we shift to an **active sensing regime**: strategically "buy" truth with latency. Resolution rates jump to 80%+, fundamentally changing the optimization surface.

**Critical insight**: Hyperparameters optimal for 14% resolution are WRONG for 80% resolution. This is a phase transition, not a gradual improvement.

## The Dissolved Decision Tree Philosophy

Every feature must enter as continuous coordinates, not categorical modes:

```rust
// DON'T: Hard fork creates fragile, untrainable system
if strategy == "greedy" { use_greedy(); }
else if strategy == "exploratory" { use_exploratory(); }

// DO: Continuous blend, trainable by L1/L2
let score = w_centrality * x + w_uncertainty * y + w_coherence * z;
let action = softmax(scores, temperature);
// w_*, temperature are trainable coordinates
```

The codebase becomes a **parameterized family of algorithms**. Training selects which member to instantiate. No discrete choices - only continuous coordinates in a smooth manifold.

## The LSP Query Problem

Each LSP query resolves a call edge with high confidence (~0.95). But:
- **Not all edges are equal**: Hubs matter more than leaves
- **Queries have diminishing returns**: Resolving a type informs all its method calls
- **The topology has curvature**: Dense clusters vs sparse bridges
- **Type inference is a DAG**: Can't resolve `x.foo()` until you know what `x` is

### The Coordinate System

```rust
pub struct LspPolicyCoordinates {
    // === Resource Physics ===
    /// The "price" of a query. Stop when ExpectedGain < this.
    /// Replaces crude budget_ratio with Lagrangian multiplier.
    pub marginal_utility_floor: f64,  // [0.001, 0.1]

    /// Latency trade-off: 0.0=Sequential (smart), 1.0=Batch (fast)
    /// Controls number of wavefronts (1-3)
    pub batch_latency_bias: f64,      // [0.0, 1.0]

    // === Signal Weights (The "Desire" Vector) ===
    pub weight_centrality: f64,       // PageRank score
    pub weight_uncertainty: f64,      // 1.0 - heuristic_confidence
    pub weight_coherence: f64,        // Group gain: shared receiver count
    pub weight_causality: f64,        // Root vs Leaf preference (DAG depth)
    pub weight_bridge: f64,           // Betweenness centrality proxy

    // === Attention Beam Direction ===
    // Softmax logits for simplex normalization (sum to 1.0)
    // Prevents "dimensionality fighting" during training
    pub spread_logit_structural: f64, // Follow call edges
    pub spread_logit_semantic: f64,   // Follow type hierarchy
    pub spread_logit_spatial: f64,    // Follow file locality

    // === Navigation ===
    pub focus_temperature: f64,       // Entropy of sampling (low = tight beam)
    pub gated_threshold: f64,         // Stochastic dropout for performance

    // === Structural Plasticity ("Gene Expression") ===
    /// Mixing: 0.0 = Additive (OR-logic), 1.0 = Multiplicative (AND-logic)
    /// Lets L2 discover optimal combination structure
    pub interaction_mixing: f64,      // [0.0, 1.0]
}
```

### Key Coordinate Insights

1. **Simplex Constraint**: `spread_*` params are NOT independent. They compete for probability mass. Use softmax logits to create unconstrained training surface.

2. **Marginal Utility > Budget**: Don't cap at "40% of queries". Instead, learn a "price" and stop when expected gain drops below price. The optimizer finds the budget.

3. **Coherence (Group Gain)**: Resolving `user: User` instantly resolves `user.name`, `user.id`, `user.email`. One query buys N edges. This axis captures fan-out potential.

4. **Causality (DAG Depth)**: Type inference flows from definitions to usages. You can't resolve `factory.create().process()` until you resolve `factory`. Prefer roots over leaves.

5. **Gene Expression**: The `interaction_mixing` parameter lets L2 discover whether optimal policy is additive (high centrality OR high uncertainty) vs multiplicative (high centrality AND high uncertainty). Structure becomes a coordinate.

## The Wavefront Execution Model

Pure batch ignores sequential information gain. Pure sequential is too slow. The compromise: **Generational Wavefront**.

```
Gen 1 (The Spine):
  Query high-centrality, high-causality roots (variables, imports, definitions)
  → Type cache populates
  → Many downstream edges resolve automatically

Gen 2 (The Frontier):
  Re-score remaining entropy with new type information
  → Query newly exposed ambiguity
  → Focus on edges that Gen 1 couldn't resolve

Gen 3 (The Fill):
  Surgical strikes on specific high-rank nodes
  → Clean up remaining uncertainty
  → Stop when marginal_utility_floor reached
```

The `batch_latency_bias` coordinate controls whether we do 1 wave (fast, dumb) or 3 waves (smart, slower). Trainable.

## The Oracle Bootstrap Training Protocol

The shift from 14% to 80% resolution is a regime change. Don't train in the wrong regime and hope it transfers.

### Phase 1: The Oracle Run (Offline)
```
1. Run `ty` on entire training corpus
2. Build the "Perfect Graph" (100% resolution)
3. Train DOWNSTREAM parameters on Perfect Graph:
   - pagerank_alpha
   - boost_caller_weight
   - focus_expansion_*
   - etc.
4. Result: "Golden Weights" - optimal for high-resolution regime
```

### Phase 2: Policy Distillation
```
1. Freeze Golden Weights
2. Train LSP POLICY coordinates:
   - marginal_utility_floor
   - weight_* (signal priorities)
   - spread_logit_* (attention direction)
   - etc.
3. Objective: Minimize KL(Rank_policy || Rank_oracle) subject to latency
4. Result: Policy that approximates Perfect Graph with 20% of queries
```

### Phase 3: Joint Fine-Tuning
```
1. Unfreeze everything
2. End-to-end polish
3. Small learning rate
4. Result: Globally optimized system
```

## Information-Theoretic Framing

The problem is **Active Learning on a Graph**.

Objective: Maximize **Reduction in Rank Entropy** per unit of **Latency**.

```
J(θ) = E[ ΔNDCG(Graph_post) - λ·Cost(Q) ]
```

### The Proxy Metric: Weighted Edge Collapse

Since NDCG is expensive, optimize for edge entropy reduction:

```
Every unresolved edge has entropy H(e):
  - NameMatch: ~0.4 (uncertain)
  - TypeHint:  ~0.2 (moderate)
  - LSP:       ~0.0 (certain)

Value of collapsing edge e:
  V(e) = H(e) × Centrality(e)

Policy maximizes ΣV(e) subject to query budget.
```

## Implementation Architecture

```
src/lsp/
├── mod.rs           # Module exports
├── coordinates.rs   # LspPolicyCoordinates struct
├── client.rs        # ty server process management, JSON-RPC
├── cache.rs         # Type cache with coherence tracking
├── policy.rs        # PolicyEngine: score_site(), select_wavefront()
└── integration.rs   # Wire into callgraph resolver

src/callgraph/
├── resolver.rs      # Add build_graph_with_lsp()
└── strategies.rs    # LspStrategy at highest priority
```

### Client Architecture

```rust
pub struct LspClient {
    process: Mutex<Option<Child>>,      // ty server process
    cache: DashMap<(File, Line, Col), TypeInfo>,  // Results cache
    pending: DashMap<RequestId, Sender>,          // In-flight requests
}

impl LspClient {
    fn ensure_started(&self) -> Result<()>;
    fn resolve_batch(&self, queries: &[(File, Line, Col)]) -> Vec<TypeInfo>;
    fn hover(&self, file: &str, line: u32, col: u32) -> Option<TypeInfo>;
    fn definition(&self, file: &str, line: u32, col: u32) -> Option<Location>;
}
```

### Policy Engine

```rust
pub struct PolicyEngine {
    coords: LspPolicyCoordinates,
}

impl PolicyEngine {
    /// Calculate "energy" (desire to query) of a call site
    fn score_site(&self, tag: &Tag, pagerank: f64, conf: f64, coherence: f64) -> f64 {
        let additive =
            self.coords.weight_centrality * pagerank.ln().max(-10.0) +
            self.coords.weight_uncertainty * (1.0 - conf) +
            self.coords.weight_coherence * coherence.ln_1p() +
            self.coords.weight_causality * is_root(tag);

        // Gene expression: blend additive and multiplicative
        let multiplicative = (pagerank * (1.0 - conf) * (1.0 + coherence)).ln();
        let t = self.coords.interaction_mixing;
        (1.0 - t) * additive + t * multiplicative
    }

    /// Select wavefront using gated stochastic attention
    fn select_wavefront(&self, candidates: Vec<Candidate>) -> Vec<&Tag>;
}
```

## Integration Points

### With Existing Training (L1)

The LSP coordinates become part of the hyperparameter grid:

```rust
ParameterGrid::default()
    // Existing downstream params
    .with("pagerank_alpha", ParamRange::linear(0.05, 0.95))
    .with("boost_caller_weight", ParamRange::linear(0.0, 5.0))
    // NEW: LSP policy coordinates
    .with("lsp_marginal_utility_floor", ParamRange::log(0.001, 0.1))
    .with("lsp_weight_centrality", ParamRange::linear(0.0, 2.0))
    .with("lsp_weight_coherence", ParamRange::linear(0.0, 2.0))
    .with("lsp_interaction_mixing", ParamRange::linear(0.0, 1.0))
```

### With L2 Meta-Learning

L2 can evolve the policy structure by tuning `interaction_mixing`:
- Discover whether additive or multiplicative combination works better
- Without recompilation - pure coordinate space

L2 CANNOT mutate the coordinate system itself (that's protocol). But it CAN discover optimal regions within the space.

## Graceful Degradation

When `ty` is unavailable:
1. LSP coordinates have no effect
2. Fall back to heuristic-only resolution
3. System behaves exactly as current (14% resolution)

The `marginal_utility_floor` naturally handles this: if LSP returns nothing, expected gain is 0, no queries are made.

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Resolution rate | 14% | 80%+ |
| NDCG | 0.88 | 0.92+ |
| Latency (cold) | ~50ms | ~500ms |
| Latency (warm cache) | ~50ms | ~60ms |
| Query efficiency | N/A | 40% budget → 95% NDCG |

## Open Questions

1. **Multi-language**: `ty` is Python-only. Rust has `rust-analyzer`, TS has `tsserver`. Should we abstract the LSP client?

2. **Incremental updates**: When a file changes, which cache entries invalidate?

3. **L2 boundary**: Should L2 be able to evolve the wavefront structure (number of generations), or is that protocol?

---

*This document captures the architectural vision from the 2024-12-08 deep analysis session. Implementation should follow the Oracle Bootstrap protocol: build perfect graph first, train downstream, then train policy.*
