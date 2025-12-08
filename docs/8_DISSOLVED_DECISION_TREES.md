# Dissolved Decision Trees: The ripmap Design Philosophy

## Core Principle

Traditional software encodes decisions as discrete branches:

```rust
// Hard-coded decision tree
match strategy {
    Strategy::Greedy => do_greedy(),
    Strategy::Exploratory => do_exploratory(),
    Strategy::Balanced => do_balanced(),
}
```

ripmap **dissolves** these into continuous coordinate spaces:

```rust
// Dissolved: no branches, only weighted blends
let score = w_greedy * greedy_signal + w_exploratory * exploration_bonus;
let action = softmax(scores, temperature);
// w_*, temperature are trainable
```

The discrete tree structure melts into a smooth manifold. Training finds the optimal point in this space.

## Why This Matters

1. **Trainability**: Discrete choices create discontinuities. Continuous coordinates enable smooth optimization by L1/L2 loops.

2. **Discoverable structure**: The optimizer can find configurations humans wouldn't think to try. "Half greedy, half exploratory with high temperature" isn't a mode - it's just a point.

3. **No brittleness**: Hard `if/else` breaks when assumptions change. Weighted blends degrade gracefully.

4. **Composability**: New signals enter as new terms in the weighted sum. No architectural surgery.

## The Parameterization Pattern

Every discrete choice becomes a continuous coordinate:

| Discrete | Dissolved |
|----------|-----------|
| `if high_centrality then X` | `score += weight_centrality * centrality` |
| `if confidence > 0.8 then accept` | `score += weight_confidence * sigmoid(k * (conf - threshold))` |
| `choose strategy A or B` | `blend(A, B, mixing_ratio)` |
| `stop after N iterations` | `stop when marginal_value < floor` |

### The Softmax Trick

When multiple weights compete for probability mass, use softmax logits:

```rust
// BAD: Independent weights fight for equilibrium
pub spread_structural: f64,  // [0, 1]
pub spread_semantic: f64,    // [0, 1]
pub spread_spatial: f64,     // [0, 1]
// Optimizer wastes cycles balancing these

// GOOD: Logits normalize automatically
pub spread_logit_structural: f64,  // (-∞, +∞)
pub spread_logit_semantic: f64,
pub spread_logit_spatial: f64,
// softmax(logits) guarantees sum = 1.0
```

### The Lagrangian Trick

Don't hard-code resource budgets. Learn the "price":

```rust
// BAD: Arbitrary cap
pub query_budget: usize = 100;  // Why 100? Magic number.

// GOOD: Learned utility floor
pub marginal_utility_floor: f64;  // Stop when next query's value < this
// The optimizer discovers the budget
```

### Gene Expression (Structural Plasticity)

Even the combination FUNCTION can be a coordinate:

```rust
// Allow L2 to discover optimal structure
let additive = w_a * A + w_b * B;
let multiplicative = (A * B).ln();  // AND-logic
let combined = (1.0 - mixing) * additive + mixing * multiplicative;
// mixing ∈ [0, 1] is trainable
```

This lets L2 evolve from OR-logic (additive) to AND-logic (multiplicative) without code changes.

## The Training Substrate

The dissolved coordinates are the "genome" that L1/L2 optimize:

```
L1 (Inner Loop):
  - Sees: NDCG scores, ranking failures, trajectory
  - Proposes: {param: [direction, magnitude, rationale]}
  - Mechanism: Gradient-in-concept-space

L2 (Outer Loop):
  - Sees: L1 performance across episodes
  - Mutates: The PROMPT that steers L1 reasoning
  - Mechanism: Promptgram evolution
```

The coordinates form a smooth surface. L1 does local optimization. L2 does global search across prompt space.

## What To Dissolve

Candidates for dissolution:

- **Thresholds**: Any `if x > threshold` becomes sigmoid gating
- **Mode switches**: Any enum becomes weighted blend
- **Resource limits**: Any cap becomes marginal utility floor
- **Priority orderings**: Any "do A then B then C" becomes weighted scoring
- **Heuristic choices**: Any "use heuristic X" becomes strategy weight

What NOT to dissolve:

- **Physical constraints**: Can't blend "use LSP" with "don't use LSP" - it's either available or not
- **Data structure choices**: HashMap vs BTreeMap isn't a continuous parameter
- **Protocol contracts**: Output schema must be exact, not blended

## Implementation Checklist

When adding a new feature:

1. **Identify the decisions**: What discrete choices does this feature imply?
2. **Parameterize each one**: What continuous coordinate captures the same degree of freedom?
3. **Define the range**: What are sensible bounds? Log scale or linear?
4. **Add to the grid**: New coordinates enter `ParameterGrid` with `ParamRange`
5. **Wire to training**: L1 promptgram should mention new params in Heuristics section
6. **Default to neutral**: Initial values should be in the middle of the range, not at extremes

## Example: LSP Query Policy

Original discrete thinking:
```
1. Query hubs first
2. Then query uncertain edges
3. Stop at 40% budget
```

Dissolved:
```rust
pub struct LspPolicyCoordinates {
    pub weight_centrality: f64,       // How much to prefer hubs
    pub weight_uncertainty: f64,      // How much to prefer uncertain edges
    pub marginal_utility_floor: f64,  // When to stop (learned, not fixed)
    pub focus_temperature: f64,       // How tight the attention beam
}
```

The optimizer might find: "Actually, coherence matters more than centrality, stop earlier, use wider beam." This configuration wasn't in the original discrete design.

## The Meta-Lesson

The codebase is not a program. It's a **parameterized family of programs**. The code defines the shape of the solution space. Training selects which member of the family to instantiate.

Design for the family, not for any single member.
