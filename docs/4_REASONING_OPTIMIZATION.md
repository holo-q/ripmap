# Reasoning-Based Hyperparameter Training via Claude as Universal Function Approximator

> *"The gradient isn't in parameter space—it's in concept space."*

## Executive Summary

This document describes the architecture for training ripmap's ranking hyperparameters using Claude as a universal function approximator. Instead of classical optimization (grid search, Bayesian optimization) which treats the parameter space as a black box, we use **reasoning-based gradient descent** where Claude analyzes *why* rankings fail and proposes semantically-informed adjustments.

The key innovation is the **sidechain scratchpad**: an accumulating mental model that distills insights across training episodes into operator wisdom—presets, heuristics, warnings, and intuitions that teach the tool's use.

---

## The Paradigm Shift

### Classical Optimization
```
observe Loss(θ) → infer ∂Loss/∂θ → step θ
       ↑                              │
       └──────────────────────────────┘
       (black box: WHY is lost)
```

Classical methods (grid search, LHS, Bayesian) observe that a configuration has loss X, but don't understand *why*. They treat the 17-dimensional parameter space as an opaque surface to minimize.

### Reasoning-Based Training
```
observe Failure(θ) → reason about WHY → propose Δθ OR Δstructure
                           │
                           ├── "boost_chat too high drowns structural signal"
                           ├── "multiplicative combination can't express OR"
                           └── "missing: decay should vary by node type"
```

Claude observes ranking failures and reasons about their causes. The "gradient" emerges from semantic understanding—not just that NDCG is low, but *what signal was missing or overwhelming*.

This unlocks a qualitative capability: **architecture search via semantic gradient**. Claude can propose not just parameter adjustments, but structural changes to the ranking formula itself.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     THE TRAINING LOOP                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐                                                   │
│  │ Training     │  Extracted from git history via retrocausal      │
│  │ Cases        │  oracle: commits → co-changed files → relevance  │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐     ┌──────────────┐                             │
│  │ Parameter    │────▶│ Ranking      │  Simulate ripmap with       │
│  │ Point θ      │     │ Evaluation   │  current params             │
│  └──────────────┘     └──────┬───────┘                             │
│         ▲                    │                                      │
│         │                    ▼                                      │
│         │            ┌──────────────┐                               │
│         │            │ Failure      │  Cases where NDCG < threshold │
│         │            │ Collection   │  (ground truth not ranked)    │
│         │            └──────┬───────┘                               │
│         │                   │                                       │
│         │                   ▼                                       │
│         │            ┌──────────────┐     ┌──────────────┐         │
│         │            │ Claude       │────▶│ Reasoning    │         │
│         │            │ Reasoning    │     │ Episode      │         │
│         │            └──────────────┘     └──────┬───────┘         │
│         │                                        │                  │
│         │                   ┌────────────────────┼────────────┐    │
│         │                   │                    │            │    │
│         │                   ▼                    ▼            ▼    │
│         │            ┌───────────┐        ┌───────────┐ ┌────────┐│
│         └────────────│ Proposed  │        │ Param     │ │Struct. ││
│                      │ Δθ        │        │ Interact. │ │Insight ││
│                      └───────────┘        └─────┬─────┘ └───┬────┘│
│                                                 │           │      │
│                                                 ▼           ▼      │
│                                          ┌─────────────────────┐   │
│                                          │     SCRATCHPAD      │   │
│                                          │  (Sidechain Memory) │   │
│                                          └─────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼ (after N episodes)
┌─────────────────────────────────────────────────────────────────────┐
│                     DISTILLATION PHASE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Scratchpad Insights                                                │
│         │                                                           │
│         ├── cluster by theme ──────▶ PRESETS                       │
│         │     "monorepo_mode": {α: 0.8, boost_chat: 5, ...}        │
│         │     "debugging_mode": {reverse_edge_bias: 2.5, ...}      │
│         │                                                           │
│         ├── extract conditionals ──▶ ADAPTIVE HEURISTICS           │
│         │     if repo.file_count < 50: boost_chat *= 0.5           │
│         │     if query.is_symbol: boost_ident *= 1.5               │
│         │                                                           │
│         ├── crystallize warnings ──▶ OPERATOR INTUITIONS           │
│         │     "⚠ high focus_decay + low max_hops = tunnel vision"  │
│         │     "⚠ temporal_coupling useless without git history"    │
│         │                                                           │
│         └── generate prose ──▶ EMBEDDED WISDOM                     │
│               "When exploring unfamiliar code, prefer higher α     │
│                to let structure guide. When debugging known area,  │
│                lower α to anchor on your context."                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Structures

### Training Case (from Git Oracle)

```rust
/// A training case extracted from git history.
/// The retrocausal oracle: commits reveal what "should" change together.
pub struct TrainingCase {
    /// The seed file (starting point for ranking)
    pub seed_file: String,

    /// Files that should rank high (co-changed in commits)
    /// with coupling weights (Jaccard similarity of commit sets)
    pub expected_related: Vec<(String, f64)>,

    /// Inferred intent from commit message parsing
    pub intent: Intent,  // Debug, Refactor, Extend, Explore

    /// Quality weight for this case (bugfix=1.5x, WIP=0.2x)
    pub case_weight: f64,
}
```

### Ranking Failure

```rust
/// A case where the ranking system failed.
pub struct RankingFailure {
    /// The focus query used
    pub query: String,

    /// The seed file
    pub seed_file: String,

    /// Ground truth: files that should have ranked high
    pub expected_top: Vec<String>,

    /// What actually ranked high
    pub actual_top: Vec<String>,

    /// NDCG score (lower = worse)
    pub ndcg: f64,

    /// Commit context for semantic reasoning
    pub commit_context: String,

    /// Repository metadata
    pub repo_name: String,
    pub repo_file_count: usize,
}
```

### Reasoning Episode

```rust
/// One round of Claude reasoning about failures.
pub struct ReasoningEpisode {
    /// Failures analyzed
    pub failures: Vec<RankingFailure>,

    /// Parameters at time of failure
    pub params: ParameterPoint,

    /// Claude's full reasoning text
    pub reasoning: String,

    /// Proposed changes: param -> (direction, magnitude, rationale)
    pub proposed_changes: HashMap<String, (String, String, String)>,

    /// Insights about ranking structure (beyond params)
    pub structural_insights: Vec<String>,

    /// Confidence in proposals (0.0 - 1.0)
    pub confidence: f64,
}
```

### Scratchpad (Sidechain Memory)

```rust
/// Accumulated mental model across training episodes.
pub struct Scratchpad {
    /// All reasoning episodes
    pub episodes: Vec<ReasoningEpisode>,

    // === Crystallized Insights ===

    /// Discovered parameter interactions
    /// "low α + high boost_chat = tunnel vision"
    pub param_interactions: Vec<String>,

    /// Recurring failure patterns
    pub failure_patterns: Vec<String>,

    /// What worked well
    pub success_patterns: Vec<String>,

    /// Proposals beyond parameter tuning
    pub structural_proposals: Vec<String>,

    // === Distilled Wisdom ===

    /// Named configurations for use cases
    pub presets: HashMap<String, (ParameterPoint, String)>,

    /// Conditional adjustments
    pub heuristics: Vec<String>,

    /// Failure mode warnings
    pub warnings: Vec<String>,
}
```

---

## Claude Prompting Protocol

### Reasoning Prompt Template

```
You are a REASONING-BASED OPTIMIZER for a code ranking system (ripmap).

Your task is to analyze ranking failures and propose hyperparameter adjustments.
You are approximating the gradient in CONCEPT SPACE, not parameter space.

The question is not just "what values minimize loss" but "what structure would
make this failure class impossible?"

=== CURRENT PARAMETERS ===
{formatted_params}

=== RANKING FAILURES TO ANALYZE ===
{formatted_failures}

=== PRIOR KNOWLEDGE (from previous episodes) ===
{accumulated_insights}

=== YOUR TASK ===

1. DIAGNOSE: Why did each failure occur? What signal was missing or overwhelming?

2. REASON ABOUT INTERACTIONS: Do any parameter pairs interact to cause this?
   (e.g., "low α + high boost_chat = tunnel vision because...")

3. PROPOSE CHANGES: For each parameter that should change:
   - Direction: increase/decrease
   - Magnitude: small (10%), medium (30%), large (2x)
   - Rationale: why this addresses the failure

4. STRUCTURAL INSIGHTS: Patterns that can't be fixed by tuning?
   (e.g., "multiplicative combination can't express OR logic")

5. CONFIDENCE: How confident are you? (0-1)

=== OUTPUT FORMAT ===
Return JSON: {diagnosis, param_interactions, proposed_changes, structural_insights, confidence}
```

### Distillation Prompt Template

```
You are distilling optimization insights into operator wisdom for ripmap.

You have accumulated {N} reasoning episodes about the ranking system.

=== DISCOVERED PARAMETER INTERACTIONS ===
{interactions}

=== STRUCTURAL PROPOSALS ===
{proposals}

=== MOST COMMON CHANGES ===
{change_frequencies}

=== YOUR TASK ===

Distill into:
1. PRESETS: Named configurations for use cases
2. HEURISTICS: "if CONDITION: ADJUSTMENT - rationale"
3. WARNINGS: "⚠ what happens and why"
4. INTUITIONS: Prose wisdom teaching the tool's use

Return JSON with these fields.
```

---

## The 17-Dimensional Parameter Space

| Parameter | Default | Description | Layer |
|-----------|---------|-------------|-------|
| `pagerank_alpha` | 0.85 | Damping factor (structure vs personalization) | Global |
| `pagerank_chat_multiplier` | 100.0 | Weight boost for files in chat context | Global |
| `depth_weight_root` | 1.0 | Weight for root-level files | Global |
| `depth_weight_moderate` | 0.5 | Weight for mid-depth files | Global |
| `depth_weight_deep` | 0.1 | Weight for deeply nested files | Global |
| `depth_weight_vendor` | 0.01 | Weight for vendor/dependency files | Global |
| `boost_mentioned_ident` | 10.0 | Boost for query-mentioned identifiers | Short-range |
| `boost_mentioned_file` | 5.0 | Boost for query-mentioned files | Short-range |
| `boost_chat_file` | 20.0 | Boost for current chat context files | Short-range |
| `boost_temporal_coupling` | 3.0 | Boost for git co-change correlation | Medium-range |
| `boost_focus_expansion` | 5.0 | Boost for graph-expanded files | Medium-range |
| `git_recency_decay_days` | 30.0 | Half-life for recency weighting | Temporal |
| `git_recency_max_boost` | 10.0 | Maximum recency boost | Temporal |
| `git_churn_threshold` | 10.0 | Commits before "high churn" | Temporal |
| `git_churn_max_boost` | 5.0 | Maximum churn boost | Temporal |
| `focus_decay` | 0.5 | Decay per hop in BFS expansion | Expansion |
| `focus_max_hops` | 2.0 | Maximum expansion radius | Expansion |

### Known Interactions (from prior training)

From previous optimization runs, we've observed:

1. **α × boost_chat**: Low damping + high chat boost = tunnel vision
2. **depth_weight_deep × α**: Double penalty on nested code when both low
3. **temporal_coupling × git_recency**: Redundant in young repos
4. **focus_decay × max_hops**: Extreme values create blind spots

---

## Training Protocol

### Phase 1: Case Generation

```bash
# Extract training cases from curated repositories
ripmap-train --curated --extract-only --output cases.json
```

Uses the **retrocausal git oracle**:
- Parse commit history (last 500 commits per repo)
- Filter: 2-12 files per commit, skip WIP/merge
- Weight by commit type (bugfix=1.5x, feature=1.2x)
- Compute coupling weights via Jaccard similarity

### Phase 2: Failure Collection

```bash
# Evaluate current params and collect failures
ripmap-train --curated --params current.json --collect-failures failures.json
```

Failures are cases where:
- NDCG@10 < 0.5 (ground truth not in top 10)
- Expected files ranked below distractors

### Phase 3: Reasoning Episodes

```bash
# Run Claude reasoning on failures
ripmap-train --reason --failures failures.json --scratchpad scratchpad.json
```

For each batch of 5 failures:
1. Format as structured prompt
2. Include prior insights from scratchpad
3. Parse Claude's reasoning for proposed changes
4. Update scratchpad with new insights
5. Apply changes to parameters

### Phase 4: Distillation

```bash
# Distill accumulated insights into wisdom
ripmap-train --distill --scratchpad scratchpad.json --output wisdom.json
```

After N episodes, crystallize:
- Presets for common use cases
- Adaptive heuristics
- Operator warnings
- Embedded documentation

---

## Integration Points

### With Existing Benchmark Infrastructure

The reasoning module (`src/benchmark/reasoning.rs`) integrates with:

- `gridsearch.rs`: Uses `ParameterPoint` structure
- `metrics.rs`: Uses NDCG/MRR for failure detection
- `git_oracle.rs`: Uses extracted training cases
- `sensitivity.rs`: Can validate interaction hypotheses

### With Claude CLI

All Claude calls go through:
```rust
pub fn call_claude(prompt: &str) -> Result<String, String> {
    Command::new("claude")
        .args(["--print", "-p", prompt])
        .output()
}
```

No API keys needed—uses the user's authenticated Claude CLI session.

### With ripmap Core

Distilled presets can be embedded in:
- `src/types.rs`: `RankingConfig` presets
- `src/cli.rs`: `--preset` flag
- Documentation: operator guidance

---

## The Meta-Loop: Knowledge Accumulation

```
ripmap v1.0 (naive defaults)
      │
      ▼ train with reasoning

ripmap v1.1 (tuned params)
      │
      ▼ distill scratchpad

ripmap v1.2 (tuned params + embedded intuitions)
      │
      ▼ users encounter edge cases → new training

ripmap v1.3 (expanded presets + refined intuitions)
      │
      ...
```

Each version accumulates self-knowledge. The tool learns to predict its own failure modes and encodes that into its interface.

---

## Future Directions

### Unified Multi-Layer PageRank

Currently, temporal and semantic signals are applied *after* PageRank. A unified formulation would incorporate them *during* the random walk:

```
PR[v] = (1-α) × pers[v] + α × (
    β_ref × Σ_ref(PR[u]/deg[u]) +
    β_temp × Σ_temp(PR[u] × jaccard[u,v]) +
    β_sem × Σ_sem(PR[u] × stem_match[u,v])
)
```

### Compositional Focus Algebra

```
auth & parser          // Intersection
auth | parser          // Union
auth -> parser         // Path
auth @ 2               // Hop limit
auth % recent          // Filter
```

### Neural Ripmap

Replace hand-tuned parameters with learned embeddings:
```
symbol_embedding = TreeSitter_encoder(AST)
query_embedding = Transformer_encoder(query)
attention = softmax(query @ symbols.T / sqrt(d))
rank = attention × structural_prior
```

---

## Appendix: The Hopfield Interpretation

The ranking system can be viewed as an associative memory:

| Hopfield Concept | Ripmap Manifestation |
|-----------------|---------------------|
| Neurons | Files/symbols as nodes |
| Synaptic weights | Reference edges + temporal coupling |
| Stored patterns | Structural importance (PageRank eigenvector) |
| Probe/cue | Focus query |
| Pattern completion | Graph expansion from seeds |
| Energy function | Negative log-rank |
| Temperature | Damping factor α |
| Attractor basins | Contextual clusters |

The ranking score is inverse energy—high rank means low energy, stable attractor state.

---

## Summary

This system transforms hyperparameter optimization from blind search into **semantic reasoning**. Claude acts as a universal function approximator that understands *why* rankings fail, not just *that* they fail. The sidechain scratchpad accumulates this understanding into a theory of the parameter space that eventually crystallizes into operator wisdom.

The end state: a tool that knows itself—its strengths, failure modes, and how to guide users toward effective configurations.
