# LLM-Based Mesa-Optimization: The Training Vision

> *"The optimizer is not gradient descent. The optimizer is Claude."*

## The Core Insight

ripmap is trained by **LLMs reasoning about why rankings fail**, not by numerical gradients. This is mesa-optimization where the inner optimizer is a language model.

```
┌─────────────────────────────────────────────────────────────────────┐
│                 Traditional Optimization                             │
├─────────────────────────────────────────────────────────────────────┤
│   Loss(θ) → ∂Loss/∂θ → θ ← θ - α∇Loss                              │
│                                                                      │
│   Problem: Blind. Knows THAT loss is high, not WHY.                 │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                 LLM Mesa-Optimization                                │
├─────────────────────────────────────────────────────────────────────┤
│   Failures → Claude/Gemini/Codex → "boost_chat too high drowns..."  │
│                    ↓                                                 │
│            {param: [direction, magnitude, rationale]}                │
│                                                                      │
│   Power: Semantic. Knows WHY rankings fail, proposes fixes.         │
└─────────────────────────────────────────────────────────────────────┘
```

The "gradient" emerges from **reasoning in concept space**. The LLM analyzes ranking failures and proposes changes with natural language rationale:

```json
{
  "strategy_capsule": "Testing whether lower alpha reduces distractor intrusion",
  "diagnosis": "High-PR distractors from unrelated modules flooding results",
  "proposed_changes": {
    "pagerank_alpha": ["decrease", "medium", "Reduce global spread"],
    "boost_focus_match": ["increase", "small", "Strengthen local signal"]
  },
  "confidence": 0.7
}
```

This is not gradient descent with extra steps. The LLM can propose **architectural changes** and **structural insights** that no gradient could express.

---

## The Two-Level Stack

```
L2 (Outer Loop): Gemini/Codex/Claude
│   Evolves the PROMPTGRAM that steers L1
│   Sees: L1 performance across episodes
│   Output: Prompt mutations (edit Policy, add Heuristics)
│
└─▶ L1 (Inner Loop): Claude/Gemini/Codex
      │   Proposes PARAMETER changes
      │   Sees: NDCG, failures, trajectory history
      │   Output: {param: [direction, magnitude, rationale]}
      │
      └─▶ L0: The code ranker being optimized
            17 parameters, NDCG metric
```

### L1: Parameter Optimizer

L1 is an LLM that observes ranking failures and proposes parameter adjustments:

- **Input**: Current parameters, ranking failures, trajectory history, accumulated insights
- **Output**: JSON with proposed changes and confidence
- **Agent options**: `--agent claude`, `--agent gemini`, `--agent codex`

L1 doesn't just optimize blindly—it **reasons about why** the system fails:

> "The depth penalty is too aggressive for monorepos. Files in `src/components/`
> are being suppressed even though they're central to the task. Recommend
> decreasing `depth_weight_deep` from 0.1 to 0.3."

### L2: Prompt Optimizer (Meta-Optimizer)

L2 observes L1's performance across multiple runs and evolves the **promptgram**—the structured prompt that steers L1's reasoning:

- **Input**: Recent L1 run summaries, promptgram performance stats, trajectory
- **Output**: Prompt edits (add heuristics, modify policy, change style)
- **Agent options**: `--agent-outer gemini`, `--agent-outer codex`

L2 mutations look like:

```json
{
  "mode": "explore",
  "edits": [{
    "section": "Heuristics",
    "edit_type": "append",
    "content": "Monorepo structures require inverting depth bias: penalize root-level noise.",
    "rationale": "L1 keeps boosting deep files but monorepos have valuable shallow modules"
  }],
  "hypothesis": "Adding monorepo-specific heuristic will reduce collapse events"
}
```

---

## The Promptgram: Prompts as Programs

A **promptgram** is not a blob of text but a structured program with independently-evolvable sections:

| Section | Purpose | Mutable by L2? |
|---------|---------|----------------|
| **Role** | Agent identity ("You approximate the gradient...") | ✗ Immutable |
| **Policy** | Decision logic for different trajectory states | ✓ |
| **Heuristics** | Domain knowledge accumulated over training | ✓ |
| **Style** | Output tone and presentation | ✓ |
| **API_contract** | Input specification | ✗ Protocol (injected at runtime) |
| **Output_schema** | JSON format for proposals | ✗ Protocol (injected at runtime) |

L2 can mutate Policy, Heuristics, and Style—but never the Role or Protocol. This prevents the meta-optimizer from breaking the communication contract while allowing it to improve the reasoning strategy.

**Critical architecture decision**: Protocol sections are **injected at runtime**, not stored in promptgram files. This prevents L2 from corrupting the output schema (which caused a production bug where L1 had great reasoning but empty `proposed_changes`).

---

## Why LLMs Work as Optimizers

### 1. Semantic Gradients

Numerical gradients know ∂Loss/∂θ but not why. LLM "gradients" are semantic:

```
Numerical: "NDCG went down when α increased"
Semantic:  "High α spreads PageRank globally, but for this debugging task
            we need local focus on the error-handling module"
```

### 2. Architecture Search

LLMs can propose changes beyond parameter tuning:

> "The multiplicative combination `boost × weight` can't express OR-logic.
> Consider adding an additive pathway for signals that should combine."

This is architecture search via natural language.

### 3. Trajectory Memory

L1 sees full training history, enabling pattern recognition:

> "α increased 3x over the last 5 episodes but NDCG dropped.
> We've overshot—recommend reverting toward 0.8."

### 4. Compositional Reasoning

Complex failure modes require understanding multiple interacting signals:

> "Low α + high boost_chat = tunnel vision because PageRank concentrates
> on chat files and the low damping prevents spreading to structural context."

No numerical gradient captures this interaction.

---

## The Training Protocol

### Phase 1: Inner Loop Training (L1)

```bash
./target/release/ripmap-train \
  --curated \
  --reason \
  --episodes 50 \
  --agent claude \
  --run-name my_training_run
```

This runs L1 optimization:
1. Evaluate current parameters on training cases
2. Collect ranking failures (NDCG < threshold)
3. Send failures to Claude for reasoning
4. Parse proposed changes
5. Apply changes, repeat

Outputs:
- `scratchpad.json`: Full reasoning history
- `results.trained.json`: Final optimized parameters
- `progress.png`: NDCG trajectory plot

### Phase 2: Outer Loop Training (L2)

```bash
./target/release/ripmap-train-outer my_l2_run \
  --steps-outer 10 \
  --episodes-inner 5 \
  --corpus curated \
  --agent-outer gemini \
  --agent-inner claude
```

This wraps L1:
1. Select a promptgram from population (explore/exploit)
2. Run K inner episodes with that promptgram
3. Summarize inner run performance
4. Invoke L2 (Gemini) to propose promptgram edits
5. Create new promptgram, add to population
6. Repeat

Outputs:
- `outer_scratchpad.json`: L2 trajectory and proposals
- `training-outer/prompts/inner/inner_v001.md`, `inner_v002.md`, ...: Evolved promptgrams

### Phase 3: Distillation (Wisdom Extraction)

After training, the scratchpad contains accumulated insights:

```bash
./target/release/ripmap-train --distill --scratchpad scratchpad.json
```

Crystallizes into:
- **Presets**: Named configurations for use cases
- **Heuristics**: Adaptive rules ("if monorepo: boost depth weight")
- **Warnings**: Failure mode alerts ("⚠ high focus_decay + low max_hops = tunnel vision")

---

## The Dissolved Decision Trees Connection

LLM mesa-optimization is only possible because of **dissolved parameters** (see `docs/8_DISSOLVED_DECISION_TREES.md`).

Traditional code has discrete branches:
```rust
match strategy {
    Strategy::Greedy => ...,
    Strategy::Exploratory => ...,
}
```

An LLM cannot propose "be 30% more greedy"—the branch is binary.

Dissolved code uses continuous coordinates:
```rust
let score = w_greedy * greedy_signal + w_exploratory * exploration_bonus;
```

Now the LLM can propose:
```json
{"w_greedy": ["increase", "small", "need more exploitation after plateau"]}
```

**Every trainable parameter must be continuous** for LLM optimization to work. The dissolved coordinate space is the search space that L1 explores via semantic reasoning.

---

## Agent Selection

Different LLMs have different strengths:

| Agent | Strengths | Use Case |
|-------|-----------|----------|
| **Claude** | Nuanced reasoning, good at multi-step analysis | L1 inner loop (default) |
| **Gemini** | Fast, good at pattern matching across runs | L2 outer loop |
| **Codex** | Exhaustive analysis, architectural thinking | Complex analysis, one-off deep dives |

Recommended combination: `--agent-inner claude --agent-outer gemini`

---

## Failure Modes and Diagnostics

### Basin Lock-in (L1)

**Symptom**: Same parameters for 5+ consecutive episodes

**Diagnosis**: L1 found local optimum, not exploring

**Fix**: L2 should add exploration heuristics; check scratchpad for diversity

```bash
jq '[.episodes[] | .proposed_changes | keys] | flatten | unique' scratchpad.json
# If few unique params touched, basin lock-in
```

### Schema Corruption (L1)

**Symptom**: Rich `.reasoning` but empty `.proposed_changes`

**Diagnosis**: Output schema example was corrupted by L2 mutation

**Fix**: Protocol separation (runtime injection prevents L2 from touching schema)

### L2 Not Steering

**Symptom**: L2 proposals look good but L1 ignores them

**Diagnosis**: L2 edits are prose, not constraints L1 can act on

**Fix**: L2 should edit Heuristics with specific param guidance

### Shadow Collapse (Bicameral Pipeline)

**Symptom**: Final graph has few edges, PageRank meaningless

**Diagnosis**: L1 suppressed name_match globally, killing shadow recall

**Fix**: Bicameral coordinates—separate shadow/final strategy weights

---

## Directory Structure

```
training/
├── runs/                        # L1 training runs
│   └── <run_name>/
│       ├── scratchpad.json      # Full reasoning history
│       ├── results.json         # Evaluation results
│       ├── results.trained.json # Final params
│       └── progress.png         # NDCG plot
│
└── prompts/
    └── protocol/
        └── inner_output_schema.md  # Immutable protocol (never L2-mutable)

training-outer/
├── runs/                        # L2 training runs
│   └── <run_name>/
│       ├── outer_scratchpad.json
│       └── inner_runs/
│           ├── step_001/        # Each L1 run under L2
│           ├── step_002/
│           └── ...
│
└── prompts/
    └── inner/                   # Evolved promptgrams
        ├── inner_v001.md        # Baseline
        ├── inner_v001.toml      # Metadata
        ├── inner_v002.md        # L2 mutation
        └── ...
```

---

## The Key Equations

### L1 Update Rule

```
θ_{t+1} = θ_t + LLM(failures, trajectory, scratchpad)
```

The "LLM" function returns `{param: [direction, magnitude, rationale]}`.

### L2 Update Rule

```
P_{t+1} = P_t + Edit(L2(summaries, promptgram_stats, trajectory))
```

The "L2" function returns prompt edits based on inner run analysis.

### The Meta-Loss

L2 optimizes for L1 performance:

```
argmax_P E[NDCG(θ* | θ* ~ L1(P))]
```

"Find the promptgram P such that L1 running with P produces the best expected final NDCG."

---

## What Makes This Work

1. **Semantic gradients**: LLMs understand *why*, not just *that*
2. **Dissolved coordinates**: Continuous parameter space enables "30% more X"
3. **Protocol separation**: L2 can't corrupt communication contract
4. **Scratchpad accumulation**: Learning compounds across episodes
5. **Warm-starting**: Each outer step builds on previous best params
6. **Agent diversity**: Different LLMs bring different reasoning strengths

---

## Summary

ripmap uses **LLMs as optimizers** in a two-level mesa-optimization stack:

- **L1** (Claude/Gemini/Codex) optimizes **parameters** based on ranking failures
- **L2** (Gemini/Codex) optimizes **prompts** based on L1 performance

This is not gradient descent with natural language. It's a fundamentally different optimization paradigm where the search happens in **concept space** via **semantic reasoning**.

The dissolved coordinate system (docs/8) provides the continuous search space. The L1/L2 architecture (docs/9) provides the optimization machinery. This document explains what those abstractions actually are: **LLMs reasoning about software behavior to make it better**.
