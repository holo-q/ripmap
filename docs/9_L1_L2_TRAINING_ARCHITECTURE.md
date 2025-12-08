# L1/L2 Training Architecture

## Overview

ripmap uses a two-level optimization loop inspired by mesa-optimization:

```
L2 (Outer Loop) - Meta-optimizer
  │
  │  Evolves the PROMPTGRAM that steers L1
  │  Sees: L1 performance across many episodes
  │  Output: Mutated promptgram (Role/Policy/Heuristics/Style)
  │
  └─▶ L1 (Inner Loop) - Parameter optimizer
        │
        │  Proposes hyperparameter changes
        │  Sees: NDCG, failures, trajectory history
        │  Output: {param: [direction, magnitude, rationale]}
        │
        └─▶ Evaluation: NDCG on ranking quality
```

## L1: The Inner Loop

### What It Does

Given the current hyperparameters and ranking failures, L1 proposes changes:

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

### The Promptgram

L1's reasoning is shaped by a **promptgram** with sections:

| Section | Purpose | Evolvable? |
|---------|---------|------------|
| **Role** | Agent identity | ✓ (by L2) |
| **Policy** | Decision logic | ✓ (by L2) |
| **Heuristics** | Domain knowledge | ✓ (by L2) |
| **Style** | Output format | ✓ (by L2) |
| **API_contract** | Input specification | ✗ (protocol) |
| **Output_schema** | JSON format | ✗ (protocol) |

### Trajectory Awareness

L1 sees the full training history:
- NDCG scores across episodes
- Parameter values at each step
- Which changes helped vs hurt

This enables pattern recognition: "α increased 3x but NDCG dropped → α is too high"

## L2: The Outer Loop

### What It Does

L2 observes L1's performance across multiple inner runs and mutates the promptgram:

```json
{
  "proposal": {
    "edits": [
      {
        "section": "Heuristics",
        "edit_type": "add",
        "content": "Monorepo structures require inverting depth bias: penalize root-level noise."
      }
    ]
  },
  "rationale": "L1 keeps boosting deep files but monorepos have valuable shallow modules"
}
```

### What L2 CAN Mutate

- **Role**: Reframe L1's identity ("You are a gradient approximator" → "You are an exploration-exploitation balancer")
- **Policy**: Change decision rules ("If plateaued, try orthogonal move" → "If plateaued, check parameter ratios")
- **Heuristics**: Add/remove domain knowledge
- **Style**: Adjust verbosity, emphasis

### What L2 CANNOT Mutate

- **Protocol** (API_contract, Output_schema): These are injected at runtime, immutable
- **Available parameters**: The coordinate space is fixed; L2 can't invent new params
- **Evaluation metric**: NDCG is sacred

## The Protocol vs Promptgram Separation

**Critical architectural decision** discovered via debugging:

### The Bug

During a 500-episode run, L1 reasoning was perfect but `proposed_changes: {}` was always empty. NDCG plateaued at 0.88.

### Root Cause

The Output_schema in the promptgram showed an empty example:
```json
"proposed_changes": {}  // ← Model copied this literally
```

L2 had mutated the schema section, corrupting it.

### The Fix

**Protocol is immutable, injected at runtime:**

```
training/prompts/protocol/
└── inner_output_schema.md   ← Never mutated, always injected

training-outer/prompts/inner/
├── inner_v001.md            ← Only Role/Policy/Heuristics/Style
├── inner_v002.md
└── ...
```

L2 can evolve the reasoning prompt but cannot touch the output contract.

### Implementation

```rust
// In reasoning.rs
fn build_full_prompt(promptgram: &str) -> String {
    let protocol = include_str!("../prompts/protocol/inner_output_schema.md");
    format!("{}\n\n{}", promptgram, protocol)
}
```

## Failure Modes and Diagnostics

### Basin Lock-in

**Symptom**: Same params for 5+ consecutive outer steps

**Diagnosis**: L1 found a local optimum and L2 isn't providing enough exploration signal

**Check**:
```bash
jq '[.episodes[] | .final_params.pagerank_alpha] | unique' outer_scratchpad.json
# If length == 1, basin lock-in
```

**Fix**: L2 should add exploration heuristics; consider warm-start disabling

### Schema Corruption

**Symptom**: Rich `.reasoning` but empty `.proposed_changes`

**Diagnosis**: Output schema example was corrupted

**Check**:
```bash
jq '.episodes[-1].proposed_changes' inner_runs/step_034/scratchpad.json
# If {}, schema corruption
```

**Fix**: Protocol separation (see above)

### L2 Not Steering

**Symptom**: L2 proposals look good but L1 ignores them

**Diagnosis**: L2 edits are prose, not constraints L1 can act on

**Check**: Compare L2's edits to what L1 actually changes

**Fix**: L2 should edit Heuristics with specific param guidance, not vague advice

## Diagnostic Queries

```bash
# NDCG trajectory
jq '[.episodes[] | {step: .outer_step, ndcg: .final_metrics.ndcg}]' outer_scratchpad.json

# Param evolution
jq '[.episodes[] | {step: .outer_step, alpha: .final_params.pagerank_alpha}]' outer_scratchpad.json

# L2 edit types
jq '[.episodes[].proposal.edits[]?.edit_type] | group_by(.) | map({type: .[0], count: length})' outer_scratchpad.json

# Strategy capsules
jq '.episodes[] | "\(.outer_step): \(.strategy_capsules[0])"' outer_scratchpad.json
```

## Agentic Mode

L1/L2 agents can have file access to explore training history:

```bash
# Claude
claude --print --tools "Read,Glob,Grep" --add-dir <run_dir> -p "..."

# Gemini
gemini-cli --tool read_file_directly "..."

# Codex (already has full access)
codex exec "..."
```

This enables L1 to:
- Read previous scratchpads
- Examine failure patterns across runs
- Discover structural insights

## Training Runs

```bash
# L1 only (quick iteration)
./target/release/ripmap-train --curated --reason --episodes 50

# L2 outer loop
./target/release/ripmap-train-outer l2_run_name \
  --steps-outer 10 \
  --episodes-inner 5 \
  --corpus curated \
  --agent-outer gemini \
  --agent-inner claude
```

## Key Files

```
src/training/
├── reasoning.rs       # L1 agent calls, protocol injection
├── gridsearch.rs      # Parameter grid definition
├── metrics.rs         # NDCG calculation
└── mod.rs

src/training_outer/
├── mesa.rs            # L2 loop, promptgram mutation
├── promptgram.rs      # Promptgram structure, baseline
├── schemas.rs         # JSON schemas for proposals
└── mod.rs

training/prompts/
├── protocol/
│   └── inner_output_schema.md  # Immutable output contract
└── inner_default.md            # Default L1 promptgram

training-outer/prompts/inner/
├── inner_v001.md               # Baseline (auto-generated)
├── inner_v002.md               # L2 mutation
└── ...                         # Evolved variants
```

## Lessons Learned

1. **Protocol is sacred**: Output schemas must never be L2-mutable
2. **Examples are training data**: Schema examples shape model output literally
3. **Trajectory visibility matters**: L1 needs to see full history, not just current state
4. **Basin escape requires exploration signal**: L2 must inject novelty, not just optimize
5. **Warm-starting can trap**: Sometimes fresh start beats warm-start from local optimum
