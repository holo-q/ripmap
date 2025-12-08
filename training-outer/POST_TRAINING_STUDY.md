# Post-Training Study Guide

Quick reference for diagnosing L1/L2 training dynamics.

## Key Files to Examine

```
training-outer/runs/{run_name}/
├── run.log                    # High-level progress, NDCG trajectory
├── outer_scratchpad.json      # Full L2 history: proposals, edits, metrics
└── inner_runs/step_NNN/
    ├── scratchpad.json        # L1 episode history for this step
    └── results.trained.json   # Final params from this step
```

## First Questions

1. **Did NDCG actually improve?**
   - Check `outer_scratchpad.json` → episodes[].final_metrics.ndcg
   - Plot trajectory: first → best → last

2. **Did params actually change?**
   ```bash
   jq '[.episodes[] | {step: .outer_step, alpha: .final_params.pagerank_alpha, ...}]' outer_scratchpad.json
   ```
   - If params are identical across steps → L1 is stuck in a basin

3. **Are proposed_changes being populated?**
   ```bash
   jq '.episodes[-1].proposed_changes' inner_runs/step_034/scratchpad.json
   ```
   - Empty `{}` = model isn't outputting structured changes (schema issue)
   - Filled = model is proposing but changes aren't being applied

## Common Failure Modes

**Basin Lock-in**
- Symptom: Same params for 5+ consecutive outer steps
- Diagnosis: L1 keeps converging to same local optimum regardless of promptgram
- Check: Compare `.final_params` across steps - are they identical?

**Schema Corruption**
- Symptom: Rich reasoning in `.reasoning` but empty `proposed_changes`
- Diagnosis: Output schema example shows `{}` instead of filled example
- Check: Look at the promptgram's Output_schema section (now fixed - protocol is injected)

**Warm-start Dependency**
- Symptom: Only steps that DON'T warm-start show param movement
- Diagnosis: Warm-starting locks into previous basin
- Check: `.warm_started_from_step` field - null = fresh start

**L2 Not Steering**
- Symptom: L2 proposals look good but L1 ignores them
- Diagnosis: L2 edits are prose, not param constraints
- Check: Compare L2's edits to what L1 actually changes

## Useful jq Queries

```bash
# NDCG trajectory
jq '[.episodes[] | {step: .outer_step, ndcg: .final_metrics.ndcg}]' outer_scratchpad.json

# Param evolution (pick your params)
jq '[.episodes[] | {step: .outer_step, alpha: .final_params.pagerank_alpha, temporal: .final_params.boost_temporal_coupling}]' outer_scratchpad.json

# Steps with actual param changes (compare adjacent)
jq '[.episodes[] | .final_params.pagerank_alpha] | [., .[1:]] | transpose | map(select(.[0] != .[1]))' outer_scratchpad.json

# L2 edit types
jq '[.episodes[].proposal.edits[]?.edit_type] | group_by(.) | map({type: .[0], count: length})' outer_scratchpad.json

# Strategy capsules (the "why")
jq '.episodes[] | "\(.outer_step): \(.strategy_capsules[0])"' outer_scratchpad.json
```

## Reading the Inner Scratchpad

Each L1 episode has:
- `.reasoning` - free-form analysis (usually correct)
- `.proposed_changes` - structured output (check if populated!)
- `.strategy_capsule` - 1-line intent
- `.structural_insights` - beyond-tuning observations

If reasoning is rich but proposed_changes empty → the model understands but isn't outputting correctly.

## Key Insight from 500ep Run

The model correctly identified:
> "NDCG has been oscillating between 0.878-0.882 for 7 episodes (plateau)"
> "Fix depth gradient inversion: root=0.45, moderate=0.30, deep=0.15"

But `proposed_changes: {}` was empty because the schema example showed empty.

**Lesson**: Schema examples aren't just documentation - they're training data that shapes model behavior.
