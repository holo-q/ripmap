# Structural Proposals

> Architecture tickets surfaced by reasoning-driven optimization.
> These are insights that **cannot** be addressed by parameter tuning alone.

## How This Works

During `--train` and `--distill`, Claude's reasoning about ranking failures sometimes
reveals fundamental limitations in the current architecture. These insights are
automatically appended here as structured proposals.

Each proposal follows a pattern:
1. **Evidence**: What failure patterns led to this insight?
2. **Limitation**: What can't the current system express?
3. **Proposal**: What architectural change would address it?
4. **Impact**: What would break/improve?

---

## Active Proposals

*None yet. Run training to surface architectural insights.*

---

## Template for Auto-Append

The distillation pipeline appends proposals in this format:

```markdown
### [CATEGORY] Title

**Episode**: #N | **Date**: YYYY-MM-DD

**Evidence**:
> Quoted reasoning from training scratchpad

**Limitation**:
What the multiplicative boost model cannot express.

**Proposal**:
Concrete architectural change.

**Impact**:
- Breaking: ...
- Improving: ...

**Status**: `proposed` | `investigating` | `accepted` | `rejected` | `implemented`
```

---

## Category Definitions

| Category | Meaning |
|----------|---------|
| `COMBINATION` | Boost combination model insufficient (multiplicative can't express OR, thresholds, etc.) |
| `TEMPORAL` | Time-aware ranking needs deeper integration (not just post-hoc multiply) |
| `GRAPH` | Call graph structure could inform PageRank differently |
| `FOCUS` | Focus expansion heuristics need rethinking |
| `RENDERING` | Output format limitations affecting ranking utility |

---

## Completed Proposals

*Archive of implemented structural changes, with links to commits.*

---

## Rejected Proposals

*Proposals that were investigated but deemed not worth the complexity.*

---

<!--
AUTO-APPEND MARKER: Do not remove this line.
The distillation pipeline appends new proposals above this marker.
-->
