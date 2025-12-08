# Belief Propagation for Prompt Optimization

> Theoretical sketch for a generalization beyond strict inner/outer hierarchy.
> Current implementation uses hierarchical L0→L1→L2 stack. This doc captures
> a more general architecture for future exploration.

## The Hierarchy Assumption

Current system:
```
L3 → L2 → L1 → L0
     ↓     ↓     ↓
   prompt  prompt  θ
```

Control flows down, metrics flow up. This is **arbitrary** - just how we wired it.

## Bidirectional Message Passing

What if nodes could signal in both directions?

L1 discovers something L2 should know: "When you tell me to explore aggressively,
I do better if you also relax my confidence threshold." Currently L1 can't
communicate this upstream.

### Node Model

```rust
Node {
    id: NodeId,
    belief: Promptgram,  // or θ, or any config
    inbox: Vec<Signal>,
    outbox: Vec<Signal>,
}

Signal {
    from: NodeId,
    to: NodeId,
    kind: SignalKind,
    payload: Value,
}

enum SignalKind {
    Metric,      // "here's how I performed"
    Need,        // "I need X to decide Y"
    Suggestion,  // "consider changing Z"
    Fork,        // "I'm splitting into variants"
    Merge,       // "variants A and B should reunify"
}
```

### Update Cycle

```rust
fn step(graph: &mut Graph) {
    // 1. Deliver messages
    for node in &mut graph.nodes {
        for signal in node.outbox.drain(..) {
            graph.get_mut(signal.to).inbox.push(signal);
        }
    }

    // 2. Each node processes inbox, updates belief, emits signals
    for node in &mut graph.nodes {
        let (new_belief, outgoing) = node.belief.process(&node.inbox);
        node.belief = new_belief;
        node.inbox.clear();
        node.outbox = outgoing;
    }

    // 3. Handle forks
    for fork in graph.pending_forks.drain(..) {
        let variant = fork.source.fork(fork.divergence);
        graph.add(variant);
    }

    // 4. Prune dead nodes
    graph.retain(|n| !n.is_dead());
}
```

### Equilibrium

Iterate until no node wants to change given current signals.
Like belief propagation on a factor graph, or Nash equilibrium for beliefs.

## Forking and Population Dynamics

When a node sees multiple viable paths, it forks into variants.
Each variant runs, emits signals, and either:
- **Converges** back (one clearly wins)
- **Specializes** (explores different regions)
- **Dies** (metrics too bad)

Population structure emerges from optimization dynamics rather than being designed.

## Local Optimization Without Inner/Outer

The "level" isn't fixed - it emerges from causal structure. A node that
depends on another's output is "inner" to it, but the optimization mechanics
are symmetric. Every node does gradient-approximation in its own belief space.

## Relationship to Current System

Current implementation is a **constrained projection**:
- Fixed topology (strict levels)
- Unidirectional control flow
- Synchronous updates (outer waits for inner)

General case would have:
- Dynamic topology (forking/merging)
- Bidirectional signals
- Asynchronous updates

We might hit walls that are artifacts of the hierarchy, not fundamental limits.

## Related Concepts

- **Factor graphs / belief propagation** - same local message-passing structure
- **Distributed consensus (Raft)** - but for beliefs not logs
- **Evolutionary strategies** - population dynamics with selection pressure
- **Constitutional AI** - principles propagate through system
- **Mixture of experts** - nodes specialize based on what they're good at

## Open Questions

1. What signals should nodes emit beyond metrics?
2. How does a promptgram "process" signals into belief updates?
3. When does forking make sense vs just exploration?
4. How to prevent combinatorial explosion of variants?
5. What's the termination condition for equilibrium?

## Why This Matters

If the hierarchical system hits a wall, it might be because:
- L1 knows something L2 can't hear
- The topology prevents certain optimizations
- Synchronous updates waste opportunities

Understanding this generalization helps us recognize when to relax constraints.

---

*Not for immediate implementation. Captured here so the idea doesn't get lost.*
