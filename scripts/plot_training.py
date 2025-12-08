#!/usr/bin/env python3
"""
Visualize reasoning-based training progress.

Plots:
1. NDCG progression across episodes
2. Key parameter evolution
3. Failure count reduction
4. Confidence over time
"""

import json
import sys
from pathlib import Path

def load_scratchpad(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def extract_metrics(scratchpad: dict) -> dict:
    """Extract plottable metrics from scratchpad."""
    episodes = scratchpad.get("episodes", [])

    metrics = {
        "episode": [],
        "num_failures": [],
        "confidence": [],
        "ndcg": [],  # NDCG@10 before each episode
        # Key parameters to track
        "pagerank_alpha": [],
        "depth_weight_deep": [],
        "boost_temporal_coupling": [],
        "boost_focus_expansion": [],
        "focus_decay": [],
    }

    for i, ep in enumerate(episodes):
        metrics["episode"].append(i + 1)
        metrics["num_failures"].append(len(ep.get("failures", [])))
        metrics["confidence"].append(ep.get("confidence", 0))
        metrics["ndcg"].append(ep.get("ndcg_before", 0))

        params = ep.get("params", {})
        for key in ["pagerank_alpha", "depth_weight_deep", "boost_temporal_coupling",
                    "boost_focus_expansion", "focus_decay"]:
            metrics[key].append(params.get(key, 0))

    return metrics

def plot_ascii(metrics: dict):
    """Generate ASCII art charts."""
    episodes = metrics["episode"]
    n = len(episodes)

    if n == 0:
        print("No episodes to plot!")
        return

    print("\n" + "="*70)
    print("  REASONING-BASED TRAINING PROGRESS")
    print("="*70)

    # NDCG chart (if available)
    ndcg_values = [n for n in metrics["ndcg"] if n > 0]
    if ndcg_values:
        print("\n  NDCG@10 (higher = better)")
        print("  " + "-"*50)
        min_ndcg = min(ndcg_values)
        max_ndcg = max(ndcg_values)
        range_ndcg = max(max_ndcg - min_ndcg, 0.01)
        for i, (ep, ndcg) in enumerate(zip(episodes, metrics["ndcg"])):
            if ndcg > 0:
                bar_len = int(40 * (ndcg - min_ndcg) / range_ndcg)
                bar = "▓" * bar_len
                print(f"  E{ep:2d} │{bar:<40} {ndcg:.4f}")

    # Failure count chart
    print("\n  FAILURES PER EPISODE (fewer = better)")
    print("  " + "-"*50)
    max_fail = max(metrics["num_failures"]) if metrics["num_failures"] else 1
    for i, (ep, fails) in enumerate(zip(episodes, metrics["num_failures"])):
        bar_len = int(40 * fails / max_fail) if max_fail > 0 else 0
        bar = "█" * bar_len
        print(f"  E{ep:2d} │{bar:<40} {fails}")

    # Confidence chart
    print("\n  CLAUDE CONFIDENCE (higher = more certain)")
    print("  " + "-"*50)
    for i, (ep, conf) in enumerate(zip(episodes, metrics["confidence"])):
        bar_len = int(40 * conf)
        bar = "▓" * bar_len
        print(f"  E{ep:2d} │{bar:<40} {conf:.2f}")

    # Parameter evolution
    print("\n  KEY PARAMETER EVOLUTION")
    print("  " + "-"*50)

    params_to_show = [
        ("pagerank_alpha", "α (damping)"),
        ("depth_weight_deep", "depth_deep"),
        ("boost_temporal_coupling", "temporal"),
        ("boost_focus_expansion", "focus_exp"),
    ]

    for param_key, label in params_to_show:
        values = metrics[param_key]
        if not values:
            continue

        min_v, max_v = min(values), max(values)
        range_v = max_v - min_v if max_v != min_v else 1

        print(f"\n  {label}: {values[0]:.3f} → {values[-1]:.3f}")

        # Sparkline
        sparkline = ""
        chars = " ▁▂▃▄▅▆▇█"
        for v in values:
            idx = int(8 * (v - min_v) / range_v) if range_v > 0 else 4
            sparkline += chars[min(idx, 8)]
        print(f"  [{sparkline}]")

    # Summary stats
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    first_fails = metrics["num_failures"][0] if metrics["num_failures"] else 0
    last_fails = metrics["num_failures"][-1] if metrics["num_failures"] else 0
    print(f"  Episodes:        {n}")
    print(f"  Failures:        {first_fails} → {last_fails} ({last_fails - first_fails:+d})")
    print(f"  Final confidence: {metrics['confidence'][-1]:.2f}")
    print(f"  Structural insights: {len(scratchpad.get('structural_proposals', []))}")
    print("="*70 + "\n")

def plot_matplotlib(metrics: dict, output_path: str = None):
    """Generate matplotlib charts if available."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not available, using ASCII charts")
        return False

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Reasoning-Based Hyperparameter Training', fontsize=14, fontweight='bold')

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    episodes = metrics["episode"]

    # 1. Failures over time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(episodes, metrics["num_failures"], color='coral', alpha=0.7)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Failures')
    ax1.set_title('Ranking Failures per Episode')
    ax1.set_xticks(episodes)

    # 2. Confidence over time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(episodes, metrics["confidence"], 'o-', color='seagreen', linewidth=2, markersize=8)
    ax2.fill_between(episodes, metrics["confidence"], alpha=0.3, color='seagreen')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Confidence')
    ax2.set_title('Claude Confidence in Proposals')
    ax2.set_ylim(0, 1)
    ax2.set_xticks(episodes)

    # 3. PageRank alpha evolution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(episodes, metrics["pagerank_alpha"], 's-', color='royalblue', linewidth=2, markersize=8)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Alpha (α)')
    ax3.set_title('PageRank Damping Factor')
    ax3.set_xticks(episodes)
    ax3.axhline(y=0.85, color='gray', linestyle='--', alpha=0.5, label='Default')
    ax3.legend()

    # 4. Boost parameters
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(episodes, metrics["boost_temporal_coupling"], 'o-', label='Temporal Coupling', linewidth=2)
    ax4.plot(episodes, metrics["boost_focus_expansion"], 's-', label='Focus Expansion', linewidth=2)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Boost Multiplier')
    ax4.set_title('Boost Parameter Evolution')
    ax4.legend()
    ax4.set_xticks(episodes)

    # 5. Depth weights
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(episodes, metrics["depth_weight_deep"], 'd-', color='purple', linewidth=2, markersize=8)
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Weight')
    ax5.set_title('Deep File Weight (depth > 4)')
    ax5.set_xticks(episodes)
    ax5.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='Default')
    ax5.legend()

    # 6. Focus decay
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(episodes, metrics["focus_decay"], '^-', color='darkorange', linewidth=2, markersize=8)
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Decay Rate')
    ax6.set_title('Focus Expansion Decay')
    ax6.set_xticks(episodes)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved chart to {output_path}")
    else:
        plt.savefig('tmp/training_progress.png', dpi=150, bbox_inches='tight')
        print("Saved chart to tmp/training_progress.png")

    return True

if __name__ == "__main__":
    scratchpad_path = sys.argv[1] if len(sys.argv) > 1 else "tmp/scratchpad.json"

    if not Path(scratchpad_path).exists():
        print(f"Scratchpad not found: {scratchpad_path}")
        sys.exit(1)

    scratchpad = load_scratchpad(scratchpad_path)
    metrics = extract_metrics(scratchpad)

    # Always show ASCII
    plot_ascii(metrics)

    # Try matplotlib if available
    plot_matplotlib(metrics)
