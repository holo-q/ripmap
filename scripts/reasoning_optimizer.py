#!/usr/bin/env python3
"""
Reasoning-Based Hyperparameter Optimization via Claude as Universal Function Approximator.

The paradigm shift:
- Classical: observe Loss(θ) → infer ∂Loss/∂θ → step θ (black box: WHY is lost)
- Reasoning: observe Failure(θ) → reason about WHY → propose Δθ OR Δstructure

The gradient isn't in parameter space—it's in concept space.

The sidechain scratchpad accumulates insights across optimization episodes,
building a theory of the hyperparameter manifold that predicts:
- "If you're in situation X, parameter Y will fail because Z"
- "When A is high, B must compensate or you'll see symptom C"

This theory gets distilled into:
- PRESETS: clustered insights → named configurations
- ADAPTIVE HEURISTICS: conditionals extracted from patterns
- OPERATOR INTUITIONS: crystallized warnings and wisdom
- EMBEDDED PROMPT: prose that teaches the tool's use

Usage:
    python scripts/reasoning_optimizer.py --failures failures.json --scratchpad scratchpad.md
    python scripts/reasoning_optimizer.py --distill scratchpad.md --output intuitions.md
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import re


@dataclass
class ParameterPoint:
    """Current hyperparameter configuration."""
    pagerank_alpha: float = 0.85
    pagerank_chat_multiplier: float = 100.0
    depth_weight_root: float = 1.0
    depth_weight_moderate: float = 0.5
    depth_weight_deep: float = 0.1
    depth_weight_vendor: float = 0.01
    boost_mentioned_ident: float = 10.0
    boost_mentioned_file: float = 5.0
    boost_chat_file: float = 20.0
    boost_temporal_coupling: float = 3.0
    boost_focus_expansion: float = 5.0
    git_recency_decay_days: float = 30.0
    git_recency_max_boost: float = 10.0
    git_churn_threshold: float = 10.0
    git_churn_max_boost: float = 5.0
    focus_decay: float = 0.5
    focus_max_hops: float = 2.0


@dataclass
class RankingFailure:
    """A case where the ranking system failed."""
    query: str
    seed_file: str
    expected_top: list[str]  # Files that should have ranked high
    actual_top: list[str]    # Files that actually ranked high
    ndcg: float              # How bad was the failure
    commit_message: str      # Context about what was happening
    repo_context: dict       # Metadata about the repo


@dataclass
class ReasoningEpisode:
    """One round of reasoning about failures."""
    failures: list[RankingFailure]
    current_params: ParameterPoint
    reasoning: str           # Claude's analysis
    proposed_changes: dict   # {param: (direction, magnitude, rationale)}
    structural_insights: list[str]  # Higher-level observations
    confidence: float        # How confident in the proposal


@dataclass
class Scratchpad:
    """Accumulated mental model across optimization episodes."""
    episodes: list[ReasoningEpisode] = field(default_factory=list)

    # Crystallized insights
    param_interactions: list[str] = field(default_factory=list)
    failure_patterns: list[str] = field(default_factory=list)
    success_patterns: list[str] = field(default_factory=list)
    structural_proposals: list[str] = field(default_factory=list)

    # Distilled wisdom
    presets: dict = field(default_factory=dict)
    heuristics: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def call_claude(prompt: str, timeout: int = 60) -> str:
    """Call Claude CLI and return response."""
    try:
        result = subprocess.run(
            ["claude", "--print", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode != 0:
            print(f"Warning: claude error: {result.stderr[:200]}")
            return ""
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print("Warning: claude timed out")
        return ""
    except Exception as e:
        print(f"Warning: Error calling claude: {e}")
        return ""


def reason_about_failures(
    failures: list[RankingFailure],
    current_params: ParameterPoint,
    scratchpad: Scratchpad
) -> ReasoningEpisode:
    """
    The core reasoning loop: analyze failures and propose parameter changes.

    This is where Claude acts as a universal function approximator:
    f(failures, params, history) -> (reasoning, proposals, insights)
    """

    # Format failures for Claude
    failure_desc = "\n\n".join([
        f"""FAILURE {i+1}:
Query: "{f.query}"
Seed file: {f.seed_file}
Expected top files: {', '.join(f.expected_top[:5])}
Actual top files: {', '.join(f.actual_top[:5])}
NDCG score: {f.ndcg:.3f}
Commit context: "{f.commit_message}"
Repo: {f.repo_context.get('name', 'unknown')} ({f.repo_context.get('file_count', '?')} files)"""
        for i, f in enumerate(failures[:5])  # Limit to 5 failures per episode
    ])

    # Format current parameters
    params_desc = "\n".join([
        f"  {k}: {v}" for k, v in asdict(current_params).items()
    ])

    # Format prior insights from scratchpad
    prior_insights = ""
    if scratchpad.param_interactions:
        prior_insights += "\nKNOWN PARAMETER INTERACTIONS:\n" + "\n".join(
            f"  • {i}" for i in scratchpad.param_interactions[-5:]
        )
    if scratchpad.failure_patterns:
        prior_insights += "\nKNOWN FAILURE PATTERNS:\n" + "\n".join(
            f"  • {p}" for p in scratchpad.failure_patterns[-5:]
        )

    prompt = f"""You are a REASONING-BASED OPTIMIZER for a code ranking system.

Your task is to analyze ranking failures and propose hyperparameter adjustments.
You are approximating the gradient in CONCEPT SPACE, not parameter space.

The question is not just "what values minimize loss" but "what structure would
make this failure class impossible?"

=== CURRENT PARAMETERS ===
{params_desc}

=== RANKING FAILURES TO ANALYZE ===
{failure_desc}

=== PRIOR KNOWLEDGE (from previous episodes) ===
{prior_insights if prior_insights else "No prior insights yet."}

=== YOUR TASK ===

1. DIAGNOSE: Why did each failure occur? What signal was missing or overwhelming?

2. REASON ABOUT INTERACTIONS: Do any parameter pairs interact to cause this?
   (e.g., "low α + high boost_chat = tunnel vision because...")

3. PROPOSE CHANGES: For each parameter that should change, specify:
   - Direction: increase/decrease
   - Magnitude: small (10%), medium (30%), large (2x)
   - Rationale: why this change addresses the failure

4. STRUCTURAL INSIGHTS: Are there patterns that can't be fixed by tuning?
   (e.g., "multiplicative combination can't express OR logic")

5. CONFIDENCE: How confident are you in these proposals? (0-1)

=== OUTPUT FORMAT ===
Respond with a structured analysis. Use these exact headers:

## Diagnosis
[Your analysis of why each failure occurred]

## Parameter Interactions
[Any discovered interactions between parameters]

## Proposed Changes
[Format: PARAM_NAME: DIRECTION MAGNITUDE "rationale"]

## Structural Insights
[Patterns that suggest architectural changes, not just tuning]

## Confidence
[A number 0-1 and brief justification]
"""

    response = call_claude(prompt, timeout=90)

    # Parse the response
    episode = ReasoningEpisode(
        failures=failures,
        current_params=current_params,
        reasoning=response,
        proposed_changes={},
        structural_insights=[],
        confidence=0.5
    )

    # Extract proposed changes
    changes_match = re.search(r'## Proposed Changes\n(.*?)(?=##|\Z)', response, re.DOTALL)
    if changes_match:
        changes_text = changes_match.group(1)
        for line in changes_text.strip().split('\n'):
            # Parse lines like "boost_chat_file: decrease medium "drowns structural signal""
            match = re.match(r'(\w+):\s*(increase|decrease)\s+(\w+)\s+"([^"]+)"', line)
            if match:
                param, direction, magnitude, rationale = match.groups()
                episode.proposed_changes[param] = (direction, magnitude, rationale)

    # Extract structural insights
    insights_match = re.search(r'## Structural Insights\n(.*?)(?=##|\Z)', response, re.DOTALL)
    if insights_match:
        for line in insights_match.group(1).strip().split('\n'):
            if line.strip() and not line.startswith('#'):
                episode.structural_insights.append(line.strip())

    # Extract confidence
    conf_match = re.search(r'## Confidence\n([\d.]+)', response)
    if conf_match:
        try:
            episode.confidence = float(conf_match.group(1))
        except ValueError:
            pass

    return episode


def update_scratchpad(scratchpad: Scratchpad, episode: ReasoningEpisode):
    """Accumulate insights from an episode into the scratchpad."""
    scratchpad.episodes.append(episode)

    # Extract parameter interactions from reasoning
    interactions_match = re.search(
        r'## Parameter Interactions\n(.*?)(?=##|\Z)',
        episode.reasoning,
        re.DOTALL
    )
    if interactions_match:
        for line in interactions_match.group(1).strip().split('\n'):
            if line.strip() and not line.startswith('#'):
                scratchpad.param_interactions.append(line.strip())

    # Add structural insights
    scratchpad.structural_proposals.extend(episode.structural_insights)

    # Pattern detection: look for repeated failure types
    # (Would be more sophisticated in practice)


def apply_proposed_changes(
    params: ParameterPoint,
    changes: dict
) -> ParameterPoint:
    """Apply proposed changes to create new parameter point."""
    new_params = ParameterPoint(**asdict(params))

    magnitude_map = {
        'small': 0.1,
        'medium': 0.3,
        'large': 1.0,  # 2x
    }

    for param_name, (direction, magnitude, _rationale) in changes.items():
        if hasattr(new_params, param_name):
            current = getattr(new_params, param_name)
            mult = magnitude_map.get(magnitude, 0.3)

            if direction == 'increase':
                new_val = current * (1 + mult)
            else:
                new_val = current * (1 - mult)

            setattr(new_params, param_name, new_val)

    return new_params


def distill_scratchpad(scratchpad: Scratchpad) -> str:
    """
    Distill accumulated insights into operator wisdom.

    This is the crystallization phase where reasoning traces become:
    - Presets (clustered episodes → named configurations)
    - Heuristics (conditionals extracted from patterns)
    - Warnings (failure modes to avoid)
    - Prose wisdom (teaching the tool's use)
    """

    # Gather all insights
    all_interactions = list(set(scratchpad.param_interactions))
    all_structural = list(set(scratchpad.structural_proposals))

    # Count proposed changes across episodes
    change_counts = {}
    for ep in scratchpad.episodes:
        for param, (direction, mag, _) in ep.proposed_changes.items():
            key = f"{param}:{direction}"
            change_counts[key] = change_counts.get(key, 0) + 1

    prompt = f"""You are distilling optimization insights into operator wisdom.

You have accumulated {len(scratchpad.episodes)} reasoning episodes about a code ranking system.

=== DISCOVERED PARAMETER INTERACTIONS ===
{chr(10).join(f"• {i}" for i in all_interactions[:20]) or "None yet"}

=== STRUCTURAL PROPOSALS (beyond tuning) ===
{chr(10).join(f"• {s}" for s in all_structural[:10]) or "None yet"}

=== MOST COMMON PARAMETER CHANGES ===
{chr(10).join(f"• {k}: {v} episodes" for k, v in sorted(change_counts.items(), key=lambda x: -x[1])[:10]) or "None yet"}

=== YOUR TASK ===

Distill these insights into operator wisdom:

1. PRESETS: Name 3-5 configurations for common use cases
   Format: preset_name: {{param: value, ...}} "when to use"

2. ADAPTIVE HEURISTICS: Conditionals that adjust params based on context
   Format: if CONDITION: ADJUSTMENT "rationale"

3. WARNINGS: Failure modes operators should avoid
   Format: ⚠ WARNING: what happens and why

4. INTUITIONS: Prose wisdom that teaches the tool's use
   (Write as if explaining to a thoughtful user)

Be specific. Reference actual parameter names and values.
"""

    return call_claude(prompt, timeout=120)


def save_scratchpad(scratchpad: Scratchpad, path: Path):
    """Save scratchpad to markdown for human review."""
    with open(path, 'w') as f:
        f.write("# Reasoning Optimization Scratchpad\n\n")

        f.write(f"## Summary\n")
        f.write(f"- Episodes: {len(scratchpad.episodes)}\n")
        f.write(f"- Parameter interactions discovered: {len(scratchpad.param_interactions)}\n")
        f.write(f"- Structural proposals: {len(scratchpad.structural_proposals)}\n\n")

        f.write("## Parameter Interactions\n")
        for interaction in scratchpad.param_interactions:
            f.write(f"- {interaction}\n")
        f.write("\n")

        f.write("## Structural Proposals\n")
        for proposal in scratchpad.structural_proposals:
            f.write(f"- {proposal}\n")
        f.write("\n")

        f.write("## Episode History\n")
        for i, ep in enumerate(scratchpad.episodes):
            f.write(f"\n### Episode {i+1}\n")
            f.write(f"Confidence: {ep.confidence}\n")
            f.write(f"Proposed changes: {len(ep.proposed_changes)}\n")
            if ep.proposed_changes:
                for param, (dir, mag, rat) in ep.proposed_changes.items():
                    f.write(f"  - {param}: {dir} {mag} \"{rat}\"\n")


def main():
    parser = argparse.ArgumentParser(
        description="Reasoning-based hyperparameter optimization"
    )
    subparsers = parser.add_subparsers(dest='command')

    # Optimize command
    opt_parser = subparsers.add_parser('optimize', help='Run optimization episode')
    opt_parser.add_argument('--failures', type=Path, required=True,
                           help='JSON file with ranking failures')
    opt_parser.add_argument('--params', type=Path,
                           help='Current parameter JSON')
    opt_parser.add_argument('--scratchpad', type=Path, default=Path('tmp/scratchpad.md'),
                           help='Scratchpad file to accumulate insights')
    opt_parser.add_argument('--output', type=Path,
                           help='Output new params JSON')

    # Distill command
    dist_parser = subparsers.add_parser('distill', help='Distill scratchpad into wisdom')
    dist_parser.add_argument('--scratchpad', type=Path, required=True,
                            help='Scratchpad to distill')
    dist_parser.add_argument('--output', type=Path, default=Path('tmp/intuitions.md'),
                            help='Output wisdom file')

    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demo with synthetic failures')

    args = parser.parse_args()

    if args.command == 'demo':
        # Create synthetic failures for demonstration
        failures = [
            RankingFailure(
                query="auth",
                seed_file="src/auth/login.rs",
                expected_top=["src/auth/session.rs", "src/auth/token.rs"],
                actual_top=["src/main.rs", "README.md", "Cargo.toml"],
                ndcg=0.23,
                commit_message="fix authentication token refresh",
                repo_context={"name": "example-app", "file_count": 150}
            ),
            RankingFailure(
                query="database",
                seed_file="src/db/connection.rs",
                expected_top=["src/db/pool.rs", "src/db/query.rs"],
                actual_top=["src/db/connection.rs", "src/config.rs"],
                ndcg=0.45,
                commit_message="optimize connection pooling",
                repo_context={"name": "example-app", "file_count": 150}
            ),
        ]

        scratchpad = Scratchpad()
        params = ParameterPoint()

        print("=== REASONING EPISODE ===\n")
        episode = reason_about_failures(failures, params, scratchpad)

        print("Reasoning:")
        print(episode.reasoning[:2000] + "..." if len(episode.reasoning) > 2000 else episode.reasoning)
        print(f"\nConfidence: {episode.confidence}")
        print(f"Proposed changes: {len(episode.proposed_changes)}")
        for param, (d, m, r) in episode.proposed_changes.items():
            print(f"  {param}: {d} {m} - \"{r}\"")

        update_scratchpad(scratchpad, episode)
        save_scratchpad(scratchpad, Path('tmp/scratchpad_demo.md'))
        print(f"\nScratchpad saved to tmp/scratchpad_demo.md")

    elif args.command == 'optimize':
        # Load failures
        with open(args.failures) as f:
            failures_data = json.load(f)
        failures = [RankingFailure(**f) for f in failures_data]

        # Load or create params
        if args.params and args.params.exists():
            with open(args.params) as f:
                params = ParameterPoint(**json.load(f))
        else:
            params = ParameterPoint()

        # Load or create scratchpad
        scratchpad = Scratchpad()  # Would load from file in practice

        # Run reasoning
        episode = reason_about_failures(failures, params, scratchpad)
        update_scratchpad(scratchpad, episode)

        # Apply changes
        new_params = apply_proposed_changes(params, episode.proposed_changes)

        # Save outputs
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(asdict(new_params), f, indent=2)

        save_scratchpad(scratchpad, args.scratchpad)

        print(f"Episode complete. Confidence: {episode.confidence}")
        print(f"Changes: {len(episode.proposed_changes)}")

    elif args.command == 'distill':
        # Would load scratchpad from file in practice
        scratchpad = Scratchpad()

        wisdom = distill_scratchpad(scratchpad)

        with open(args.output, 'w') as f:
            f.write("# Distilled Operator Wisdom\n\n")
            f.write(wisdom)

        print(f"Wisdom distilled to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
