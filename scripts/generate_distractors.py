#!/usr/bin/env python3
"""
Generate semantic distractors using Claude CLI.

For each training case (seed file + expected related files), Claude generates
plausible-but-wrong file paths that could confuse a naive ranker. These create
a realistic challenge: files that LOOK relevant but weren't actually changed together.

The key insight: synthetic distractors like "src/distractor_0.rs" are too easy to filter.
Claude generates paths that share semantic keywords, similar structure, and plausible
relationships - but weren't in the actual commit.

Uses the `claude` CLI directly - no API keys needed!

Usage:
    python scripts/generate_distractors.py --repo ./bench_repos/ripgrep --output distractors.json

Or for all curated repos:
    python scripts/generate_distractors.py --all-curated --output all_distractors.json
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def get_repo_file_tree(repo_path: Path, max_files: int = 500) -> list[str]:
    """Get list of files in repo using fd for speed."""
    try:
        result = subprocess.run(
            ["fd", "-t", "f", "--max-results", str(max_files), ".", str(repo_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        files = result.stdout.strip().split("\n")
        # Convert to relative paths
        return [str(Path(f).relative_to(repo_path)) for f in files if f]
    except Exception as e:
        print(f"Warning: fd failed, falling back to walk: {e}")
        files = []
        for root, _, filenames in os.walk(repo_path):
            for fname in filenames:
                full = Path(root) / fname
                try:
                    files.append(str(full.relative_to(repo_path)))
                except ValueError:
                    pass
                if len(files) >= max_files:
                    return files
        return files


def extract_cases_from_git(repo_path: Path, max_commits: int = 100) -> list[dict]:
    """Extract training cases from git history."""
    result = subprocess.run(
        ["git", "-C", str(repo_path), "log",
         f"--max-count={max_commits}",
         "--pretty=format:%H|%s",
         "--name-only"],
        capture_output=True,
        text=True,
        timeout=60
    )

    cases = []
    current_commit = None
    current_msg = None
    current_files = []

    for line in result.stdout.split("\n"):
        if "|" in line and len(line.split("|")[0]) == 40:
            # New commit
            if current_commit and len(current_files) >= 2:
                cases.append({
                    "commit": current_commit,
                    "message": current_msg,
                    "files": current_files[:10]  # Cap at 10 files per commit
                })
            parts = line.split("|", 1)
            current_commit = parts[0]
            current_msg = parts[1] if len(parts) > 1 else ""
            current_files = []
        elif line.strip():
            current_files.append(line.strip())

    # Don't forget last commit
    if current_commit and len(current_files) >= 2:
        cases.append({
            "commit": current_commit,
            "message": current_msg,
            "files": current_files[:10]
        })

    return cases


def generate_distractors_for_case(
    case: dict,
    all_files: list[str],
    n_distractors: int = 5
) -> list[str]:
    """
    Ask Claude CLI to generate plausible but wrong file paths.

    The magic: Claude understands semantic relationships. If the commit touches
    "src/parser/lexer.rs" and "src/parser/tokens.rs", Claude might suggest
    "src/parser/ast.rs" as a distractor - semantically related but not in this commit.
    """
    import re

    seed_file = case["files"][0]
    related_files = case["files"][1:]
    commit_msg = case["message"]

    # Sample of actual files in repo (for realism)
    file_sample = "\n".join(all_files[:100])

    prompt = f"""You are acting as a UNIVERSAL FUNCTION APPROXIMATOR for code relevance prediction.

Your task is PURE REASONING - no code editing, no file modification, no execution.
You are approximating the function: f(commit_context) -> plausible_but_wrong_files

The ground truth is: files that LOOK semantically related but were NOT actually changed together.
This is adversarial data generation for training a ranking system.

=== COMMIT CONTEXT ===
Seed file: {seed_file}
Co-changed files: {', '.join(related_files)}
Commit message: "{commit_msg}"

=== CODEBASE STRUCTURE (sample) ===
{file_sample}

=== YOUR TASK ===
Generate exactly {n_distractors} DISTRACTOR file paths that:

1. SEMANTIC SIMILARITY: Names/paths that share keywords, concepts, or structural patterns
   with the changed files (sibling modules, related utilities, corresponding tests)

2. STRUCTURAL PLAUSIBILITY: Follow the exact naming conventions visible in this codebase
   (case style, directory depth, file extensions, test patterns)

3. ADVERSARIAL CHALLENGE: These should confuse a naive keyword/embedding ranker -
   files that LOOK relevant based on surface features but lack true coupling

4. EXCLUSION: Must NOT include any file from the commit (seed or co-changed)

Think like this: "What files would a junior developer THINK should change together
with these files, but actually wouldn't need to?"

=== OUTPUT FORMAT ===
Return ONLY a valid JSON array of file path strings. No explanation, no markdown.
Example: ["src/parser/ast.rs", "src/utils/helpers.rs", "tests/parser_test.rs"]
"""

    try:
        # Use claude CLI with --print flag for non-interactive output
        result = subprocess.run(
            ["claude", "--print", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"Warning: claude CLI error: {result.stderr[:200]}")
            return []

        text = result.stdout.strip()

        # Parse JSON array from response
        if text.startswith("["):
            return json.loads(text)

        # Try to find JSON array in response
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())

        return []
    except subprocess.TimeoutExpired:
        print("Warning: claude CLI timed out")
        return []
    except Exception as e:
        print(f"Warning: Error calling claude: {e}")
        return []


def process_repo(
    repo_path: Path,
    max_cases: int = 50,
    distractors_per_case: int = 5
) -> dict:
    """Process a single repo and generate distractors for its cases."""

    print(f"\nProcessing {repo_path.name}...")

    # Get file tree
    all_files = get_repo_file_tree(repo_path)
    print(f"  Found {len(all_files)} files")

    # Extract cases
    cases = extract_cases_from_git(repo_path, max_commits=max_cases * 2)
    cases = cases[:max_cases]
    print(f"  Extracted {len(cases)} training cases")

    # Generate distractors for each case
    results = []
    for i, case in enumerate(cases):
        if i % 10 == 0:
            print(f"  Generating distractors: {i}/{len(cases)}")

        distractors = generate_distractors_for_case(
            case, all_files, distractors_per_case
        )

        results.append({
            "seed_file": case["files"][0],
            "expected_related": case["files"][1:],
            "commit_message": case["message"],
            "semantic_distractors": distractors
        })

    return {
        "repo": repo_path.name,
        "n_cases": len(results),
        "cases": results
    }


CURATED_REPOS = [
    ("BurntSushi/ripgrep", "ripgrep"),
    ("sharkdp/bat", "bat"),
    ("casey/just", "just"),
    ("starship/starship", "starship"),
    ("XAMPPRocky/tokei", "tokei"),
    ("eza-community/eza", "eza"),
    ("encode/httpx", "httpx"),
    ("Textualize/rich", "rich"),
    ("Textualize/textual", "textual"),
    ("pydantic/pydantic", "pydantic"),
    ("astral-sh/ruff", "ruff"),
    ("colinhacks/zod", "zod"),
    ("charmbracelet/bubbletea", "bubbletea"),
    ("charmbracelet/lipgloss", "lipgloss"),
    ("jesseduffield/lazygit", "lazygit"),
]


def main():
    parser = argparse.ArgumentParser(description="Generate semantic distractors with Claude CLI")
    parser.add_argument("--repo", type=Path, help="Single repo path")
    parser.add_argument("--all-curated", action="store_true", help="Process all curated repos")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON file")
    parser.add_argument("--max-cases", type=int, default=30, help="Max cases per repo")
    parser.add_argument("--distractors-per-case", type=int, default=5, help="Distractors per case")
    parser.add_argument("--bench-repos-dir", type=Path, default=Path("./bench_repos"),
                        help="Directory containing cloned repos")
    args = parser.parse_args()

    all_results = []

    if args.repo:
        result = process_repo(
            args.repo, args.max_cases, args.distractors_per_case
        )
        all_results.append(result)

    elif args.all_curated:
        for _, name in CURATED_REPOS:
            repo_path = args.bench_repos_dir / name
            if repo_path.exists():
                result = process_repo(
                    repo_path, args.max_cases, args.distractors_per_case
                )
                all_results.append(result)
            else:
                print(f"Warning: {repo_path} not found, skipping")

    else:
        parser.error("Specify --repo or --all-curated")

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    total_cases = sum(r["n_cases"] for r in all_results)
    total_distractors = sum(
        len(c["semantic_distractors"])
        for r in all_results
        for c in r["cases"]
    )

    print(f"\n=== Summary ===")
    print(f"Repos processed: {len(all_results)}")
    print(f"Total cases: {total_cases}")
    print(f"Total semantic distractors: {total_distractors}")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
