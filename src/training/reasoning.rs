//! Reasoning-Based Hyperparameter Optimization via LLM as Universal Function Approximator.
//!
//! The paradigm shift:
//! - Classical: observe Loss(θ) → infer ∂Loss/∂θ → step θ (black box: WHY is lost)
//! - Reasoning: observe Failure(θ) → reason about WHY → propose Δθ OR Δstructure
//!
//! The gradient isn't in parameter space—it's in concept space.
//!
//! ## The Sidechain Architecture
//!
//! The scratchpad accumulates insights across optimization episodes, building a theory
//! of the hyperparameter manifold that can predict:
//! - "If you're in situation X, parameter Y will fail because Z"
//! - "When A is high, B must compensate or you'll see symptom C"
//!
//! This theory gets distilled into:
//! - PRESETS: clustered insights → named configurations
//! - ADAPTIVE HEURISTICS: conditionals extracted from patterns
//! - OPERATOR INTUITIONS: crystallized warnings and wisdom
//!
//! ## Agent Support
//!
//! Supports multiple LLM backends via `--agent`:
//! - `claude` (default): Uses Claude CLI (`claude --print -p`)
//! - `gemini`: Uses Gemini CLI (`gemini -o json`)

use std::collections::HashMap;
use std::process::Command;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use super::gridsearch::ParameterPoint;

/// Which LLM agent to use for reasoning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Agent {
    #[default]
    Claude,
    Gemini,
    /// OpenAI Codex CLI (ChatGPT Pro / o3)
    Codex,
}

impl FromStr for Agent {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "claude" => Ok(Agent::Claude),
            "gemini" => Ok(Agent::Gemini),
            "codex" | "openai" | "o3" => Ok(Agent::Codex),
            _ => Err(format!(
                "Unknown agent: {}. Use 'claude', 'gemini', or 'codex'",
                s
            )),
        }
    }
}

impl std::fmt::Display for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Agent::Claude => write!(f, "claude"),
            Agent::Gemini => write!(f, "gemini"),
            Agent::Codex => write!(f, "codex"),
        }
    }
}

/// A ranking failure case for reasoning analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingFailure {
    /// The focus query used
    pub query: String,
    /// The seed file (starting point)
    pub seed_file: String,
    /// Files that should have ranked high (ground truth)
    pub expected_top: Vec<String>,
    /// Files that actually ranked high
    pub actual_top: Vec<String>,
    /// NDCG score (lower = worse failure)
    pub ndcg: f64,
    /// Commit message providing context
    pub commit_context: String,
    /// Repository metadata
    pub repo_name: String,
    pub repo_file_count: usize,
    /// Pipeline statistics (if bicameral mode enabled)
    /// Provides critical diagnostic signals for L1 reasoning:
    /// - shadow_connectivity: Edge density revealing shadow graph quality
    /// - lsp_utilization: Query efficiency (resolved / issued)
    /// - shadow_final_rank_correlation: How well shadow predicts final
    /// - lsp_latency_ms: Cost metric for policy optimization
    #[serde(default)]
    pub pipeline_stats: Option<PipelineStatsSnapshot>,
}

/// Serializable snapshot of PipelineStats for training context.
/// Captures the bicameral pipeline diagnostics at the moment of failure.
/// L1 uses these metrics to diagnose failure modes:
/// - Shadow Collapse: average_degree < 1.0 → shadow_strategy.weight_name_match too low
/// - LSP Waste: utilization < 0.3 → lsp_policy.marginal_utility_floor too low
/// - Rank Divergence: correlation < 0.5 → shadow graph too noisy
/// - Cost Overrun: latency > 10s → query_budget too high or floor too low
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineStatsSnapshot {
    /// Edge density of shadow graph (edges / possible_edges)
    /// Note: Raw density scales inversely with graph size! Use average_degree for alerts.
    pub shadow_connectivity: f64,
    /// Average degree of shadow graph (edges / nodes)
    /// This metric is size-invariant and should be >= 1.0 for tree-like connectivity.
    /// Low values (<1.0) indicate shadow collapse - each node has less than one edge on average.
    #[serde(default)]
    pub average_degree: f64,
    /// LSP success rate (resolved / queried)
    /// Low values (<0.3) indicate wasted queries - policy selecting bad sites
    pub lsp_utilization: f64,
    /// Spearman correlation between shadow and final PageRank
    /// Low values (<0.5) indicate shadow graph doesn't predict final structure
    pub shadow_final_rank_correlation: f64,
    /// Total LSP latency in milliseconds
    /// High values (>10000) indicate excessive query cost
    pub lsp_latency_ms: u64,
}

/// One round of reasoning about failures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningEpisode {
    /// Unix timestamp (epoch seconds) when this episode was created.
    /// Used for sorting runs by date and displaying temporal context.
    #[serde(default)]
    pub timestamp: i64,
    /// Duration of this episode in seconds (LLM call + evaluation).
    /// Useful for benchmarking different agents and tracking cost.
    #[serde(default)]
    pub duration_secs: f64,
    /// Failures analyzed in this episode
    pub failures: Vec<RankingFailure>,
    /// Parameters at time of failure
    pub params: ParameterPoint,
    /// NDCG@10 before this episode's changes (for tracking convergence)
    #[serde(default)]
    pub ndcg_before: f64,
    /// LLM's free-form reasoning about the trajectory and failures
    pub reasoning: String,
    /// 1-2 sentence capsule encoding the intent/strategy for this episode.
    /// E.g., "Testing counterfactual: reversing depth weights to see if shallow penalty helps"
    /// This lets future episodes understand not just WHAT changed but WHY.
    #[serde(default)]
    pub strategy_capsule: String,
    /// Proposed parameter changes: param_name -> (direction, magnitude, rationale)
    /// Direction: "increase" or "decrease"
    /// Magnitude: "small" (10%), "medium" (30%), "large" (2x)
    pub proposed_changes: HashMap<String, (String, String, String)>,
    /// Discovered parameter interactions (e.g., "low α + high boost = tunnel vision")
    #[serde(default)]
    pub param_interactions: Vec<String>,
    /// Structural insights beyond parameter tuning
    pub structural_insights: Vec<String>,
    /// Confidence in proposals (0.0 - 1.0)
    pub confidence: f64,
}

/// Accumulated mental model across optimization episodes.
/// This is the sidechain scratchpad that builds theory of the parameter space.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Scratchpad {
    /// All reasoning episodes
    pub episodes: Vec<ReasoningEpisode>,

    /// Discovered parameter interactions
    /// e.g., "low α + high boost_chat = tunnel vision"
    pub param_interactions: Vec<String>,

    /// Recurring failure patterns
    pub failure_patterns: Vec<String>,

    /// Success patterns (what worked)
    pub success_patterns: Vec<String>,

    /// Proposals for structural changes (beyond tuning)
    pub structural_proposals: Vec<String>,

    /// Distilled presets: name -> (params, description)
    pub presets: HashMap<String, (ParameterPoint, String)>,

    /// Adaptive heuristics: "if CONDITION: ADJUSTMENT"
    pub heuristics: Vec<String>,

    /// Warnings about failure modes
    pub warnings: Vec<String>,
}

/// Call Claude CLI and return response.
///
/// Uses `claude --print -p "prompt"` for non-interactive output.
///
/// ## Phase 1 Agentic Mode (when `run_dir` is provided):
/// - Enables Read, Grep, Glob tools for single-shot file exploration
/// - Grants access to scratchpad and episode history
/// - Agent can read files before reasoning, but still produces response in one turn
pub fn call_claude(
    prompt: &str,
    model: Option<&str>,
    run_dir: Option<&str>,
) -> Result<String, String> {
    let mut args = vec!["--print"];
    let model_str;
    if let Some(m) = model {
        model_str = m.to_string();
        args.insert(0, "--model");
        args.insert(1, &model_str);
    }

    // Phase 1 agentic mode: enable read-only tools and directory access
    if let Some(dir) = run_dir {
        args.push("--tools");
        args.push("Read,Glob,Grep");
        args.push("--permission-mode");
        args.push("bypassPermissions");
        args.push("--add-dir");
        args.push(dir);
    }

    args.push("-p");
    args.push(prompt);

    let output = Command::new("claude")
        .args(&args)
        .output()
        .map_err(|e| format!("Failed to execute claude: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Claude returned error: {}", stderr));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Call Gemini CLI and return response.
///
/// Uses `gemini -o text "prompt"` for non-interactive output.
/// Gemini outputs plain text by default, we ask for text mode explicitly.
///
/// ## Phase 1 Agentic Mode (when `run_dir` is provided):
/// - Gemini is ALREADY agentic with `-y` (yolo mode)
/// - Adds `--include-directories` to grant file access to scratchpad/episodes
/// - Agent can explore files during single-shot reasoning
pub fn call_gemini(
    prompt: &str,
    model: Option<&str>,
    run_dir: Option<&str>,
) -> Result<String, String> {
    let mut cmd = Command::new("gemini");
    cmd.args(["-o", "text", "-y"]);
    if let Some(m) = model {
        cmd.args(["-m", m]);
    }

    // Phase 1 agentic mode: grant directory access (Gemini is already agentic with -y)
    if let Some(dir) = run_dir {
        cmd.arg("--include-directories");
        cmd.arg(dir);
    }

    cmd.arg(prompt);

    let output = cmd
        .output()
        .map_err(|e| format!("Failed to execute gemini: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Gemini returned error: {}", stderr));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Call Codex CLI (OpenAI o3) and return response.
///
/// Uses `codex exec` for non-interactive output.
/// Reads prompt from stdin and writes output to a temp file.
///
/// ## Phase 1 Agentic Mode (when `run_dir` is provided):
/// - Codex is ALREADY agentic with `exec`
/// - Adds `--add-dir` to grant file access to scratchpad/episodes
/// - Agent can explore files during execution before producing final output
pub fn call_codex(
    prompt: &str,
    model: Option<&str>,
    run_dir: Option<&str>,
) -> Result<String, String> {
    use std::io::Write;

    // Create temp file for output
    let output_file = std::env::temp_dir().join(format!("codex_out_{}.txt", std::process::id()));

    let mut args = vec![
        "exec".to_string(),
        "--skip-git-repo-check".to_string(),
        "--dangerously-bypass-approvals-and-sandbox".to_string(),
    ];
    if let Some(m) = model {
        args.push("-m".to_string());
        args.push(m.to_string());
    }

    // Phase 1 agentic mode: grant directory access (Codex is already agentic with exec)
    if let Some(dir) = run_dir {
        args.push("--add-dir".to_string());
        args.push(dir.to_string());
    }

    args.push("-o".to_string());
    args.push(output_file.to_str().unwrap().to_string());
    args.push("-".to_string()); // read prompt from stdin

    // Run codex exec with prompt from stdin, output to file
    let mut child = Command::new("codex")
        .args(&args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to execute codex: {}", e))?;

    // Write prompt to stdin
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(prompt.as_bytes())
            .map_err(|e| format!("Failed to write to codex stdin: {}", e))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|e| format!("Failed to wait for codex: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let exit_code = output.status.code().unwrap_or(-1);

        // Save failed prompt for debugging
        let debug_file =
            std::env::temp_dir().join(format!("codex_failed_{}.txt", std::process::id()));
        let _ = std::fs::write(
            &debug_file,
            format!(
                "=== CODEX CALL FAILED ===\nExit code: {}\nArgs: {:?}\n\n=== PROMPT ({} chars) ===\n{}\n\n=== STDOUT ===\n{}\n\n=== STDERR ===\n{}",
                exit_code,
                args,
                prompt.len(),
                prompt,
                stdout,
                stderr
            ),
        );

        // Extract actual error from stderr (skip Codex session header noise)
        let error_lines: Vec<&str> = stderr
            .lines()
            .filter(|l| {
                !l.starts_with("OpenAI Codex")
                    && !l.starts_with("--------")
                    && !l.starts_with("workdir:")
                    && !l.starts_with("model:")
                    && !l.starts_with("provider:")
                    && !l.starts_with("approval:")
                    && !l.starts_with("sandbox:")
                    && !l.starts_with("reasoning")
                    && !l.starts_with("session id:")
                    && !l.trim().is_empty()
            })
            .collect();

        let clean_error = if error_lines.is_empty() {
            format!(
                "Exit code {} (no error message, check {})",
                exit_code,
                debug_file.display()
            )
        } else {
            error_lines.join("\n")
        };

        // Clean up temp file
        let _ = std::fs::remove_file(&output_file);

        eprintln!(
            "[codex] Failed with exit code {}. Debug saved to: {}",
            exit_code,
            debug_file.display()
        );
        return Err(format!(
            "Codex failed (exit {}): {}",
            exit_code, clean_error
        ));
    }

    // Read output from file
    let response = std::fs::read_to_string(&output_file)
        .map_err(|e| format!("Failed to read codex output: {}", e))?;

    // Clean up temp file
    let _ = std::fs::remove_file(&output_file);

    Ok(response.trim().to_string())
}

/// Call the specified LLM agent and return response.
///
/// ## Phase 1 Agentic Mode (when `run_dir` is provided):
/// - Grants agents read-only file access to scratchpad and episode history
/// - Agents can explore files during reasoning before producing structured output
/// - Still single-shot execution (no multi-turn iteration)
pub fn call_agent(
    agent: Agent,
    prompt: &str,
    model: Option<&str>,
    run_dir: Option<&str>,
) -> Result<String, String> {
    match agent {
        Agent::Claude => call_claude(prompt, model, run_dir),
        Agent::Gemini => call_gemini(prompt, model, run_dir),
        Agent::Codex => call_codex(prompt, model, run_dir),
    }
}

/// Reason about ranking failures and propose parameter changes.
///
/// This is where the LLM acts as a universal function approximator:
/// f(failures, params, history) -> (reasoning, proposals, insights)
///
/// The `prompt_template` should contain:
/// - Role, Policy, Heuristics, Style sections (editable by L2)
/// - Placeholders for dynamic context:
///   - `{current_ndcg:.4}` - current NDCG score
///   - `{episode_num}` - current episode number
///   - `{episode_history}` - formatted history of recent episodes
///   - `{params_desc}` - current parameter values
///   - `{failure_desc}` - formatted failure cases
///
/// The immutable output schema (API_contract + Output_schema) is injected
/// at runtime from `training/prompts/protocol/inner_output_schema.md` to prevent
/// L2 from corrupting the structured output format during meta-optimization.
///
/// Supports multiple agents via the `agent` parameter.
/// Optionally specify a model (e.g., "opus", "o3", "gemini-2.0-flash").
///
/// ## Phase 1 Agentic Mode (when `run_dir` is provided):
/// - Agent gets read-only access to scratchpad.json and episode history
/// - Agent can explore previous reasoning before proposing changes
/// - Enables informed decision-making based on trajectory analysis
pub fn reason_about_failures(
    prompt_template: &str,
    failures: &[RankingFailure],
    current_params: &ParameterPoint,
    scratchpad: &Scratchpad,
    current_ndcg: f64,
    agent: Agent,
    model: Option<&str>,
    run_dir: Option<&str>,
) -> Result<ReasoningEpisode, String> {
    let episode_start = std::time::Instant::now();

    // Format failures for Claude
    let failure_desc: String = failures
        .iter()
        .take(5) // Limit to 5 failures per episode
        .enumerate()
        .map(|(i, f)| {
            format!(
                r#"FAILURE {}:
Query: "{}"
Seed file: {}
Expected top files: {}
Actual top files: {}
NDCG score: {:.3}
Commit context: "{}"
Repo: {} ({} files)"#,
                i + 1,
                f.query,
                f.seed_file,
                f.expected_top
                    .iter()
                    .take(5)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", "),
                f.actual_top
                    .iter()
                    .take(5)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", "),
                f.ndcg,
                f.commit_context,
                f.repo_name,
                f.repo_file_count
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    // Extract pipeline diagnostics if available (bicameral mode)
    // These metrics reveal failure modes that NDCG alone cannot diagnose:
    // - Is the shadow graph too sparse? (connectivity)
    // - Are LSP queries being wasted? (utilization)
    // - Is PageRank misleading the policy? (correlation)
    // - Is the pipeline too slow? (latency)
    let pipeline_context =
        if let Some(ref stats) = failures.first().and_then(|f| f.pipeline_stats.as_ref()) {
            format!(
                r#"

═══════════════════════════════════════════════════════════════════════════════
PIPELINE DIAGNOSTICS (Bicameral Shadow→LSP→Final)
═══════════════════════════════════════════════════════════════════════════════

Shadow Graph Metrics:
  Edge Connectivity: {:.3} ({:.1}% edge density)
    → Measures shadow graph sparsity. <0.01 = COLLAPSE (PageRank has no structure)

  Rank Correlation: {:.3} (shadow vs final PageRank)
    → Measures how well shadow predicts final importance
    → <0.5 = shadow graph is NOISY, misleading the LSP policy

LSP Query Metrics:
  Query Utilization: {:.1}% (queries that successfully resolved)
    → Measures policy efficiency. <30% = WASTING queries on bad sites

  Query Latency: {}ms ({:.1}s total)
    → Direct cost metric. >10s = TOO EXPENSIVE for production

⚠️  DIAGNOSTIC ALERTS:
{}

Context: These metrics diagnose bicameral pipeline health. L1 should consider:
- If shadow_connectivity is low → shadow_strategy heuristics may be too conservative
- If lsp_utilization is low → lsp_policy is selecting unresolvable sites
- If correlation is low → shadow graph doesn't predict final structure (noise!)
- If latency is high → reduce query_budget or increase marginal_utility_floor
═══════════════════════════════════════════════════════════════════════════════
"#,
                stats.shadow_connectivity,
                stats.shadow_connectivity * 100.0,
                stats.shadow_final_rank_correlation,
                stats.lsp_utilization * 100.0,
                stats.lsp_latency_ms,
                stats.lsp_latency_ms as f64 / 1000.0,
                generate_pipeline_alerts(stats),
            )
        } else {
            String::new()
        };

    // Format current parameters - base ranking params
    let mut params_desc = format!(
        r#"  pagerank_alpha: {:.3}
  pagerank_chat_multiplier: {:.1}
  depth_weight_root: {:.2}
  depth_weight_moderate: {:.2}
  depth_weight_deep: {:.2}
  depth_weight_vendor: {:.4}
  boost_mentioned_ident: {:.1}
  boost_mentioned_file: {:.1}
  boost_chat_file: {:.1}
  boost_temporal_coupling: {:.2}
  boost_focus_expansion: {:.2}
  git_recency_decay_days: {:.1}
  git_recency_max_boost: {:.2}
  git_churn_threshold: {:.1}
  git_churn_max_boost: {:.2}
  focus_decay: {:.2}
  focus_max_hops: {:.1}"#,
        current_params.pagerank_alpha,
        current_params.pagerank_chat_multiplier,
        current_params.depth_weight_root,
        current_params.depth_weight_moderate,
        current_params.depth_weight_deep,
        current_params.depth_weight_vendor,
        current_params.boost_mentioned_ident,
        current_params.boost_mentioned_file,
        current_params.boost_chat_file,
        current_params.boost_temporal_coupling,
        current_params.boost_focus_expansion,
        current_params.git_recency_decay_days,
        current_params.git_recency_max_boost,
        current_params.git_churn_threshold,
        current_params.git_churn_max_boost,
        current_params.focus_decay,
        current_params.focus_max_hops,
    );

    // Append pipeline coordinates if present (38 additional params for bicameral LSP resolution)
    // L1 can now see and modify the full resolution pipeline, not just ranking heuristics
    if let Some(ref pipeline) = current_params.pipeline {
        params_desc.push_str(&format!(
            r#"

  # Shadow Strategy (recall-optimized, noisy heuristics for hub discovery)
  pipeline.shadow_strategy.weight_same_file: {:.2}
  pipeline.shadow_strategy.weight_type_hint: {:.2}
  pipeline.shadow_strategy.weight_import: {:.2}
  pipeline.shadow_strategy.weight_name_match: {:.2}
  pipeline.shadow_strategy.weight_lsp: {:.2}
  pipeline.shadow_strategy.acceptance_bias: {:.2}
  pipeline.shadow_strategy.acceptance_slope: {:.1}
  pipeline.shadow_strategy.selection_temperature: {:.2}
  pipeline.shadow_strategy.evidence_accumulation: {:.2}
  pipeline.shadow_strategy.proximity_boost: {:.2}

  # Final Strategy (precision-optimized, LSP-enhanced resolution)
  pipeline.final_strategy.weight_same_file: {:.2}
  pipeline.final_strategy.weight_type_hint: {:.2}
  pipeline.final_strategy.weight_import: {:.2}
  pipeline.final_strategy.weight_name_match: {:.2}
  pipeline.final_strategy.weight_lsp: {:.2}
  pipeline.final_strategy.acceptance_bias: {:.2}
  pipeline.final_strategy.acceptance_slope: {:.1}
  pipeline.final_strategy.selection_temperature: {:.2}
  pipeline.final_strategy.evidence_accumulation: {:.2}
  pipeline.final_strategy.proximity_boost: {:.2}

  # LSP Query Policy (resource allocation and signal weighting)
  pipeline.lsp_policy.marginal_utility_floor: {:.4}
  pipeline.lsp_policy.batch_latency_bias: {:.2}
  pipeline.lsp_policy.query_timeout_secs: {:.1}
  pipeline.lsp_policy.max_retries: {:.1}
  pipeline.lsp_policy.cache_negative_bias: {:.2}
  pipeline.lsp_policy.weight_centrality: {:.2}
  pipeline.lsp_policy.weight_uncertainty: {:.2}
  pipeline.lsp_policy.weight_coherence: {:.2}
  pipeline.lsp_policy.weight_causality: {:.2}
  pipeline.lsp_policy.weight_bridge: {:.2}
  pipeline.lsp_policy.spread_logit_structural: {:.2}
  pipeline.lsp_policy.spread_logit_semantic: {:.2}
  pipeline.lsp_policy.spread_logit_spatial: {:.2}
  pipeline.lsp_policy.focus_temperature: {:.2}
  pipeline.lsp_policy.gated_threshold: {:.2}
  pipeline.lsp_policy.exploration_floor: {:.2}
  pipeline.lsp_policy.interaction_mixing: {:.2}
  pipeline.lsp_policy.centrality_normalization: {:.2}"#,
            // Shadow strategy (10 params)
            pipeline.shadow_strategy.weight_same_file,
            pipeline.shadow_strategy.weight_type_hint,
            pipeline.shadow_strategy.weight_import,
            pipeline.shadow_strategy.weight_name_match,
            pipeline.shadow_strategy.weight_lsp,
            pipeline.shadow_strategy.acceptance_bias,
            pipeline.shadow_strategy.acceptance_slope,
            pipeline.shadow_strategy.selection_temperature,
            pipeline.shadow_strategy.evidence_accumulation,
            pipeline.shadow_strategy.proximity_boost,
            // Final strategy (10 params)
            pipeline.final_strategy.weight_same_file,
            pipeline.final_strategy.weight_type_hint,
            pipeline.final_strategy.weight_import,
            pipeline.final_strategy.weight_name_match,
            pipeline.final_strategy.weight_lsp,
            pipeline.final_strategy.acceptance_bias,
            pipeline.final_strategy.acceptance_slope,
            pipeline.final_strategy.selection_temperature,
            pipeline.final_strategy.evidence_accumulation,
            pipeline.final_strategy.proximity_boost,
            // LSP policy (18 params)
            pipeline.lsp_policy.marginal_utility_floor,
            pipeline.lsp_policy.batch_latency_bias,
            pipeline.lsp_policy.query_timeout_secs,
            pipeline.lsp_policy.max_retries,
            pipeline.lsp_policy.cache_negative_bias,
            pipeline.lsp_policy.weight_centrality,
            pipeline.lsp_policy.weight_uncertainty,
            pipeline.lsp_policy.weight_coherence,
            pipeline.lsp_policy.weight_causality,
            pipeline.lsp_policy.weight_bridge,
            pipeline.lsp_policy.spread_logit_structural,
            pipeline.lsp_policy.spread_logit_semantic,
            pipeline.lsp_policy.spread_logit_spatial,
            pipeline.lsp_policy.focus_temperature,
            pipeline.lsp_policy.gated_threshold,
            pipeline.lsp_policy.exploration_floor,
            pipeline.lsp_policy.interaction_mixing,
            pipeline.lsp_policy.centrality_normalization,
        ));
    }

    // Build FULL episode history - the model needs to see the trajectory
    let episode_history = if scratchpad.episodes.is_empty() {
        "This is the FIRST episode. No history yet.".to_string()
    } else {
        let mut history = String::new();
        let recent_episodes: Vec<_> = scratchpad.episodes.iter().rev().take(10).collect();

        // Show NDCG trajectory with strategy intent - the model sees both WHAT happened and WHY
        history.push_str("EPISODE HISTORY (recent → older):\n");
        for (i, ep) in recent_episodes.iter().enumerate() {
            let trend = if i == 0 {
                ""
            } else {
                let prev_ndcg = recent_episodes
                    .get(i - 1)
                    .map(|e| e.ndcg_before)
                    .unwrap_or(0.0);
                if ep.ndcg_before > prev_ndcg + 0.01 {
                    " ↗"
                } else if ep.ndcg_before < prev_ndcg - 0.01 {
                    " ↘"
                } else {
                    " →"
                }
            };
            let ep_num = scratchpad.episodes.len() - i;
            let strategy = if ep.strategy_capsule.is_empty() {
                String::new()
            } else {
                format!(
                    "\n      Strategy: \"{}\"",
                    ep.strategy_capsule.chars().take(100).collect::<String>()
                )
            };
            history.push_str(&format!(
                "  E{}: NDCG={:.3}{} | failures={}{}\n",
                ep_num,
                ep.ndcg_before,
                trend,
                ep.failures.len(),
                strategy
            ));
        }

        // Show recent parameter changes - THE GRADIENT
        history.push_str("\nRECENT PARAMETER CHANGES:\n");
        for (i, ep) in recent_episodes.iter().take(5).enumerate() {
            let ep_num = scratchpad.episodes.len() - i;
            if !ep.proposed_changes.is_empty() {
                let changes: Vec<_> = ep
                    .proposed_changes
                    .iter()
                    .map(|(k, (dir, mag, _))| format!("{} {} {}", k, dir, mag))
                    .take(3)
                    .collect();
                history.push_str(&format!(
                    "  E{}: {} (conf={:.2})\n",
                    ep_num,
                    changes.join(", "),
                    ep.confidence
                ));
            }
        }

        // Trajectory analysis - help the model see the pattern
        if recent_episodes.len() >= 3 {
            let ndcgs: Vec<f64> = recent_episodes.iter().map(|e| e.ndcg_before).collect();
            let first = ndcgs.last().unwrap_or(&0.0);
            let last = ndcgs.first().unwrap_or(&0.0);
            let trend = last - first;

            history.push_str(&format!("\n⚠️ TRAJECTORY ANALYSIS:\n"));
            if trend < -0.05 {
                history.push_str(&format!(
                    "  ALERT: NDCG dropped {:.3} over last {} episodes!\n",
                    -trend,
                    recent_episodes.len()
                ));
                history.push_str("  Consider: Are recent changes making things WORSE?\n");
                history.push_str("  Consider: Should we REVERT to earlier params?\n");
            } else if trend > 0.02 {
                history.push_str(&format!(
                    "  Good: NDCG improved {:.3} - current direction is working\n",
                    trend
                ));
            } else {
                history.push_str("  Plateau: NDCG stable - may need different approach\n");
            }
        }

        // Key structural insights discovered
        if !scratchpad.structural_proposals.is_empty() {
            history.push_str("\nKEY INSIGHTS FROM TRAINING:\n");
            for insight in scratchpad.structural_proposals.iter().rev().take(5) {
                history.push_str(&format!("  • {}\n", insight));
            }
        }

        history
    };

    // Load the immutable output schema protocol
    // This is separate from the evolved policy to prevent L2 from corrupting the structured output
    let protocol_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("training/prompts/protocol/inner_output_schema.md");
    let protocol = std::fs::read_to_string(&protocol_path).map_err(|e| {
        format!(
            "Failed to load protocol from {}: {}",
            protocol_path.display(),
            e
        )
    })?;

    // Inject dynamic context into the prompt template (evolved policy)
    // Pipeline diagnostics are injected after failure_desc to provide structural context
    let evolved_policy = prompt_template
        .replace("{current_ndcg:.4}", &format!("{:.4}", current_ndcg))
        .replace(
            "{episode_num}",
            &(scratchpad.episodes.len() + 1).to_string(),
        )
        .replace("{episode_history}", &episode_history)
        .replace("{params_desc}", &params_desc)
        .replace(
            "{failure_desc}",
            &format!("{}{}", failure_desc, pipeline_context),
        );

    // Assemble final prompt: {protocol} + {evolved_policy} + {context}
    // The protocol is immutable and defines the output format
    // The evolved_policy contains Role, Policy, Heuristics, Style that L2 can evolve
    // The context (episode_history, params, failures) is dynamically injected
    let prompt = format!("{}\n\n{}", evolved_policy, protocol);

    // Phase 1 agentic mode: agent can read scratchpad.json and episode files if run_dir provided
    let response = call_agent(agent, &prompt, model, run_dir)?;

    // Extract the reasoning section (everything before JSON:) for storage
    let reasoning_text = if let Some(json_marker) = response.find("JSON:") {
        response[..json_marker]
            .replace("REASONING:", "")
            .trim()
            .to_string()
    } else {
        // No explicit sections, use everything before first {
        response.split('{').next().unwrap_or("").trim().to_string()
    };

    // Parse JSON response - look for JSON after the "JSON:" marker or find first {
    let json_str = if let Some(json_marker) = response.find("JSON:") {
        &response[json_marker + 5..]
    } else {
        &response
    };

    let parsed: serde_json::Value = serde_json::from_str(json_str.trim())
        .or_else(|_| {
            // Try to find JSON in response
            if let Some(start) = json_str.find('{') {
                if let Some(end) = json_str.rfind('}') {
                    return serde_json::from_str(&json_str[start..=end]);
                }
            }
            Err(serde_json::Error::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "No JSON found in response",
            )))
        })
        .map_err(|e| {
            format!(
                "Failed to parse response as JSON: {}\nResponse: {}",
                e,
                &response[..response.len().min(500)]
            )
        })?;

    // Extract fields
    let mut proposed_changes = HashMap::new();
    if let Some(changes) = parsed.get("proposed_changes").and_then(|v| v.as_object()) {
        for (param, value) in changes {
            if let Some(arr) = value.as_array() {
                if arr.len() >= 3 {
                    let direction = arr[0].as_str().unwrap_or("").to_string();
                    let magnitude = arr[1].as_str().unwrap_or("").to_string();
                    let rationale = arr[2].as_str().unwrap_or("").to_string();
                    proposed_changes.insert(param.clone(), (direction, magnitude, rationale));
                }
            }
        }
    }

    let strategy_capsule = parsed
        .get("strategy_capsule")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let param_interactions: Vec<String> = parsed
        .get("param_interactions")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default();

    let structural_insights: Vec<String> = parsed
        .get("structural_insights")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default();

    let confidence = parsed
        .get("confidence")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.5);

    // Capture the current timestamp for this episode
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    let duration_secs = episode_start.elapsed().as_secs_f64();

    Ok(ReasoningEpisode {
        timestamp,
        duration_secs,
        failures: failures.to_vec(),
        params: current_params.clone(),
        ndcg_before: current_ndcg,
        reasoning: reasoning_text, // Store the free-form reasoning, not the raw JSON
        strategy_capsule,          // The intent/mode for this episode
        proposed_changes,
        param_interactions,
        structural_insights,
        confidence,
    })
}

/// Update scratchpad with insights from a reasoning episode.
pub fn update_scratchpad(scratchpad: &mut Scratchpad, episode: &ReasoningEpisode) {
    scratchpad.episodes.push(episode.clone());

    // Add parameter interactions (now stored directly in episode)
    for interaction in &episode.param_interactions {
        if !scratchpad.param_interactions.contains(interaction) {
            scratchpad.param_interactions.push(interaction.clone());
        }
    }

    // Add structural insights
    for insight in &episode.structural_insights {
        if !scratchpad.structural_proposals.contains(insight) {
            scratchpad.structural_proposals.push(insight.clone());
        }
    }
}

/// Apply proposed changes to create a new parameter point.
pub fn apply_changes(
    params: &ParameterPoint,
    changes: &HashMap<String, (String, String, String)>,
) -> ParameterPoint {
    let mut new_params = params.clone();

    for (param_name, (direction, magnitude, _rationale)) in changes {
        let mult = match magnitude.as_str() {
            "small" => 0.1,
            "medium" => 0.3,
            "large" => 1.0,
            _ => 0.3,
        };

        let factor = if direction == "increase" {
            1.0 + mult
        } else {
            1.0 - mult
        };

        // Parse dot-notation for pipeline coordinates: "pipeline.shadow_strategy.weight_name_match"
        // This enables L1 to tune the bicameral LSP resolution system
        if param_name.starts_with("pipeline.") {
            if let Some(ref mut pipeline) = new_params.pipeline {
                let parts: Vec<&str> = param_name.split('.').collect();
                if parts.len() == 3 {
                    let section = parts[1]; // "shadow_strategy" or "final_strategy" or "lsp_policy"
                    let field = parts[2]; // "weight_name_match" etc

                    match section {
                        "shadow_strategy" => match field {
                            "weight_same_file" => {
                                pipeline.shadow_strategy.weight_same_file *= factor
                            }
                            "weight_type_hint" => {
                                pipeline.shadow_strategy.weight_type_hint *= factor
                            }
                            "weight_import" => pipeline.shadow_strategy.weight_import *= factor,
                            "weight_name_match" => {
                                pipeline.shadow_strategy.weight_name_match *= factor
                            }
                            "weight_lsp" => pipeline.shadow_strategy.weight_lsp *= factor,
                            "acceptance_bias" => pipeline.shadow_strategy.acceptance_bias *= factor,
                            "acceptance_slope" => {
                                pipeline.shadow_strategy.acceptance_slope *= factor
                            }
                            "selection_temperature" => {
                                pipeline.shadow_strategy.selection_temperature *= factor
                            }
                            "evidence_accumulation" => {
                                pipeline.shadow_strategy.evidence_accumulation *= factor
                            }
                            "proximity_boost" => pipeline.shadow_strategy.proximity_boost *= factor,
                            _ => {}
                        },
                        "final_strategy" => match field {
                            "weight_same_file" => {
                                pipeline.final_strategy.weight_same_file *= factor
                            }
                            "weight_type_hint" => {
                                pipeline.final_strategy.weight_type_hint *= factor
                            }
                            "weight_import" => pipeline.final_strategy.weight_import *= factor,
                            "weight_name_match" => {
                                pipeline.final_strategy.weight_name_match *= factor
                            }
                            "weight_lsp" => pipeline.final_strategy.weight_lsp *= factor,
                            "acceptance_bias" => pipeline.final_strategy.acceptance_bias *= factor,
                            "acceptance_slope" => {
                                pipeline.final_strategy.acceptance_slope *= factor
                            }
                            "selection_temperature" => {
                                pipeline.final_strategy.selection_temperature *= factor
                            }
                            "evidence_accumulation" => {
                                pipeline.final_strategy.evidence_accumulation *= factor
                            }
                            "proximity_boost" => pipeline.final_strategy.proximity_boost *= factor,
                            _ => {}
                        },
                        "lsp_policy" => match field {
                            "marginal_utility_floor" => {
                                pipeline.lsp_policy.marginal_utility_floor *= factor
                            }
                            "batch_latency_bias" => {
                                pipeline.lsp_policy.batch_latency_bias *= factor
                            }
                            "query_timeout_secs" => {
                                pipeline.lsp_policy.query_timeout_secs *= factor
                            }
                            "max_retries" => pipeline.lsp_policy.max_retries *= factor,
                            "cache_negative_bias" => {
                                pipeline.lsp_policy.cache_negative_bias *= factor
                            }
                            "weight_centrality" => pipeline.lsp_policy.weight_centrality *= factor,
                            "weight_uncertainty" => {
                                pipeline.lsp_policy.weight_uncertainty *= factor
                            }
                            "weight_coherence" => pipeline.lsp_policy.weight_coherence *= factor,
                            "weight_causality" => pipeline.lsp_policy.weight_causality *= factor,
                            "weight_bridge" => pipeline.lsp_policy.weight_bridge *= factor,
                            "spread_logit_structural" => {
                                pipeline.lsp_policy.spread_logit_structural *= factor
                            }
                            "spread_logit_semantic" => {
                                pipeline.lsp_policy.spread_logit_semantic *= factor
                            }
                            "spread_logit_spatial" => {
                                pipeline.lsp_policy.spread_logit_spatial *= factor
                            }
                            "focus_temperature" => pipeline.lsp_policy.focus_temperature *= factor,
                            "gated_threshold" => pipeline.lsp_policy.gated_threshold *= factor,
                            "exploration_floor" => pipeline.lsp_policy.exploration_floor *= factor,
                            "interaction_mixing" => {
                                pipeline.lsp_policy.interaction_mixing *= factor
                            }
                            "centrality_normalization" => {
                                pipeline.lsp_policy.centrality_normalization *= factor
                            }
                            _ => {}
                        },
                        _ => {}
                    }
                }
            }
        } else {
            // Base ranking parameters (no dot notation)
            match param_name.as_str() {
                "pagerank_alpha" => new_params.pagerank_alpha *= factor,
                "pagerank_chat_multiplier" => new_params.pagerank_chat_multiplier *= factor,
                "depth_weight_root" => new_params.depth_weight_root *= factor,
                "depth_weight_moderate" => new_params.depth_weight_moderate *= factor,
                "depth_weight_deep" => new_params.depth_weight_deep *= factor,
                "depth_weight_vendor" => new_params.depth_weight_vendor *= factor,
                "boost_mentioned_ident" => new_params.boost_mentioned_ident *= factor,
                "boost_mentioned_file" => new_params.boost_mentioned_file *= factor,
                "boost_chat_file" => new_params.boost_chat_file *= factor,
                "boost_temporal_coupling" => new_params.boost_temporal_coupling *= factor,
                "boost_focus_expansion" => new_params.boost_focus_expansion *= factor,
                "git_recency_decay_days" => new_params.git_recency_decay_days *= factor,
                "git_recency_max_boost" => new_params.git_recency_max_boost *= factor,
                "git_churn_threshold" => new_params.git_churn_threshold *= factor,
                "git_churn_max_boost" => new_params.git_churn_max_boost *= factor,
                "focus_decay" => new_params.focus_decay *= factor,
                "focus_max_hops" => new_params.focus_max_hops *= factor,
                _ => {}
            }
        }
    }

    new_params
}

/// Distill accumulated insights into operator wisdom.
///
/// ## Phase 1 Agentic Mode (when `run_dir` is provided):
/// - Agent can read episode files to extract deeper patterns
/// - Enables more comprehensive distillation of insights
pub fn distill_scratchpad(
    scratchpad: &Scratchpad,
    agent: Agent,
    model: Option<&str>,
    run_dir: Option<&str>,
) -> Result<String, String> {
    if scratchpad.episodes.is_empty() {
        return Err("No episodes to distill".to_string());
    }

    // Gather all insights
    let all_interactions: Vec<_> = scratchpad.param_interactions.iter().take(20).collect();
    let all_structural: Vec<_> = scratchpad.structural_proposals.iter().take(10).collect();

    // Count proposed changes across episodes
    let mut change_counts: HashMap<String, usize> = HashMap::new();
    for ep in &scratchpad.episodes {
        for (param, (direction, _, _)) in &ep.proposed_changes {
            let key = format!("{}:{}", param, direction);
            *change_counts.entry(key).or_insert(0) += 1;
        }
    }

    let mut sorted_changes: Vec<_> = change_counts.iter().collect();
    sorted_changes.sort_by(|a, b| b.1.cmp(a.1));

    let prompt = format!(
        r#"You are distilling optimization insights into operator wisdom for ripmap.

You have accumulated {} reasoning episodes about the ranking system.

=== DISCOVERED PARAMETER INTERACTIONS ===
{}

=== STRUCTURAL PROPOSALS (beyond tuning) ===
{}

=== MOST COMMON PARAMETER CHANGES ===
{}

=== YOUR TASK ===

Distill these insights into operator wisdom. Return JSON:
{{
  "presets": {{
    "preset_name": {{
      "description": "when to use",
      "key_params": {{"param": value}}
    }}
  }},
  "heuristics": [
    "if CONDITION: ADJUSTMENT - rationale"
  ],
  "warnings": [
    "⚠ what happens and why"
  ],
  "intuitions": [
    "Prose wisdom teaching the tool's use"
  ]
}}
"#,
        scratchpad.episodes.len(),
        all_interactions
            .iter()
            .map(|i| format!("• {}", i))
            .collect::<Vec<_>>()
            .join("\n"),
        all_structural
            .iter()
            .map(|s| format!("• {}", s))
            .collect::<Vec<_>>()
            .join("\n"),
        sorted_changes
            .iter()
            .take(10)
            .map(|(k, v)| format!("• {}: {} episodes", k, v))
            .collect::<Vec<_>>()
            .join("\n"),
    );

    // Phase 1 agentic mode: agent can read episode files if run_dir provided
    call_agent(agent, &prompt, model, run_dir)
}

/// Generate diagnostic alerts from pipeline statistics.
/// Translates metrics into actionable failure mode warnings for L1.
/// These alerts help L1 diagnose structural issues beyond parameter tuning.
fn generate_pipeline_alerts(stats: &PipelineStatsSnapshot) -> String {
    let mut alerts = Vec::new();

    // Shadow Collapse: heuristics failing to connect the graph
    // This prevents PageRank from identifying hubs, crippling LSP query selection
    // Use average_degree (size-invariant) instead of raw density (which breaks on large repos)
    if stats.average_degree < 1.0 {
        alerts.push(format!(
            "• SHADOW COLLAPSE: Average degree {:.2} < 1.0. The shadow graph is disconnected!\n\
             → Each node has less than 1 edge on average - graph is forest-like\n\
             → Check shadow_strategy.weight_name_match - may be too conservative\n\
             → Consider lowering shadow_strategy.acceptance_bias threshold\n\
             → Impact: PageRank has no structure to work with, LSP queries are blind",
            stats.average_degree
        ));
    } else if stats.average_degree < 2.0 {
        alerts.push(format!(
            "• LOW SHADOW CONNECTIVITY: Average degree {:.2}. Graph may be too sparse.\n\
             → Shadow pass should be recall-optimized - consider boosting heuristic weights",
            stats.average_degree
        ));
    }

    // LSP Waste: policy selecting bad query sites
    // Low utilization means queries are issued but fail to resolve
    if stats.lsp_utilization < 0.3 {
        alerts.push(
            "• LOW LSP UTILIZATION: <30% of queries resolving successfully!\n\
             → Policy is selecting sites that LSP can't resolve\n\
             → Check lsp_policy.marginal_utility_floor - too low wastes queries\n\
             → Check lsp_policy.centrality_weight - may be over-indexing on PageRank\n\
             → Impact: Wasting query budget on low-value positions"
                .to_string(),
        );
    } else if stats.lsp_utilization < 0.5 {
        alerts.push(format!(
            "• MODERATE LSP UTILIZATION: {:.1}% success rate. Room for policy improvement.",
            stats.lsp_utilization * 100.0
        ));
    }

    // Rank Divergence: shadow graph doesn't predict final structure
    // This means PageRank on shadow graph is misleading the policy
    if stats.shadow_final_rank_correlation < 0.3 {
        alerts.push(
            "• SEVERE RANK DIVERGENCE: Shadow and final PageRank have <0.3 correlation!\n\
             → Shadow graph structure is NOISE - doesn't reflect true importance\n\
             → PageRank is misleading the LSP query policy\n\
             → Check if shadow_strategy heuristics are creating false edges\n\
             → Consider tightening shadow_strategy.acceptance_bias\n\
             → Impact: Policy thinks unimportant nodes are hubs, wastes queries"
                .to_string(),
        );
    } else if stats.shadow_final_rank_correlation < 0.5 {
        alerts.push(format!(
            "• RANK DIVERGENCE: {:.2} correlation between shadow and final PageRank.\n\
             → Shadow graph is moderately noisy - consider tuning shadow heuristics",
            stats.shadow_final_rank_correlation
        ));
    }

    // Cost Overrun: too many queries or queries taking too long
    if stats.lsp_latency_ms > 10000 {
        alerts.push(format!(
            "• HIGH LATENCY: {:.1}s spent in LSP queries!\n\
             → Reduce query_budget to cap number of queries\n\
             → Increase lsp_policy.marginal_utility_floor to be more selective\n\
             → Impact: Training/production will be slow",
            stats.lsp_latency_ms as f64 / 1000.0
        ));
    }

    if alerts.is_empty() {
        "  ✓ No critical issues - pipeline metrics look healthy".to_string()
    } else {
        alerts.join("\n\n")
    }
}

/// Print a summary of the scratchpad state.
pub fn print_scratchpad_summary(scratchpad: &Scratchpad) {
    println!("\n=== SCRATCHPAD SUMMARY ===\n");
    println!("Episodes: {}", scratchpad.episodes.len());
    println!(
        "Parameter interactions: {}",
        scratchpad.param_interactions.len()
    );
    println!(
        "Structural proposals: {}",
        scratchpad.structural_proposals.len()
    );

    if !scratchpad.param_interactions.is_empty() {
        println!("\nTop Interactions:");
        for interaction in scratchpad.param_interactions.iter().take(5) {
            println!("  • {}", interaction);
        }
    }

    if !scratchpad.structural_proposals.is_empty() {
        println!("\nStructural Proposals:");
        for proposal in scratchpad.structural_proposals.iter().take(5) {
            println!("  • {}", proposal);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_changes() {
        let params = ParameterPoint::default();
        let mut changes = HashMap::new();
        changes.insert(
            "boost_chat_file".to_string(),
            (
                "decrease".to_string(),
                "medium".to_string(),
                "too dominant".to_string(),
            ),
        );

        let new_params = apply_changes(&params, &changes);
        assert!(new_params.boost_chat_file < params.boost_chat_file);
    }
}
