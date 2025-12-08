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
            _ => Err(format!("Unknown agent: {}. Use 'claude', 'gemini', or 'codex'", s)),
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
pub fn call_claude(prompt: &str, model: Option<&str>) -> Result<String, String> {
    let mut args = vec!["--print", "-p", prompt];
    let model_str;
    if let Some(m) = model {
        model_str = m.to_string();
        args.insert(0, "--model");
        args.insert(1, &model_str);
    }

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
pub fn call_gemini(prompt: &str, model: Option<&str>) -> Result<String, String> {
    let mut cmd = Command::new("gemini");
    cmd.args(["-o", "text", "-y"]);
    if let Some(m) = model {
        cmd.args(["-m", m]);
    }
    cmd.arg(prompt);

    let output = cmd.output()
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
pub fn call_codex(prompt: &str, model: Option<&str>) -> Result<String, String> {
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
    args.push("-o".to_string());
    args.push(output_file.to_str().unwrap().to_string());
    args.push("-".to_string());  // read prompt from stdin

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
        stdin.write_all(prompt.as_bytes())
            .map_err(|e| format!("Failed to write to codex stdin: {}", e))?;
    }

    let output = child.wait_with_output()
        .map_err(|e| format!("Failed to wait for codex: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // Clean up temp file
        let _ = std::fs::remove_file(&output_file);
        return Err(format!("Codex returned error: {}", stderr));
    }

    // Read output from file
    let response = std::fs::read_to_string(&output_file)
        .map_err(|e| format!("Failed to read codex output: {}", e))?;

    // Clean up temp file
    let _ = std::fs::remove_file(&output_file);

    Ok(response.trim().to_string())
}

/// Call the specified LLM agent and return response.
pub fn call_agent(agent: Agent, prompt: &str, model: Option<&str>) -> Result<String, String> {
    match agent {
        Agent::Claude => call_claude(prompt, model),
        Agent::Gemini => call_gemini(prompt, model),
        Agent::Codex => call_codex(prompt, model),
    }
}

/// Reason about ranking failures and propose parameter changes.
///
/// This is where the LLM acts as a universal function approximator:
/// f(failures, params, history) -> (reasoning, proposals, insights)
///
/// The `prompt_template` should contain placeholders:
/// - `{current_ndcg:.4}` - current NDCG score
/// - `{episode_num}` - current episode number
/// - `{episode_history}` - formatted history of recent episodes
/// - `{params_desc}` - current parameter values
/// - `{failure_desc}` - formatted failure cases
///
/// Supports multiple agents via the `agent` parameter.
/// Optionally specify a model (e.g., "opus", "o3", "gemini-2.0-flash").
pub fn reason_about_failures(
    prompt_template: &str,
    failures: &[RankingFailure],
    current_params: &ParameterPoint,
    scratchpad: &Scratchpad,
    current_ndcg: f64,
    agent: Agent,
    model: Option<&str>,
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
                f.expected_top.iter().take(5).cloned().collect::<Vec<_>>().join(", "),
                f.actual_top.iter().take(5).cloned().collect::<Vec<_>>().join(", "),
                f.ndcg,
                f.commit_context,
                f.repo_name,
                f.repo_file_count
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    // Format current parameters
    let params_desc = format!(
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

    // Build FULL episode history - the model needs to see the trajectory
    let episode_history = if scratchpad.episodes.is_empty() {
        "This is the FIRST episode. No history yet.".to_string()
    } else {
        let mut history = String::new();
        let recent_episodes: Vec<_> = scratchpad.episodes.iter().rev().take(10).collect();

        // Show NDCG trajectory with strategy intent - the model sees both WHAT happened and WHY
        history.push_str("EPISODE HISTORY (recent → older):\n");
        for (i, ep) in recent_episodes.iter().enumerate() {
            let trend = if i == 0 { "" } else {
                let prev_ndcg = recent_episodes.get(i - 1).map(|e| e.ndcg_before).unwrap_or(0.0);
                if ep.ndcg_before > prev_ndcg + 0.01 { " ↗" }
                else if ep.ndcg_before < prev_ndcg - 0.01 { " ↘" }
                else { " →" }
            };
            let ep_num = scratchpad.episodes.len() - i;
            let strategy = if ep.strategy_capsule.is_empty() {
                String::new()
            } else {
                format!("\n      Strategy: \"{}\"", ep.strategy_capsule.chars().take(100).collect::<String>())
            };
            history.push_str(&format!("  E{}: NDCG={:.3}{} | failures={}{}\n",
                ep_num, ep.ndcg_before, trend, ep.failures.len(), strategy));
        }

        // Show recent parameter changes - THE GRADIENT
        history.push_str("\nRECENT PARAMETER CHANGES:\n");
        for (i, ep) in recent_episodes.iter().take(5).enumerate() {
            let ep_num = scratchpad.episodes.len() - i;
            if !ep.proposed_changes.is_empty() {
                let changes: Vec<_> = ep.proposed_changes.iter()
                    .map(|(k, (dir, mag, _))| format!("{} {} {}", k, dir, mag))
                    .take(3)
                    .collect();
                history.push_str(&format!("  E{}: {} (conf={:.2})\n",
                    ep_num, changes.join(", "), ep.confidence));
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
                history.push_str(&format!("  ALERT: NDCG dropped {:.3} over last {} episodes!\n", -trend, recent_episodes.len()));
                history.push_str("  Consider: Are recent changes making things WORSE?\n");
                history.push_str("  Consider: Should we REVERT to earlier params?\n");
            } else if trend > 0.02 {
                history.push_str(&format!("  Good: NDCG improved {:.3} - current direction is working\n", trend));
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

    // Inject dynamic context into the prompt template
    let prompt = prompt_template
        .replace("{current_ndcg:.4}", &format!("{:.4}", current_ndcg))
        .replace("{episode_num}", &(scratchpad.episodes.len() + 1).to_string())
        .replace("{episode_history}", &episode_history)
        .replace("{params_desc}", &params_desc)
        .replace("{failure_desc}", &failure_desc);

    let response = call_agent(agent, &prompt, model)?;

    // Extract the reasoning section (everything before JSON:) for storage
    let reasoning_text = if let Some(json_marker) = response.find("JSON:") {
        response[..json_marker].replace("REASONING:", "").trim().to_string()
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
        .map_err(|e| format!("Failed to parse response as JSON: {}\nResponse: {}", e, &response[..response.len().min(500)]))?;

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
        reasoning: reasoning_text,  // Store the free-form reasoning, not the raw JSON
        strategy_capsule,           // The intent/mode for this episode
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
pub fn apply_changes(params: &ParameterPoint, changes: &HashMap<String, (String, String, String)>) -> ParameterPoint {
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

    new_params
}

/// Distill accumulated insights into operator wisdom.
pub fn distill_scratchpad(scratchpad: &Scratchpad, agent: Agent, model: Option<&str>) -> Result<String, String> {
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
        all_interactions.iter().map(|i| format!("• {}", i)).collect::<Vec<_>>().join("\n"),
        all_structural.iter().map(|s| format!("• {}", s)).collect::<Vec<_>>().join("\n"),
        sorted_changes.iter().take(10).map(|(k, v)| format!("• {}: {} episodes", k, v)).collect::<Vec<_>>().join("\n"),
    );

    call_agent(agent, &prompt, model)
}

/// Print a summary of the scratchpad state.
pub fn print_scratchpad_summary(scratchpad: &Scratchpad) {
    println!("\n=== SCRATCHPAD SUMMARY ===\n");
    println!("Episodes: {}", scratchpad.episodes.len());
    println!("Parameter interactions: {}", scratchpad.param_interactions.len());
    println!("Structural proposals: {}", scratchpad.structural_proposals.len());

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
            ("decrease".to_string(), "medium".to_string(), "too dominant".to_string()),
        );

        let new_params = apply_changes(&params, &changes);
        assert!(new_params.boost_chat_file < params.boost_chat_file);
    }
}
