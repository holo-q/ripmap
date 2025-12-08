//! Mesa Optimizer: Unified abstraction for inner/outer optimization loops.
//!
//! The same pattern applies at both levels:
//! - **Inner (L1)**: Optimizes params θ based on ranking failures
//! - **Outer (L2)**: Optimizes prompts P based on inner run summaries
//!
//! Both levels:
//! 1. Observe current state (failures or run summaries)
//! 2. Reason about what to change (params or prompt sections)
//! 3. Propose changes with rationale
//! 4. Accumulate insights in a scratchpad
//!
//! The outer loop wraps the inner loop - when running L2, each outer episode
//! spawns multiple inner episodes. Directory structure auto-adjusts:
//!
//! ```text
//! training-outer/runs/<outer_run>/
//!   outer_scratchpad.json
//!   config.toml
//!   inner_runs/
//!     step_001/              # Inner run for outer step 1
//!       scratchpad.json
//!       results.json
//!       progress.png
//!     step_002/
//!       ...
//! ```

use std::path::{Path, PathBuf};
use serde::{de::DeserializeOwned, Serialize};

use super::schemas::*;
use super::promptgram::Promptgram;

/// A proposal from a mesa optimizer.
///
/// Generic over the type of changes being proposed.
pub trait Proposal: Serialize + DeserializeOwned {
    /// Get the confidence level of this proposal.
    fn confidence(&self) -> f64;

    /// Get the mode/intent of this proposal (explore, exploit, consolidate).
    fn mode(&self) -> &str;

    /// Get the strategy capsule summarizing intent.
    fn strategy_capsule(&self) -> &str;
}

/// Scratchpad for accumulating insights across episodes.
pub trait Scratchpad: Default + Serialize + DeserializeOwned {
    /// Number of episodes recorded.
    fn episode_count(&self) -> usize;

    /// Save to a JSON file.
    fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize scratchpad: {}", e))?;
        std::fs::write(path.as_ref(), content)
            .map_err(|e| format!("Failed to write scratchpad: {}", e))
    }

    /// Load from a JSON file.
    fn load(path: impl AsRef<Path>) -> Result<Self, String>
    where
        Self: Sized,
    {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| format!("Failed to read scratchpad: {}", e))?;
        serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse scratchpad: {}", e))
    }
}

/// Configuration for a mesa optimizer run.
#[derive(Debug, Clone)]
pub struct RunConfig {
    /// Name of this run (used for directory naming)
    pub run_name: String,

    /// Base directory for outputs
    pub base_dir: PathBuf,

    /// Number of episodes to run
    pub episodes: usize,

    /// Which agent to use
    pub agent: crate::training::reasoning::Agent,

    /// Optional model override
    pub model: Option<String>,

    /// Save interval (checkpoint every N episodes)
    pub save_interval: usize,
}

impl RunConfig {
    /// Get the output directory for this run.
    pub fn output_dir(&self) -> PathBuf {
        self.base_dir.join(&self.run_name)
    }

    /// Get the scratchpad path.
    pub fn scratchpad_path(&self) -> PathBuf {
        self.output_dir().join("scratchpad.json")
    }

    /// Get the results path.
    pub fn results_path(&self) -> PathBuf {
        self.output_dir().join("results.json")
    }

    /// Get the plot path.
    pub fn plot_path(&self) -> PathBuf {
        self.output_dir().join("progress.png")
    }

    /// Create inner run config for a given outer step.
    ///
    /// When running under L2, inner runs go into `inner_runs/step_NNN/`.
    pub fn inner_run_config(&self, outer_step: usize, inner_episodes: usize) -> RunConfig {
        RunConfig {
            run_name: format!("step_{:03}", outer_step),
            base_dir: self.output_dir().join("inner_runs"),
            episodes: inner_episodes,
            agent: self.agent,
            model: self.model.clone(),
            save_interval: self.save_interval,
        }
    }
}

/// Run context passed to the optimizer.
///
/// Contains paths and configuration for the current run.
pub struct RunContext {
    /// Configuration for this run
    pub config: RunConfig,

    /// Current episode number
    pub episode: usize,

    /// Is this an outer loop run?
    pub is_outer: bool,

    /// Parent context (if running inner under outer)
    pub parent_step: Option<usize>,
}

impl RunContext {
    /// Create directories for this run.
    pub fn setup_dirs(&self) -> Result<(), String> {
        std::fs::create_dir_all(self.config.output_dir())
            .map_err(|e| format!("Failed to create output directory: {}", e))?;

        if self.is_outer {
            std::fs::create_dir_all(self.config.output_dir().join("inner_runs"))
                .map_err(|e| format!("Failed to create inner_runs directory: {}", e))?;
        }

        Ok(())
    }

    /// Get path for inner run results.
    pub fn inner_run_path(&self, step: usize) -> PathBuf {
        self.config.output_dir()
            .join("inner_runs")
            .join(format!("step_{:03}", step))
    }
}

/// The MesaOptimizer trait: unified interface for inner/outer loops.
///
/// Type parameters:
/// - `State`: The state being optimized (ParameterPoint for L1, Promptgram for L2)
/// - `Observation`: What we observe (RankingFailures for L1, OuterEpisodeSummary for L2)
/// - `P`: The proposal type
/// - `S`: The scratchpad type
pub trait MesaOptimizer {
    /// The state being optimized.
    type State: Clone + Serialize + DeserializeOwned;

    /// What we observe to inform optimization.
    type Observation;

    /// Proposals for changes.
    type Proposal: Proposal;

    /// Scratchpad for accumulating insights.
    type Scratchpad: Scratchpad;

    /// Initialize default state.
    fn default_state() -> Self::State;

    /// Reason about observations and propose changes.
    fn reason(
        &self,
        state: &Self::State,
        observations: &[Self::Observation],
        scratchpad: &Self::Scratchpad,
        ctx: &RunContext,
    ) -> Result<Self::Proposal, String>;

    /// Apply a proposal to get new state.
    fn apply(&self, state: &Self::State, proposal: &Self::Proposal) -> Self::State;

    /// Update scratchpad with results of an episode.
    fn update_scratchpad(
        &self,
        scratchpad: &mut Self::Scratchpad,
        state_before: &Self::State,
        proposal: &Self::Proposal,
        state_after: &Self::State,
        metrics_delta: f64,
    );

    /// Evaluate current state, returning a metric (higher = better).
    fn evaluate(&self, state: &Self::State) -> Result<(f64, Vec<Self::Observation>), String>;
}

/// Selection mode for choosing promptgram from population.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionMode {
    /// Always pick the best-performing promptgram
    Best,
    /// Randomly explore a non-best promptgram
    Explore,
    /// Pick the most recently created (newest mutations)
    Recent,
}

impl SelectionMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            SelectionMode::Best => "best",
            SelectionMode::Explore => "explore",
            SelectionMode::Recent => "recent",
        }
    }
}

/// Outer loop: wraps inner loop, evolves promptgrams.
pub struct OuterLoop {
    /// Configuration for inner runs
    pub inner_config: OuterConfig,

    /// Current population of promptgrams
    pub population: Vec<Promptgram>,

    /// The meta-promptgram (P_outer) for L2 reasoning
    pub meta_promptgram: Promptgram,

    /// Directory where evolved promptgrams are persisted
    pub promptgram_dir: PathBuf,
}

impl OuterLoop {
    /// Create a new outer loop with default configuration.
    pub fn new() -> Self {
        let promptgram_dir = PathBuf::from("training-outer/prompts/inner");
        OuterLoop {
            inner_config: OuterConfig::default(),
            population: vec![super::promptgram::baseline_promptgram()],
            meta_promptgram: create_meta_promptgram(),
            promptgram_dir,
        }
    }

    /// Select a promptgram from the population.
    ///
    /// Selection strategy balances exploitation (best performers) with exploration
    /// (trying newer or less-tested prompts). The exploration_quota in config
    /// controls how often we explore vs exploit.
    pub fn select_promptgram(
        &self,
        outer_scratchpad: &OuterScratchpad,
    ) -> (Promptgram, SelectionMode) {
        use rand::Rng;

        if self.population.is_empty() {
            panic!("Population is empty - cannot select promptgram");
        }

        if self.population.len() == 1 {
            // Only one option
            return (self.population[0].clone(), SelectionMode::Best);
        }

        let mut rng = rand::thread_rng();
        let should_explore = rng.r#gen::<f64>() < self.inner_config.exploration_quota;

        if should_explore {
            // Exploration: pick from non-best candidates
            // Prefer less-tested promptgrams or recent mutations
            let best_id = &outer_scratchpad.best_prompt_id;
            let candidates: Vec<&Promptgram> = self.population
                .iter()
                .filter(|p| &p.id != best_id)
                .collect();

            if candidates.is_empty() {
                // All are "best" (only one promptgram evaluated so far)
                return (self.population.last().unwrap().clone(), SelectionMode::Recent);
            }

            // Weight by recency and inverse run count
            let weights: Vec<f64> = candidates.iter().map(|p| {
                let stats = outer_scratchpad.promptgram_stats(&p.id);
                let run_count = stats.map(|s| s.run_count).unwrap_or(0);
                // Never-run prompts get high weight, heavily-run get low weight
                1.0 / (run_count as f64 + 1.0)
            }).collect();

            let total: f64 = weights.iter().sum();
            let mut pick = rng.r#gen::<f64>() * total;

            for (i, w) in weights.iter().enumerate() {
                pick -= w;
                if pick <= 0.0 {
                    return (candidates[i].clone(), SelectionMode::Explore);
                }
            }

            // Fallback to last candidate
            (candidates.last().unwrap().clone().clone(), SelectionMode::Explore)
        } else {
            // Exploitation: pick the best-performing promptgram
            let best_id = &outer_scratchpad.best_prompt_id;
            let best = self.population
                .iter()
                .find(|p| &p.id == best_id)
                .or_else(|| self.population.first())
                .unwrap();
            (best.clone(), SelectionMode::Best)
        }
    }

    /// Persist a promptgram to disk as markdown.
    ///
    /// Promptgrams are saved to `training-outer/prompts/inner/<id>.md`.
    pub fn persist_promptgram(&self, promptgram: &Promptgram) -> Result<PathBuf, String> {
        std::fs::create_dir_all(&self.promptgram_dir)
            .map_err(|e| format!("Failed to create promptgram dir: {}", e))?;

        let filename = format!("{}.md", promptgram.id);
        let path = self.promptgram_dir.join(&filename);

        std::fs::write(&path, promptgram.render())
            .map_err(|e| format!("Failed to write promptgram: {}", e))?;

        // Also save TOML metadata alongside
        let meta_path = self.promptgram_dir.join(format!("{}.toml", promptgram.id));
        promptgram.save(&meta_path)?;

        Ok(path)
    }

    /// Load all promptgrams from disk into the population.
    pub fn load_promptgrams(&mut self) -> Result<usize, String> {
        if !self.promptgram_dir.exists() {
            return Ok(0);
        }

        let mut loaded = 0;
        for entry in std::fs::read_dir(&self.promptgram_dir)
            .map_err(|e| format!("Failed to read promptgram dir: {}", e))?
        {
            let entry = entry.map_err(|e| format!("Failed to read dir entry: {}", e))?;
            let path = entry.path();

            // Load from TOML files (they have full metadata)
            if path.extension().map(|e| e == "toml").unwrap_or(false) {
                match Promptgram::load(&path) {
                    Ok(pg) => {
                        // Check if already in population
                        if !self.population.iter().any(|p| p.id == pg.id) {
                            self.population.push(pg);
                            loaded += 1;
                        }
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to load promptgram {:?}: {}", path, e);
                    }
                }
            }
        }

        Ok(loaded)
    }

    /// Run one outer episode.
    ///
    /// 1. Select a promptgram from population (explore/exploit)
    /// 2. Run K inner episodes with that promptgram
    /// 3. Summarize the inner run
    /// 4. Invoke L2 to propose promptgram edits
    /// 5. Update population and persist new promptgrams
    pub fn run_outer_episode(
        &mut self,
        outer_step: usize,
        ctx: &RunContext,
        outer_scratchpad: &mut OuterScratchpad,
    ) -> Result<OuterEpisodeSummary, String> {
        let episode_start = std::time::Instant::now();

        // 1. Select promptgram using selection strategy
        let (promptgram, selection_mode) = self.select_promptgram(outer_scratchpad);
        println!("  📋 Selected promptgram: {} (mode: {})", promptgram.id, selection_mode.as_str());

        // 2. Create inner run config
        let inner_ctx = RunContext {
            config: ctx.config.inner_run_config(outer_step, self.inner_config.inner_episodes),
            episode: 0,
            is_outer: false,
            parent_step: Some(outer_step),
        };
        inner_ctx.setup_dirs()?;

        // 3. Run inner episodes (this would call ripmap-train internally)
        // For Stage 0, we'll shell out to the existing binary
        let inner_result = self.run_inner_loop(&promptgram, &inner_ctx)?;

        // 4. Summarize the run (includes selection mode)
        let mut summary = self.summarize_inner_run(outer_step, &promptgram, &inner_result, &episode_start)?;
        summary.selection_mode = selection_mode.as_str().to_string();

        // 5. Update scratchpad first (so L2 can see this episode)
        if summary.final_metrics.ndcg > outer_scratchpad.best_ndcg {
            outer_scratchpad.best_ndcg = summary.final_metrics.ndcg;
            outer_scratchpad.best_prompt_id = promptgram.id.clone();
        }

        // 6. Invoke L2 reasoning if enabled
        if self.inner_config.edit_prompts {
            // Get recent summaries for context (last 3)
            let recent: Vec<&OuterEpisodeSummary> = outer_scratchpad
                .recent_episodes(3)
                .into_iter()
                .rev()  // Oldest first
                .collect();

            // Parse outer agent
            let outer_agent: crate::training::reasoning::Agent = self.inner_config.outer_agent
                .parse()
                .map_err(|e: String| e)?;

            // Call L2 reasoning with full history context
            match self.reason_about_prompt(&promptgram, &recent, outer_scratchpad, outer_agent) {
                Ok(proposal) => {
                    println!("  📝 L2 proposal: mode={}, confidence={:.2}", proposal.mode, proposal.confidence);
                    println!("     {} edits proposed", proposal.edits.len());
                    for edit in &proposal.edits {
                        println!("       • {}: {} in {}", edit.edit_type, edit.section,
                            if edit.target.is_empty() { "(new)" } else { &edit.target });
                    }

                    // Track proposal in summary
                    summary.proposal = Some(proposal.clone());

                    // Apply edits to create new promptgram
                    if !proposal.edits.is_empty() {
                        let new_id = format!("inner_v{:03}", self.population.len() + 1);
                        let mut new_promptgram = promptgram.fork(&new_id);

                        let mut edits_applied = 0;
                        for edit in &proposal.edits {
                            if let Err(e) = new_promptgram.apply_edit(edit) {
                                println!("     ⚠️ Edit failed: {}", e);
                            } else {
                                edits_applied += 1;
                            }
                        }

                        if edits_applied > 0 {
                            // Persist the new promptgram to disk
                            match self.persist_promptgram(&new_promptgram) {
                                Ok(path) => {
                                    println!("     💾 Persisted to {:?}", path);
                                }
                                Err(e) => {
                                    println!("     ⚠️ Failed to persist: {}", e);
                                }
                            }

                            // Add to population
                            self.population.push(new_promptgram);
                            println!("     ✓ Created new promptgram: {}", new_id);
                        }
                    }
                }
                Err(e) => {
                    println!("  ⚠️ L2 reasoning failed: {}", e);
                }
            }
        }

        // 7. Push summary to scratchpad (after L2 proposal is attached)
        outer_scratchpad.episodes.push(summary.clone());

        Ok(summary)
    }

    /// Run the inner loop with a given promptgram.
    ///
    /// For Stage 0, we shell out to `ripmap-train`. Later this could be
    /// integrated directly.
    fn run_inner_loop(
        &self,
        promptgram: &Promptgram,
        ctx: &RunContext,
    ) -> Result<InnerRunResult, String> {
        use std::process::Command;

        let output_dir = ctx.config.output_dir();

        // Determine prompt path: either use configured template or render promptgram
        let prompt_path = if let Some(template_path) = &self.inner_config.prompt_template_path {
            template_path.clone()
        } else {
            // Render promptgram to a temp file in the output directory
            let prompt_file = output_dir.join("prompt.md");
            std::fs::write(&prompt_file, promptgram.render())
                .map_err(|e| format!("Failed to write prompt file: {}", e))?;
            prompt_file.to_str().unwrap().to_string()
        };

        // Build command
        let mut cmd = Command::new("./target/release/ripmap-train");
        cmd.args([
            "--corpus", &self.inner_config.corpus,
            "--reason",
            "--prompt", &prompt_path,
            "--episodes", &ctx.config.episodes.to_string(),
            "--agent", &ctx.config.agent.to_string(),
            "--output", output_dir.join("results.json").to_str().unwrap(),
            "--scratchpad", output_dir.join("scratchpad.json").to_str().unwrap(),
            "--plot", output_dir.join("progress.png").to_str().unwrap(),
        ]);

        // Run and capture output
        let output = cmd.output()
            .map_err(|e| format!("Failed to run inner loop: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Inner loop failed: {}", stderr));
        }

        // Load results
        let scratchpad_path = output_dir.join("scratchpad.json");
        let scratchpad: crate::training::reasoning::Scratchpad =
            serde_json::from_str(&std::fs::read_to_string(&scratchpad_path)
                .map_err(|e| format!("Failed to read scratchpad: {}", e))?)
            .map_err(|e| format!("Failed to parse scratchpad: {}", e))?;

        // Extract metrics from scratchpad
        let first_ndcg = scratchpad.episodes.first()
            .map(|e| e.ndcg_before)
            .unwrap_or(0.0);
        let final_ndcg = scratchpad.episodes.last()
            .map(|e| e.ndcg_before)
            .unwrap_or(0.0);

        let strategy_capsules: Vec<String> = scratchpad.episodes.iter()
            .filter(|e| !e.strategy_capsule.is_empty())
            .map(|e| e.strategy_capsule.clone())
            .collect();

        let mean_confidence = if scratchpad.episodes.is_empty() {
            0.0
        } else {
            scratchpad.episodes.iter().map(|e| e.confidence).sum::<f64>()
                / scratchpad.episodes.len() as f64
        };

        Ok(InnerRunResult {
            baseline_ndcg: first_ndcg,
            final_ndcg,
            episodes: scratchpad.episodes.len(),
            mean_confidence,
            strategy_capsules,
            structural_insights: scratchpad.structural_proposals.clone(),
            scratchpad,
        })
    }

    /// Summarize an inner run into an OuterEpisodeSummary.
    fn summarize_inner_run(
        &self,
        outer_step: usize,
        promptgram: &Promptgram,
        result: &InnerRunResult,
        start_time: &std::time::Instant,
    ) -> Result<OuterEpisodeSummary, String> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        // Estimate meta-levers from final params
        let final_params = result.scratchpad.episodes.last()
            .map(|e| &e.params)
            .cloned()
            .unwrap_or_default();
        let meta_levers = MetaLevers::from_params(&final_params);

        // Count failures from last episode
        let final_failures = result.scratchpad.episodes.last()
            .map(|e| e.failures.len())
            .unwrap_or(0);

        let baseline = RunMetrics {
            ndcg: result.baseline_ndcg,
            failures: 10, // Approximate initial
            mean_confidence: 0.5,
        };

        let final_metrics = RunMetrics {
            ndcg: result.final_ndcg,
            failures: final_failures,
            mean_confidence: result.mean_confidence,
        };

        // Compute stability metrics
        let ndcg_values: Vec<f64> = result.scratchpad.episodes.iter()
            .map(|e| e.ndcg_before)
            .collect();
        let ndcg_variance = variance(&ndcg_values);

        let collapse_events = ndcg_values.windows(2)
            .filter(|w| w[1] < w[0] - 0.05)
            .count();

        let oscillations = ndcg_values.windows(3)
            .filter(|w| (w[1] > w[0] && w[2] < w[1]) || (w[1] < w[0] && w[2] > w[1]))
            .count();

        Ok(OuterEpisodeSummary {
            outer_step,
            prompt_id: promptgram.id.clone(),
            baseline_metrics: baseline.clone(),
            final_metrics: final_metrics.clone(),
            delta: final_metrics - baseline,
            stability: StabilityMetrics {
                collapse_events,
                ndcg_variance,
                converged: result.final_ndcg > 0.85 && ndcg_variance < 0.01,
                oscillations,
            },
            meta_levers_estimate: meta_levers,
            strategy_capsules: result.strategy_capsules.clone(),
            notable_failures: vec![], // TODO: extract from inner scratchpad
            structural_insights: result.structural_insights.clone(),
            inner_episodes: result.episodes,
            duration_secs: start_time.elapsed().as_secs_f64(),
            timestamp: now,
            proposal: None, // Will be set after L2 reasoning
            selection_mode: String::new(), // Will be set by caller
        })
    }

    /// Invoke L2 reasoning to propose prompt edits.
    ///
    /// The outer agent analyzes recent episodes and proposes changes to the
    /// inner promptgram to improve optimization performance.
    fn reason_about_prompt(
        &self,
        current_promptgram: &Promptgram,
        recent_summaries: &[&OuterEpisodeSummary],
        outer_scratchpad: &OuterScratchpad,
        outer_agent: crate::training::reasoning::Agent,
    ) -> Result<OuterProposal, String> {
        use crate::training::call_agent;

        // Build the L2 prompt with pre-queried history context
        let prompt = self.build_l2_prompt(current_promptgram, recent_summaries, outer_scratchpad);

        println!("  🧠 Invoking L2 reasoning ({})...", outer_agent);

        // Call the outer agent
        let response = call_agent(outer_agent, &prompt, None)?;

        // Parse the response
        self.parse_l2_response(&response)
    }

    /// Build the prompt for L2 reasoning.
    ///
    /// Injects pre-queried history context so L2 can reason about patterns
    /// without needing interactive tool access.
    fn build_l2_prompt(
        &self,
        current_promptgram: &Promptgram,
        recent_summaries: &[&OuterEpisodeSummary],
        outer_scratchpad: &OuterScratchpad,
    ) -> String {
        let mut prompt = String::new();

        // Add meta-promptgram (L2's instructions)
        prompt.push_str(&self.meta_promptgram.render());
        prompt.push_str("\n\n");

        // Add current inner promptgram
        prompt.push_str("=== CURRENT INNER PROMPTGRAM ===\n");
        prompt.push_str(&format!("ID: {}\n", current_promptgram.id));
        prompt.push_str(&format!("Version: {}\n", current_promptgram.version));
        if let Some(ref parent) = current_promptgram.parent_id {
            prompt.push_str(&format!("Parent: {}\n", parent));
        }
        prompt.push_str("\n");
        prompt.push_str(&current_promptgram.render());
        prompt.push_str("\n\n");

        // === PRE-QUERIED HISTORY CONTEXT ===
        // This replaces interactive tool calls - we inject the context L2 needs
        prompt.push_str("=== HISTORY CONTEXT (auto-queried) ===\n\n");

        // Promptgram stats
        if let Some(stats) = outer_scratchpad.promptgram_stats(&current_promptgram.id) {
            prompt.push_str(&format!("Promptgram '{}' stats:\n", stats.prompt_id));
            prompt.push_str(&format!("  Runs: {} | Mean NDCG: {:.4} | Best: {:.4} | Worst: {:.4}\n",
                stats.run_count, stats.mean_ndcg, stats.best_ndcg, stats.worst_ndcg));
            prompt.push_str(&format!("  Active from step {} to {}\n\n", stats.first_step, stats.last_step));
        }

        // Diff against parent if exists
        if let Some(ref parent_id) = current_promptgram.parent_id {
            if let Some(parent) = self.population.iter().find(|p| &p.id == parent_id) {
                let diffs = super::diff_prompts(parent, current_promptgram);
                if !diffs.is_empty() {
                    prompt.push_str(&format!("Changes from parent '{}':\n", parent_id));
                    for diff in &diffs {
                        prompt.push_str(&format!("  {}\n", diff.summary()));
                    }
                    prompt.push_str("\n");
                }
            }
        }

        // Search for common failure patterns
        let failure_searches = ["depth", "temporal", "boost", "collapse", "oscillat"];
        let mut found_patterns = Vec::new();
        for pattern in failure_searches {
            let matches = outer_scratchpad.search_failures(pattern);
            if !matches.is_empty() {
                found_patterns.push((pattern, matches.len()));
            }
        }
        if !found_patterns.is_empty() {
            prompt.push_str("Recurring failure patterns:\n");
            for (pattern, count) in found_patterns {
                prompt.push_str(&format!("  '{}': {} episodes\n", pattern, count));
            }
            prompt.push_str("\n");
        }

        // Population diversity
        let unique_prompts = outer_scratchpad.unique_promptgrams();
        prompt.push_str(&format!("Population: {} unique promptgrams tested\n", unique_prompts.len()));
        if unique_prompts.len() > 1 {
            prompt.push_str("  IDs: ");
            prompt.push_str(&unique_prompts.join(", "));
            prompt.push_str("\n");
        }
        prompt.push_str("\n");

        // Extreme meta-lever episodes (what's been tried)
        let extreme_exploration = outer_scratchpad.extreme_lever_episodes("exploration_bias");
        let extreme_structural = outer_scratchpad.extreme_lever_episodes("structural_trust");
        if !extreme_exploration.is_empty() || !extreme_structural.is_empty() {
            prompt.push_str("Extreme lever episodes (what's been explored):\n");
            for (step, val, _) in extreme_exploration.iter().take(3) {
                prompt.push_str(&format!("  Step {}: exploration_bias={:.2}\n", step, val));
            }
            for (step, val, _) in extreme_structural.iter().take(3) {
                prompt.push_str(&format!("  Step {}: structural_trust={:.2}\n", step, val));
            }
            prompt.push_str("\n");
        }

        // Add recent episode summaries
        prompt.push_str("=== RECENT OUTER EPISODES ===\n");
        for (i, summary) in recent_summaries.iter().enumerate() {
            prompt.push_str(&format!("\n--- Episode {} (step {}) [{}] ---\n",
                i + 1, summary.outer_step, summary.selection_mode));
            prompt.push_str(&format!("NDCG: {:.4} → {:.4} (Δ{:+.4})\n",
                summary.baseline_metrics.ndcg,
                summary.final_metrics.ndcg,
                summary.delta.ndcg));
            prompt.push_str(&format!("Failures: {} → {}\n",
                summary.baseline_metrics.failures,
                summary.final_metrics.failures));
            prompt.push_str(&format!("Stability: {} collapses, {:.4} variance, {} oscillations\n",
                summary.stability.collapse_events,
                summary.stability.ndcg_variance,
                summary.stability.oscillations));
            prompt.push_str(&format!("Converged: {}\n", summary.stability.converged));

            // Meta-levers
            let ml = &summary.meta_levers_estimate;
            prompt.push_str(&format!("Meta-levers: struct={:.2} temp={:.2} explore={:.2} depth={:.2} hub={:.2} focus={:.2}\n",
                ml.structural_trust, ml.temporal_horizon, ml.exploration_bias,
                ml.depth_sensitivity, ml.hub_damping, ml.focus_locality));

            // Strategy capsules
            if !summary.strategy_capsules.is_empty() {
                prompt.push_str("Strategy capsules:\n");
                for capsule in summary.strategy_capsules.iter().take(5) {
                    prompt.push_str(&format!("  • {}\n", capsule));
                }
            }

            // Structural insights
            if !summary.structural_insights.is_empty() {
                prompt.push_str("Structural insights:\n");
                for insight in summary.structural_insights.iter().take(3) {
                    prompt.push_str(&format!("  • {}\n", insight));
                }
            }

            // Previous proposal (if any)
            if let Some(ref proposal) = summary.proposal {
                prompt.push_str(&format!("Previous L2 proposal: mode={}, confidence={:.2}, {} edits\n",
                    proposal.mode, proposal.confidence, proposal.edits.len()));
            }
        }

        // Add trajectory summary
        if recent_summaries.len() >= 2 {
            let first = recent_summaries.first().unwrap();
            let last = recent_summaries.last().unwrap();
            let trend = last.final_metrics.ndcg - first.final_metrics.ndcg;
            prompt.push_str(&format!("\n=== TRAJECTORY ===\n"));
            prompt.push_str(&format!("Overall trend: {:+.4} NDCG over {} episodes\n", trend, recent_summaries.len()));

            // Check for plateau/collapse using scratchpad methods
            if outer_scratchpad.is_collapse(3, 0.02) {
                prompt.push_str("⚠️ COLLAPSE DETECTED - NDCG degrading over last 3 episodes\n");
            } else if outer_scratchpad.is_plateau(3, 0.01) {
                prompt.push_str("⚠️ PLATEAU DETECTED - no improvement in last 3 episodes\n");
            } else if trend > 0.02 {
                prompt.push_str("✓ IMPROVING - continue current direction\n");
            }

            // Best vs current
            prompt.push_str(&format!("Best NDCG ever: {:.4} (prompt: {})\n",
                outer_scratchpad.best_ndcg, outer_scratchpad.best_prompt_id));
        }

        prompt.push_str("\n=== YOUR TASK ===\n");
        prompt.push_str("Analyze the trajectory and history context, then propose edits to the inner promptgram.\n");
        prompt.push_str("Consider: What patterns recur? What hasn't been tried? What's the risk?\n");
        prompt.push_str("Output your reasoning, then a JSON block with your proposal.\n");

        prompt
    }

    /// Parse the L2 response into an OuterProposal.
    fn parse_l2_response(&self, response: &str) -> Result<OuterProposal, String> {
        // Find JSON block in response
        let json_start = response.find('{')
            .ok_or("No JSON found in L2 response")?;

        // Find matching closing brace
        let mut depth = 0;
        let mut json_end = json_start;
        for (i, c) in response[json_start..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        json_end = json_start + i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }

        let json_str = &response[json_start..json_end];

        // Parse JSON
        serde_json::from_str(json_str)
            .map_err(|e| format!("Failed to parse L2 JSON: {} in: {}", e, json_str))
    }
}

/// Result from running the inner loop.
struct InnerRunResult {
    baseline_ndcg: f64,
    final_ndcg: f64,
    episodes: usize,
    mean_confidence: f64,
    strategy_capsules: Vec<String>,
    structural_insights: Vec<String>,
    scratchpad: crate::training::reasoning::Scratchpad,
}

/// Create the meta-promptgram (P_outer) for L2 reasoning.
///
/// This is the prompt that evolves inner prompts.
pub fn create_meta_promptgram() -> Promptgram {
    Promptgram::new("outer_default_v1")
        .with_section("Role", r#"You are a META-OPTIMIZER that evolves prompts for hyperparameter tuning.

You sit at Level 2 of the optimization stack:
- L0: The code ranker being tuned (17 params, NDCG metric)
- L1: Inner optimizer (the prompts you're evolving)
- L2: You - the promptgram optimizer

Your goal: modify the L1 promptgram to improve optimization performance."#, true)

        .with_section("API_contract", r#"You receive:
- Current promptgram (structured with sections: Role, Policy, Heuristics, Style)
- Last N outer episode summaries (each summarizing an inner run)
- Meta-lever estimates (structural_trust, exploration_bias, etc.)
- Strategy capsules from inner runs (the "why" behind changes)

You output JSON with:
- mode: "explore" | "exploit" | "consolidate"
- mode_justification: why this mode
- confidence: 0.0-1.0
- edits: [{section, edit_type, target, content, rationale}]
- expected_effects: what should improve
- hypothesis: what we're testing
- risk_level: "low" | "medium" | "high""#, true)

        .with_section("Policy", r#"### Mode Selection
Choose your mode based on trajectory:
- **explore**: Try something different. Use when plateaued or after success.
- **exploit**: Refine what's working. Use when improving steadily.
- **consolidate**: Lock in gains. Use after significant improvement.

### Edit Constraints
- Only edit mutable sections (Policy, Heuristics, Style)
- Never touch Role, API_contract, or Output_schema
- Small edits (1-2 lines) for exploit mode
- Larger structural changes (new rules, reframing) for explore mode

### Anisotropy
Maintain diversity in your exploration:
- If recent edits were to Policy, try Heuristics next
- If recent changes were conservative, try something bold
- Track what hasn't been tried"#, false)

        .with_section("Heuristics", r#"- Inner runs with high strategy_capsule diversity often find better optima
- Collapse events (NDCG drops > 5%) suggest the prompt is too aggressive
- High oscillation suggests conflicting heuristics in the prompt
- Meta-lever imbalance (all exploration_bias > 0.7) can cause instability
- Prompts that encourage "why" reasoning outperform pure "what" reasoning"#, false)

        .with_section("Output_schema", r#"REASONING:
[Your analysis of the outer trajectory and promptgram performance]

JSON:
{
  "mode": "explore|exploit|consolidate",
  "mode_justification": "...",
  "confidence": 0.7,
  "edits": [
    {
      "section": "Policy|Heuristics|Style",
      "edit_type": "append|replace|delete",
      "target": "text to replace (for replace/delete)",
      "content": "new content",
      "rationale": "why this helps"
    }
  ],
  "expected_effects": ["..."],
  "hypothesis": "what we're testing",
  "risk_level": "low|medium|high"
}"#, true)
}

/// Compute variance of a slice of f64.
fn variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sq_diff_sum: f64 = values.iter().map(|v| (v - mean).powi(2)).sum();
    sq_diff_sum / values.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_config_inner() {
        let outer = RunConfig {
            run_name: "test_outer".to_string(),
            base_dir: PathBuf::from("training-outer/runs"),
            episodes: 50,
            agent: crate::training::reasoning::Agent::Claude,
            model: None,
            save_interval: 5,
        };

        let inner = outer.inner_run_config(1, 20);
        assert_eq!(inner.run_name, "step_001");
        assert!(inner.base_dir.to_str().unwrap().contains("inner_runs"));
    }

    #[test]
    fn test_variance() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let v = variance(&vals);
        assert!((v - 2.0).abs() < 0.001);
    }
}
