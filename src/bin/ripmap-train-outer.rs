//! ripmap-train-outer: L2 Promptgram Optimizer
//!
//! The outer loop that evolves prompts for the inner reasoning-based optimizer.
//! Each run is self-contained - output promptgram lives in the run directory.
//!
//! ## Usage
//!
//! ```bash
//! # Full outer loop with L2 prompt evolution (default)
//! ripmap-train-outer my_experiment --steps-outer 50 --episodes-inner 30
//!
//! # Dry run: just record metrics, no L2 reasoning/prompt editing
//! ripmap-train-outer baseline_run --dry
//!
//! # Seed from a previous run's promptgram
//! ripmap-train-outer new_experiment --promptgram training-outer/runs/old_run/promptgram.toml
//! ```
//!
//! ## Directory Structure
//!
//! ```text
//! training-outer/runs/<run-name>/
//!   config.toml           # Outer run configuration
//!   promptgram.toml       # Current/final promptgram (self-contained)
//!   outer_scratchpad.json # L2 scratchpad
//!   inner_runs/
//!     step_001/           # Inner run for outer step 1
//!       scratchpad.json
//!       results.json
//!       progress.png
//!     step_002/
//!       ...
//! ```

use std::path::PathBuf;
use clap::Parser;
use ripmap::training::reasoning::Agent;
use ripmap::training_outer::{
    OuterLoop, OuterScratchpad, OuterConfig, RunConfig, RunContext,
    Promptgram, baseline_promptgram,
};

#[derive(Parser)]
#[command(name = "ripmap-train-outer")]
#[command(about = "L2 Promptgram Optimizer - evolves prompts for the inner optimizer")]
struct Args {
    /// Name for this outer run (creates training-outer/runs/<name>/)
    run_name: String,

    /// Number of outer steps to run
    #[arg(long, default_value = "10")]
    steps_outer: usize,

    /// Number of inner episodes per outer step
    #[arg(long, default_value = "20")]
    episodes_inner: usize,

    /// Agent to use for inner loop (claude, gemini, codex)
    #[arg(long, default_value = "claude")]
    agent_inner: String,

    /// Agent to use for outer loop L2 reasoning
    #[arg(long, default_value = "codex")]
    agent_outer: String,

    /// Seed promptgram (path or run name). Uses default if not specified.
    #[arg(long)]
    promptgram: Option<String>,

    /// Dry run: record metrics only, no L2 reasoning or prompt editing
    #[arg(long)]
    dry: bool,

    /// Resume from existing outer scratchpad
    #[arg(long)]
    resume: bool,

    /// Corpus to use: quick, curated, or full
    #[arg(long, default_value = "curated")]
    corpus: String,

    /// Save interval for checkpoints
    #[arg(long, default_value = "1")]
    save_interval: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use owo_colors::OwoColorize;

    let args = Args::parse();

    println!();
    println!("{}", " RIPMAP L2 OUTER LOOP OPTIMIZER ".bold().on_magenta());
    println!();

    // Parse agents
    let inner_agent: Agent = args.agent_inner.parse()
        .map_err(|e: String| e)?;
    let _outer_agent: Agent = args.agent_outer.parse()
        .map_err(|e: String| e)?;

    // Set up run configuration
    let run_config = RunConfig {
        run_name: args.run_name.clone(),
        base_dir: PathBuf::from("training-outer/runs"),
        episodes: args.episodes_inner,
        agent: inner_agent,
        model: None,
        save_interval: args.save_interval,
    };

    // Create run context
    let ctx = RunContext {
        config: run_config.clone(),
        episode: 0,
        is_outer: true,
        parent_step: None,
    };

    // Set up directories
    ctx.setup_dirs()?;

    // Load or create outer scratchpad
    let scratchpad_path = ctx.config.output_dir().join("outer_scratchpad.json");
    let mut outer_scratchpad = if args.resume && scratchpad_path.exists() {
        println!("📂 Resuming from existing scratchpad...");
        let content = std::fs::read_to_string(&scratchpad_path)?;
        serde_json::from_str(&content)?
    } else {
        OuterScratchpad::default()
    };

    // Load or create inner promptgram
    // --promptgram can be a path or a run name (resolves to training-outer/runs/<name>/promptgram.toml)
    let inner_promptgram = if let Some(ref source) = args.promptgram {
        let path = if source.contains('/') || source.ends_with(".toml") {
            PathBuf::from(source)
        } else {
            // Treat as run name
            PathBuf::from(format!("training-outer/runs/{}/promptgram.toml", source))
        };
        println!("📄 Loading promptgram from {:?}", path);
        Promptgram::load(&path)?
    } else {
        println!("📄 Using default inner promptgram");
        baseline_promptgram()
    };

    // Set up outer loop
    // --dry inverts the default (edit_prompts=true means L2 reasoning is active)
    let edit_prompts = !args.dry;
    let mut outer_loop = OuterLoop::new();
    outer_loop.population = vec![inner_promptgram];
    outer_loop.inner_config = OuterConfig {
        inner_episodes: args.episodes_inner,
        inner_agent: args.agent_inner.clone(),
        outer_agent: args.agent_outer.clone(),
        max_outer_steps: args.steps_outer,
        exploration_quota: 0.2,
        corpus: args.corpus.clone(),
        edit_prompts,
        ..Default::default()
    };

    // Print configuration
    println!("Configuration:");
    println!("  Run name:       {}", args.run_name);
    println!("  Outer steps:    {}", args.steps_outer);
    println!("  Inner episodes: {}", args.episodes_inner);
    println!("  Inner agent:    {}", args.agent_inner);
    println!("  Outer agent:    {}", args.agent_outer);
    println!("  Corpus:         {}", args.corpus);
    println!("  L2 reasoning:   {}", if edit_prompts { "enabled" } else { "dry run" });
    println!("  Output dir:     {:?}", ctx.config.output_dir());
    println!();

    // Determine starting step
    let start_step = outer_scratchpad.episodes.len();
    if start_step > 0 {
        println!("📊 Resuming from step {} (best NDCG: {:.4})",
            start_step, outer_scratchpad.best_ndcg);
    }

    // Save initial config
    let config_path = ctx.config.output_dir().join("config.toml");
    let config_content = format!(r#"# L2 Outer Loop Configuration
[run]
name = "{}"
steps_outer = {}
episodes_inner = {}
agent_inner = "{}"
agent_outer = "{}"
corpus = "{}"
dry = {}
"#,
        args.run_name,
        args.steps_outer,
        args.episodes_inner,
        args.agent_inner,
        args.agent_outer,
        args.corpus,
        args.dry,
    );
    std::fs::write(&config_path, config_content)?;

    // Run outer loop
    println!("{}", "─".repeat(65));
    println!("Starting outer loop ({} steps)...\n", args.steps_outer - start_step);

    for step in start_step..args.steps_outer {
        println!("{}", format!(" OUTER STEP {}/{} ", step + 1, args.steps_outer).bold().on_cyan());

        // Run one outer episode
        match outer_loop.run_outer_episode(step + 1, &ctx, &mut outer_scratchpad) {
            Ok(summary) => {
                // Print summary
                println!("\n  📈 Results:");
                println!("     NDCG: {:.4} → {:.4} (Δ{:+.4})",
                    summary.baseline_metrics.ndcg,
                    summary.final_metrics.ndcg,
                    summary.delta.ndcg);
                println!("     Failures: {} → {}",
                    summary.baseline_metrics.failures,
                    summary.final_metrics.failures);
                println!("     Duration: {:.1}s", summary.duration_secs);
                println!("     Stability: {} collapses, {:.4} variance",
                    summary.stability.collapse_events,
                    summary.stability.ndcg_variance);

                // Print meta-levers
                println!("     Meta-levers: {}", summary.meta_levers_estimate.summary());

                // Print strategy capsules (if any)
                if !summary.strategy_capsules.is_empty() {
                    println!("     Strategy capsules:");
                    for capsule in summary.strategy_capsules.iter().take(3) {
                        println!("       • {}", capsule.chars().take(60).collect::<String>());
                    }
                }

                // Update best if improved
                if summary.final_metrics.ndcg > outer_scratchpad.best_ndcg {
                    println!("\n  🏆 NEW BEST! {:.4} → {:.4}",
                        outer_scratchpad.best_ndcg, summary.final_metrics.ndcg);
                }
            }
            Err(e) => {
                eprintln!("\n  ❌ Error in outer step: {}", e);
                // Continue to next step
            }
        }

        // Save scratchpad after each step
        let scratchpad_json = serde_json::to_string_pretty(&outer_scratchpad)?;
        std::fs::write(&scratchpad_path, &scratchpad_json)?;
        println!("\n  💾 Checkpoint saved\n");
    }

    // Print final summary
    println!();
    println!("{}", "─".repeat(65));
    println!("{}", " OUTER LOOP COMPLETE ".bold().on_green());
    println!("{}", "─".repeat(65));
    println!();

    println!("Total outer steps: {}", outer_scratchpad.episodes.len());
    println!("Best NDCG: {:.4} (prompt: {})",
        outer_scratchpad.best_ndcg, outer_scratchpad.best_prompt_id);

    if !outer_scratchpad.episodes.is_empty() {
        let first = outer_scratchpad.episodes.first().unwrap();
        let last = outer_scratchpad.episodes.last().unwrap();
        println!("NDCG trajectory: {:.4} → {:.4}",
            first.baseline_metrics.ndcg, last.final_metrics.ndcg);

        // Print strategy capsule diversity
        let all_capsules: Vec<_> = outer_scratchpad.episodes.iter()
            .flat_map(|e| e.strategy_capsules.iter())
            .collect();
        println!("Total strategy capsules: {}", all_capsules.len());

        // Print structural insights
        let all_insights: std::collections::HashSet<_> = outer_scratchpad.episodes.iter()
            .flat_map(|e| e.structural_insights.iter())
            .collect();
        if !all_insights.is_empty() {
            println!("\nStructural insights discovered:");
            for insight in all_insights.iter().take(5) {
                println!("  • {}", insight);
            }
        }
    }

    println!("\nOutputs saved to: {:?}", ctx.config.output_dir());

    Ok(())
}
