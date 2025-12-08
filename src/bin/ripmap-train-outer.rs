//! ripmap-train-outer: L2 Promptgram Optimizer
//!
//! The outer loop that evolves prompts for the inner reasoning-based optimizer.
//!
//! ## Usage
//!
//! ```bash
//! # Run Stage 0: baseline outer loop (no prompt edits, just recording)
//! ripmap-train-outer --outer-steps 10 --inner-episodes 20 --run-name test_outer
//!
//! # Run with a specific inner promptgram
//! ripmap-train-outer --promptgram training-outer/prompts/inner/explorer.toml
//!
//! # Full outer loop with prompt editing (Stage 1+)
//! ripmap-train-outer --outer-steps 50 --inner-episodes 30 --edit-prompts
//! ```
//!
//! ## Directory Structure
//!
//! ```text
//! training-outer/runs/<run-name>/
//!   config.toml           # Outer run configuration
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
    /// Number of outer steps to run
    #[arg(long, default_value = "10")]
    outer_steps: usize,

    /// Number of inner episodes per outer step
    #[arg(long, default_value = "20")]
    inner_episodes: usize,

    /// Name for this outer run
    #[arg(long, default_value = "outer_run")]
    run_name: String,

    /// Agent to use for inner loop (claude, gemini, codex)
    #[arg(long, default_value = "claude")]
    inner_agent: String,

    /// Agent to use for outer loop reasoning
    #[arg(long, default_value = "codex")]
    outer_agent: String,

    /// Path to inner promptgram (optional, uses default if not specified)
    #[arg(long)]
    promptgram: Option<PathBuf>,

    /// Enable prompt editing (Stage 1+). Without this, runs Stage 0 (recording only).
    #[arg(long)]
    edit_prompts: bool,

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
    let inner_agent: Agent = args.inner_agent.parse()
        .map_err(|e: String| e)?;
    let _outer_agent: Agent = args.outer_agent.parse()
        .map_err(|e: String| e)?;

    // Set up run configuration
    let run_config = RunConfig {
        run_name: args.run_name.clone(),
        base_dir: PathBuf::from("training-outer/runs"),
        episodes: args.inner_episodes,
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
    let inner_promptgram = if let Some(path) = &args.promptgram {
        println!("📄 Loading promptgram from {:?}", path);
        Promptgram::load(path)?
    } else {
        println!("📄 Using default inner promptgram");
        baseline_promptgram()
    };

    // Set up outer loop
    let mut outer_loop = OuterLoop::new();
    outer_loop.population = vec![inner_promptgram];
    outer_loop.inner_config = OuterConfig {
        inner_episodes: args.inner_episodes,
        inner_agent: args.inner_agent.clone(),
        outer_agent: args.outer_agent.clone(),
        max_outer_steps: args.outer_steps,
        exploration_quota: 0.2,
        corpus: args.corpus.clone(),
        edit_prompts: args.edit_prompts,
        ..Default::default()
    };

    // Print configuration
    println!("Configuration:");
    println!("  Run name:       {}", args.run_name);
    println!("  Outer steps:    {}", args.outer_steps);
    println!("  Inner episodes: {}", args.inner_episodes);
    println!("  Inner agent:    {}", args.inner_agent);
    println!("  Outer agent:    {}", args.outer_agent);
    println!("  Corpus:         {}", args.corpus);
    println!("  Edit prompts:   {}", args.edit_prompts);
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
outer_steps = {}
inner_episodes = {}
inner_agent = "{}"
outer_agent = "{}"
corpus = "{}"
edit_prompts = {}
"#,
        args.run_name,
        args.outer_steps,
        args.inner_episodes,
        args.inner_agent,
        args.outer_agent,
        args.corpus,
        args.edit_prompts,
    );
    std::fs::write(&config_path, config_content)?;

    // Run outer loop
    println!("{}", "─".repeat(65));
    println!("Starting outer loop ({} steps)...\n", args.outer_steps - start_step);

    for step in start_step..args.outer_steps {
        println!("{}", format!(" OUTER STEP {}/{} ", step + 1, args.outer_steps).bold().on_cyan());

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
