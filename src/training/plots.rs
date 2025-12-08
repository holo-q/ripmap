//! Training progress visualization.
//!
//! Two modes:
//! - **Live terminal**: Unicode sparklines during training
//! - **PNG export**: Publication-quality charts via plotters (optional feature)

#[cfg(feature = "plotters")]
use plotters::prelude::*;

use super::reasoning::Scratchpad;

/// Live training progress display for terminal.
/// Shows sparklines and key metrics as training progresses.
pub struct LiveProgress {
    ndcg_history: Vec<f64>,
    failure_history: Vec<usize>,
    confidence_history: Vec<f64>,
    alpha_history: Vec<f64>,
}

impl LiveProgress {
    pub fn new() -> Self {
        Self {
            ndcg_history: Vec::new(),
            failure_history: Vec::new(),
            confidence_history: Vec::new(),
            alpha_history: Vec::new(),
        }
    }

    /// Record metrics for current episode.
    pub fn record(&mut self, ndcg: f64, failures: usize, confidence: f64, alpha: f64) {
        self.ndcg_history.push(ndcg);
        self.failure_history.push(failures);
        self.confidence_history.push(confidence);
        self.alpha_history.push(alpha);
    }

    /// Render sparkline from values.
    fn sparkline(values: &[f64], width: usize) -> String {
        if values.is_empty() {
            return " ".repeat(width);
        }

        let chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = (max - min).max(0.001);

        // Sample or pad to width
        let mut result = String::new();
        for i in 0..width {
            let idx = if values.len() <= width {
                if i < values.len() { Some(i) } else { None }
            } else {
                Some(i * values.len() / width)
            };

            if let Some(idx) = idx {
                let normalized = (values[idx] - min) / range;
                let char_idx = ((normalized * 7.0).round() as usize).min(7);
                result.push(chars[char_idx]);
            } else {
                result.push(' ');
            }
        }
        result
    }

    /// Print current progress to terminal.
    pub fn display(&self, episode: usize, total: usize) {
        let width = 20;

        // Clear line and move cursor
        print!("\x1b[2K\r");

        // Episode counter
        print!("E{:2}/{} ", episode, total);

        // NDCG sparkline
        if !self.ndcg_history.is_empty() {
            let last_ndcg = self.ndcg_history.last().unwrap();
            print!("NDCG[{}]{:.3} ", Self::sparkline(&self.ndcg_history, width), last_ndcg);
        }

        // Failures sparkline (convert to f64)
        if !self.failure_history.is_empty() {
            let fail_f64: Vec<f64> = self.failure_history.iter().map(|&f| f as f64).collect();
            let last_fail = self.failure_history.last().unwrap();
            print!("Fail[{}]{} ", Self::sparkline(&fail_f64, width), last_fail);
        }

        // Convergence indicator
        if self.ndcg_history.len() >= 3 {
            let recent: Vec<_> = self.ndcg_history.iter().rev().take(3).collect();
            let variance: f64 = recent.iter()
                .map(|&&x| (x - recent[0]).powi(2))
                .sum::<f64>() / 3.0;

            if variance < 0.0001 {
                print!("⚡CONVERGED");
            } else if variance < 0.001 {
                print!("~stabilizing");
            }
        }

        // Flush without newline for live update
        use std::io::Write;
        std::io::stdout().flush().ok();
    }

    /// Print final summary with full sparklines.
    pub fn final_summary(&self) {
        use owo_colors::OwoColorize;

        println!("\n");
        println!("{}", " TRAINING COMPLETE ".bold().on_green());
        println!();

        if !self.ndcg_history.is_empty() {
            let first = self.ndcg_history.first().unwrap();
            let last = self.ndcg_history.last().unwrap();
            let delta = last - first;
            let (arrow, delta_str) = if delta > 0.0 {
                ("↑", format!("{:+.4}", delta).green().to_string())
            } else if delta < 0.0 {
                ("↓", format!("{:+.4}", delta).red().to_string())
            } else {
                ("→", format!("{:+.4}", delta).dimmed().to_string())
            };
            println!("  {}: {:.4} {} {:.4}  ({})",
                     "NDCG@10".bold(), first, arrow, last, delta_str);
            println!("          [{}]", Self::sparkline(&self.ndcg_history, 40).cyan());
        }

        if !self.failure_history.is_empty() {
            let first = self.failure_history.first().unwrap();
            let last = self.failure_history.last().unwrap();
            let fail_f64: Vec<f64> = self.failure_history.iter().map(|&f| f as f64).collect();
            let delta = (*last as i32) - (*first as i32);
            let delta_str = if delta < 0 {
                format!("{:+}", delta).green().to_string()
            } else if delta > 0 {
                format!("{:+}", delta).red().to_string()
            } else {
                format!("{:+}", delta).dimmed().to_string()
            };
            println!("  {}: {:3} → {:3}  ({})",
                     "Failures".bold(), first, last, delta_str);
            println!("          [{}]", Self::sparkline(&fail_f64, 40).cyan());
        }

        if !self.alpha_history.is_empty() {
            let first = self.alpha_history.first().unwrap();
            let last = self.alpha_history.last().unwrap();
            println!("  {}: {:.3} → {:.3}", "α".bold(), first, last);
            println!("          [{}]", Self::sparkline(&self.alpha_history, 40).cyan());
        }
        println!();
    }
}

impl Default for LiveProgress {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate training progress charts from scratchpad data.
#[cfg(feature = "plotters")]
pub fn plot_training_progress(scratchpad: &Scratchpad, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (1200, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let episodes: Vec<_> = scratchpad.episodes.iter().enumerate().collect();
    let n = episodes.len();

    if n == 0 {
        return Ok(());
    }

    // Split into 2x3 grid
    let areas = root.split_evenly((3, 2));

    // 1. Failures per episode
    {
        let failures: Vec<_> = episodes.iter()
            .map(|(i, ep)| (*i as f64, ep.failures.len() as f64))
            .collect();

        let max_fail = failures.iter().map(|(_, f)| *f).fold(0.0_f64, f64::max);

        let mut chart = ChartBuilder::on(&areas[0])
            .caption("Ranking Failures per Episode", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0.0..(n as f64), 0.0..max_fail.max(1.0))?;

        chart.configure_mesh().draw()?;

        chart.draw_series(
            failures.iter().map(|(x, y)| {
                Rectangle::new([(*x - 0.3, 0.0), (*x + 0.3, *y)], RED.mix(0.7).filled())
            })
        )?;
    }

    // 2. Confidence over time
    {
        let confidence: Vec<_> = episodes.iter()
            .map(|(i, ep)| (*i as f64, ep.confidence))
            .collect();

        let mut chart = ChartBuilder::on(&areas[1])
            .caption("Claude Confidence", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0.0..(n as f64), 0.0..1.0)?;

        chart.configure_mesh().draw()?;

        chart.draw_series(LineSeries::new(confidence.clone(), &GREEN))?;
        chart.draw_series(confidence.iter().map(|(x, y)| {
            Circle::new((*x, *y), 4, GREEN.filled())
        }))?;
    }

    // 3. PageRank alpha
    {
        let alpha: Vec<_> = episodes.iter()
            .map(|(i, ep)| (*i as f64, ep.params.pagerank_alpha))
            .collect();

        let (min_a, max_a): (f64, f64) = alpha.iter()
            .map(|(_, a)| *a)
            .fold((1.0_f64, 0.0_f64), |(min, max), a| (min.min(a), max.max(a)));

        let mut chart = ChartBuilder::on(&areas[2])
            .caption("PageRank α (damping)", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0.0..(n as f64), (min_a - 0.1)..(max_a + 0.1))?;

        chart.configure_mesh().draw()?;

        // Default line
        chart.draw_series(LineSeries::new(
            vec![(0.0, 0.85), (n as f64, 0.85)],
            &RGBColor(128, 128, 128).mix(0.5),
        ))?.label("Default").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RGBColor(128, 128, 128)));

        chart.draw_series(LineSeries::new(alpha.clone(), &BLUE))?;
        chart.draw_series(alpha.iter().map(|(x, y)| {
            Circle::new((*x, *y), 4, BLUE.filled())
        }))?;

        chart.configure_series_labels().draw()?;
    }

    // 4. Boost parameters
    {
        let temporal: Vec<_> = episodes.iter()
            .map(|(i, ep)| (*i as f64, ep.params.boost_temporal_coupling))
            .collect();
        let focus: Vec<_> = episodes.iter()
            .map(|(i, ep)| (*i as f64, ep.params.boost_focus_expansion))
            .collect();

        let max_boost = temporal.iter().chain(focus.iter())
            .map(|(_, b)| *b)
            .fold(0.0_f64, f64::max);

        let mut chart = ChartBuilder::on(&areas[3])
            .caption("Boost Parameters", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0.0..(n as f64), 0.0..max_boost.max(1.0))?;

        chart.configure_mesh().draw()?;

        chart.draw_series(LineSeries::new(temporal.clone(), &BLUE))?
            .label("Temporal Coupling")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        chart.draw_series(LineSeries::new(focus.clone(), &MAGENTA))?
            .label("Focus Expansion")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &MAGENTA));

        chart.configure_series_labels()
            .position(SeriesLabelPosition::UpperLeft)
            .draw()?;
    }

    // 5. Depth weight deep
    {
        let deep: Vec<_> = episodes.iter()
            .map(|(i, ep)| (*i as f64, ep.params.depth_weight_deep))
            .collect();

        let max_d = deep.iter().map(|(_, d)| *d).fold(0.0_f64, f64::max);

        let mut chart = ChartBuilder::on(&areas[4])
            .caption("Deep File Weight", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0.0..(n as f64), 0.0..max_d.max(0.2))?;

        chart.configure_mesh().draw()?;

        chart.draw_series(LineSeries::new(
            vec![(0.0, 0.1), (n as f64, 0.1)],
            &RGBColor(128, 128, 128).mix(0.5),
        ))?;

        chart.draw_series(LineSeries::new(deep.clone(), &RGBColor(128, 0, 128)))?;
        chart.draw_series(deep.iter().map(|(x, y)| {
            Circle::new((*x, *y), 4, RGBColor(128, 0, 128).filled())
        }))?;
    }

    // 6. NDCG progression (if available)
    {
        let ndcg: Vec<_> = episodes.iter()
            .map(|(i, ep)| (*i as f64, ep.ndcg_before))
            .filter(|(_, n)| *n > 0.0)
            .collect();

        let mut chart = ChartBuilder::on(&areas[5])
            .caption("NDCG@10 (higher = better)", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0.0..(n as f64), 0.8..1.0)?;

        chart.configure_mesh().draw()?;

        if !ndcg.is_empty() {
            chart.draw_series(LineSeries::new(ndcg.clone(), &GREEN))?;
            chart.draw_series(ndcg.iter().map(|(x, y)| {
                Circle::new((*x, *y), 4, GREEN.filled())
            }))?;
        } else {
            // No NDCG data yet - show message
            chart.draw_series(std::iter::once(Text::new(
                "Run with --reason to track NDCG",
                (n as f64 / 2.0, 0.9),
                ("sans-serif", 14).into_font().color(&RGBColor(128, 128, 128)),
            )))?;
        }
    }

    root.present()?;
    println!("Saved training chart to {}", output_path);

    Ok(())
}

/// Stub when plotters feature is disabled.
#[cfg(not(feature = "plotters"))]
pub fn plot_training_progress(_scratchpad: &Scratchpad, _output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Plotting requires --features plotters");
    Ok(())
}
