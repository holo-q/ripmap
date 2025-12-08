//! ripmap CLI - Ultra-fast codebase cartography
//!
//! This is the command-line entry point for ripmap, the 1000x faster Rust
//! rewrite of grepmap. It orchestrates the full pipeline:
//!
//! 1. File Discovery: Find source files respecting .gitignore
//! 2. Tag Extraction: Parse files with tree-sitter, extract symbols (cached)
//! 3. Graph Building: Build symbol reference graph
//! 4. PageRank: Compute importance scores via iterative power method
//! 5. Contextual Boosts: Apply intent-aware, focus-aware, git-aware multipliers
//! 6. Rendering: Output rich terminal visualization within token budget
//!
//! Design philosophy:
//! - Start simple, expand incrementally
//! - Fail fast with clear error messages
//! - Respect token budgets (LLM context is precious)
//! - Make defaults sane (--color=true, --directory=true)
//! - Verbose mode for debugging the debugger

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

/// Ultra-fast codebase cartography - 1000x faster grepmap
///
/// ripmap discovers the load-bearing structure of your codebase using
/// tree-sitter parsing, PageRank importance ranking, and git-aware boosts.
/// It reveals what matters, not by size but by structural significance.
///
/// Examples:
///   ripmap .                           # Map entire codebase
///   ripmap src/lib.rs                  # Focus on specific file
///   ripmap --focus "auth parser"       # Semantic focus query
///   ripmap --tokens 4096 --git-weight  # Customize output & ranking
#[derive(Parser, Debug)]
#[command(name = "ripmap")]
#[command(version)]
#[command(about, long_about = None)]
pub struct Cli {
    /// Files or directories to focus on
    ///
    /// Can be specific file paths (src/lib.rs) or symbol queries.
    /// If empty, scans the entire project root.
    #[arg(value_name = "FILES")]
    pub files: Vec<String>,

    /// Focus query for symbol matching
    ///
    /// Semantic search across symbol names. Examples:
    ///   --focus "auth"           Match authentication-related symbols
    ///   --focus "parse render"   Match parsing OR rendering symbols
    ///
    /// Focus symbols receive boost multipliers in ranking.
    #[arg(short, long)]
    pub focus: Option<String>,

    /// Additional files to include in analysis
    ///
    /// Use this to add extra context files beyond the main focus.
    /// Useful for including related modules or config files.
    #[arg(long, value_name = "FILES")]
    pub other: Vec<PathBuf>,

    /// Maximum output tokens (LLM context budget)
    ///
    /// Controls how much content to render. Higher = more context but
    /// uses more of your LLM context window. Typical values:
    ///   2048  - Quick overview
    ///   8192  - Standard exploration (default)
    ///   32768 - Deep dive with full signatures
    #[arg(short = 't', long, default_value = "8192")]
    pub tokens: usize,

    /// Force cache refresh
    ///
    /// Ignores cached tag extractions and re-parses all files.
    /// Use this if you suspect cache corruption or after major
    /// refactoring that changes many files.
    #[arg(long)]
    pub refresh: bool,

    /// Enable colored output
    ///
    /// Uses ANSI colors for badges, ranks, and syntax highlighting.
    /// Disable with --no-color for piping to files or LLMs that
    /// don't handle ANSI well.
    #[arg(long, default_value = "true")]
    pub color: bool,

    /// Use directory mode (vs tree mode)
    ///
    /// Directory mode groups symbols by file in a flat list.
    /// Tree mode shows hierarchical file structure.
    /// Directory mode is usually clearer for LLM consumption.
    #[arg(long, default_value = "true")]
    pub directory: bool,

    /// Show statistics
    ///
    /// Prints performance stats at the end:
    ///   - Files scanned
    ///   - Tags extracted
    ///   - Graph size (nodes/edges)
    ///   - PageRank iterations
    ///   - Time breakdown
    #[arg(long)]
    pub stats: bool,

    /// Enable git-based weighting
    ///
    /// Boosts recently changed and high-churn files.
    /// Requires git repository. Adds ~10-50ms overhead.
    /// Recommended for debugging and refactoring tasks.
    #[arg(long)]
    pub git_weight: bool,

    /// Enable diagnostic output
    ///
    /// Shows internal state for debugging ripmap itself:
    ///   - File discovery details
    ///   - Cache hit/miss rates
    ///   - Graph construction logs
    ///   - Ranking computation traces
    ///
    /// Very verbose - use for ripmap development.
    #[arg(long)]
    pub diagnose: bool,

    /// Show call graph relationships
    ///
    /// Displays what each function calls and what calls it.
    /// Uses type hints and imports for resolution when available.
    /// Adds visual indicators like:
    ///   → calls: foo(), bar()
    ///   ← called by: main(), handler()
    #[arg(long)]
    pub calls: bool,

    /// Project root directory
    ///
    /// Base path for file discovery and git operations.
    /// Defaults to current directory.
    #[arg(short, long, default_value = ".")]
    pub root: PathBuf,

    /// Verbose output
    ///
    /// Shows progress messages during execution:
    ///   "Scanning: /path/to/project"
    ///   "Found 1234 files"
    ///   "Extracting tags..."
    ///   "Computing PageRank..."
    ///
    /// Helpful for understanding performance on large codebases.
    #[arg(short, long)]
    pub verbose: bool,

    /// Output full file contents joined with clear separators
    ///
    /// Instead of generating a ranked symbol map, concatenates the full
    /// content of all specified files with clear separators between them.
    /// Useful for creating a codebase artifact when the project is small
    /// enough that full file content fits in context.
    ///
    /// Works with extension filters (-e/--ext) to select file types.
    /// Warns if output exceeds 200KB.
    #[arg(long)]
    pub join: bool,

    /// Filter files by extension (can be repeated)
    ///
    /// Only include files with these extensions. Examples:
    ///   -e rs          # Rust files only
    ///   -e py -e pyi   # Python files and stubs
    ///   --ext ts --ext tsx  # TypeScript
    #[arg(short = 'e', long = "ext", value_name = "EXT")]
    pub extensions: Vec<String>,

    /// Disable colored output
    ///
    /// Equivalent to --color=false. Useful for piping to files or
    /// LLMs that don't handle ANSI escape codes well.
    #[arg(long)]
    pub no_color: bool,
}

/// Size threshold (200KB) above which we warn the user about large output in join mode
const JOIN_SIZE_WARNING_THRESHOLD: usize = 200 * 1024;

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Join mode: concatenate full file contents instead of symbol map
    if cli.join {
        run_join_mode(&cli)?;
        return Ok(());
    }

    // Run the cartography pipeline
    let output = run(&cli)?;

    // Print to stdout (can be piped or redirected)
    println!("{}", output);

    Ok(())
}

/// Execute join mode - output full file contents with clear separators
///
/// This is a fundamentally different output mode from the symbol map. Instead of
/// extracting signatures and ranking, we just concatenate full files. Useful for
/// small-to-medium projects where the full codebase artifact fits in context.
///
/// Features:
/// - Extension filtering via -e/--ext flags
/// - Size warning when output exceeds 200KB
/// - Optional colored output with ANSI codes
/// - Sorted file order for deterministic output
fn run_join_mode(cli: &Cli) -> Result<()> {
    use ripmap::discovery::find_source_files;

    // Determine effective color setting (--no-color overrides --color)
    let use_color = cli.color && !cli.no_color;

    // Canonicalize root path
    let root = cli.root.canonicalize().map_err(|e| {
        anyhow::anyhow!(
            "Failed to resolve root path '{}': {}",
            cli.root.display(),
            e
        )
    })?;

    if cli.verbose {
        eprintln!("🔗 ripmap join mode");
        eprintln!("📂 Root: {}", root.display());
    }

    // Discover files - respect input paths if provided
    let all_files = if cli.files.is_empty() {
        // No paths specified: scan entire root
        if cli.verbose {
            eprintln!("📂 Scanning entire directory");
        }
        find_source_files(&root, false)?
    } else {
        // Paths specified: expand each (files pass through, directories get scanned)
        let mut files = Vec::new();
        for input in &cli.files {
            let path = std::path::Path::new(input);
            let abs_path = if path.is_absolute() {
                path.to_path_buf()
            } else {
                root.join(path)
            };

            if abs_path.is_file() {
                files.push(abs_path);
            } else if abs_path.is_dir() {
                if cli.verbose {
                    eprintln!("📂 Scanning: {}", abs_path.display());
                }
                files.extend(find_source_files(&abs_path, false)?);
            } else {
                eprintln!("⚠️  Skipping non-existent path: {}", input);
            }
        }
        files
    };

    // Filter by extension if specified
    let files: Vec<_> = if cli.extensions.is_empty() {
        all_files
    } else {
        let exts: std::collections::HashSet<_> = cli
            .extensions
            .iter()
            .map(|e| e.strip_prefix('.').unwrap_or(e).to_lowercase())
            .collect();
        all_files
            .into_iter()
            .filter(|f| {
                f.extension()
                    .map(|e| exts.contains(&e.to_string_lossy().to_lowercase()))
                    .unwrap_or(false)
            })
            .collect()
    };

    if files.is_empty() {
        return Err(anyhow::anyhow!(
            "No files to join. Provide paths or directories, or check --ext filters."
        ));
    }

    // Collect all content and compute total size
    let mut segments: Vec<(String, String)> = Vec::new();
    let mut total_size: usize = 0;

    for fpath in files.iter() {
        let content = match std::fs::read_to_string(fpath) {
            Ok(c) => c,
            Err(_) => continue, // Skip unreadable files (binary, etc.)
        };

        // Compute relative path for display
        let rel_path = fpath
            .strip_prefix(&root)
            .unwrap_or(fpath)
            .to_string_lossy()
            .to_string();

        total_size += content.len();
        segments.push((rel_path, content));
    }

    // Sort by path for deterministic output
    segments.sort_by(|a, b| a.0.cmp(&b.0));

    // Warn if output is large
    if total_size > JOIN_SIZE_WARNING_THRESHOLD {
        let size_kb = total_size / 1024;
        eprintln!(
            "⚠️  Warning: Output is {}KB ({} files). Consider using the default map mode for large codebases.",
            size_kb,
            segments.len()
        );
    }

    if cli.verbose {
        eprintln!(
            "✓ Joining {} files ({:.1}KB)",
            segments.len(),
            total_size as f64 / 1024.0
        );
    }

    // Output with separators
    let separator = "─".repeat(80);

    for (rel_path, content) in &segments {
        // Header with file path
        if use_color {
            // Blue separator and inverted white-on-blue path
            println!("\n\x1b[1;34m{}\x1b[0m", separator);
            println!("\x1b[1;37;44m {} \x1b[0m", rel_path);
            println!("\x1b[1;34m{}\x1b[0m\n", separator);
        } else {
            println!("\n{}", separator);
            println!(" {} ", rel_path);
            println!("{}\n", separator);
        }

        // File content (no syntax highlighting for now - keeps deps minimal)
        print!("{}", content);

        // Ensure content ends with newline
        if !content.ends_with('\n') {
            println!();
        }
    }

    // Final separator
    if use_color {
        println!("\n\x1b[1;34m{}\x1b[0m", separator);
    } else {
        println!("\n{}", separator);
    }

    Ok(())
}

/// Execute the full ripmap pipeline
///
/// This orchestrates all stages of the codebase cartography:
/// 1. File Discovery - find source files respecting .gitignore
/// 2. Tag Extraction - parse with regex (tree-sitter later), cached
/// 3. Graph Building - symbol references as edges
/// 4. PageRank - importance scores via power iteration
/// 5. Contextual Boosts - focus, git, intent multipliers
/// 6. Rendering - rich terminal output within token budget
fn run(cli: &Cli) -> Result<String> {
    use ripmap::callgraph::ResolverBuilder;
    use ripmap::config::Config;
    use ripmap::discovery::find_source_files_with_config;
    use ripmap::extraction::{Parser, extract_tags};
    use ripmap::ranking::{BoostCalculator, PageRanker};
    use ripmap::rendering::DirectoryRenderer;
    use ripmap::types::{DetailLevel, RankingConfig};
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;
    use std::time::Instant;

    let start = Instant::now();

    // Canonicalize root path for consistent cache keys
    let root = cli.root.canonicalize().map_err(|e| {
        anyhow::anyhow!(
            "Failed to resolve root path '{}': {}",
            cli.root.display(),
            e
        )
    })?;

    // Load configuration from pyproject.toml or ripmap.toml
    let file_config = Config::load(&root);

    if cli.verbose {
        eprintln!("🗺️  ripmap v{}", env!("CARGO_PKG_VERSION"));
        eprintln!("📂 Scanning: {}", root.display());
        eprintln!("{}", file_config.display_summary());
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Stage 1: File Discovery
    // ══════════════════════════════════════════════════════════════════════════
    let files = find_source_files_with_config(&root, &file_config, false)?;

    if files.is_empty() {
        return Ok("No source files found. Check your path and .gitignore settings.".into());
    }

    if cli.verbose {
        eprintln!("✓ Found {} files ({:.2?})", files.len(), start.elapsed());
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Stage 2: Tag Extraction (regex-based, tree-sitter later)
    // ══════════════════════════════════════════════════════════════════════════
    let extract_start = Instant::now();
    let parser = Parser::new();
    let mut tags_by_file: HashMap<String, Vec<ripmap::Tag>> = HashMap::new();
    let mut total_tags = 0;

    for file in &files {
        let rel_fname = file
            .strip_prefix(&root)
            .unwrap_or(file)
            .to_string_lossy()
            .to_string();

        match extract_tags(file, &rel_fname, &parser) {
            Ok(tags) => {
                total_tags += tags.len();
                if !tags.is_empty() {
                    tags_by_file.insert(rel_fname, tags);
                }
            }
            Err(_) => continue, // Skip files that fail to parse
        }
    }

    if cli.verbose {
        eprintln!(
            "✓ Extracted {} tags from {} files ({:.2?})",
            total_tags,
            tags_by_file.len(),
            extract_start.elapsed()
        );
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Stage 3 & 4: PageRank on Symbol Graph
    // ══════════════════════════════════════════════════════════════════════════
    let rank_start = Instant::now();

    // RankingConfig defaults are the baseline. Future: meta-lever inference
    // will derive these from context signals (query, git state, repo shape).
    let config = RankingConfig::default();

    let page_ranker = PageRanker::new(config.clone());

    // Determine chat files (focus files get boost)
    let chat_fnames: Vec<String> = cli
        .files
        .iter()
        .filter_map(|f| {
            let path = std::path::Path::new(f);
            if path.exists() {
                path.strip_prefix(&root)
                    .ok()
                    .or(Some(path))
                    .map(|p| p.to_string_lossy().to_string())
            } else {
                None
            }
        })
        .collect();

    let file_ranks = page_ranker.compute_ranks(&tags_by_file, &chat_fnames);

    if cli.verbose {
        eprintln!(
            "✓ Computed PageRank for {} files ({:.2?})",
            file_ranks.len(),
            rank_start.elapsed()
        );
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Stage 4.5: Build Call Graph
    // ══════════════════════════════════════════════════════════════════════════
    // Call graph is built unconditionally - it powers focus expansion and
    // caller weight boosts, not just rendering. The --calls flag only controls
    // whether call relationships are displayed in output.
    let cg_start = Instant::now();

    // Flatten all tags for call graph construction
    let all_tags: Vec<ripmap::Tag> = tags_by_file
        .values()
        .flat_map(|tags| tags.iter().cloned())
        .collect();

    // Build resolver with all strategies enabled
    let resolver = ResolverBuilder::new()
        .same_file(true)
        .type_hints(true)
        .imports(true)
        .name_match(true)
        .build();

    let call_graph = resolver.build_graph(&all_tags);

    if cli.verbose {
        let stats = resolver.stats(&all_tags);
        let resolved = stats.total_calls - stats.unresolved;
        eprintln!(
            "✓ Built call graph: {} functions, {} calls ({:.2?})",
            call_graph.function_count(),
            call_graph.call_count(),
            cg_start.elapsed()
        );
        eprintln!(
            "  Resolution: {:.1}% success ({} resolved, {} unresolved)",
            stats.resolution_rate() * 100.0,
            resolved,
            stats.unresolved
        );
    }

    // Compute function-level PageRank on the call graph
    // This gives more precise importance scores than file-level PageRank
    let function_ranks = page_ranker.compute_function_ranks(&call_graph);

    // Convert FunctionId -> (file, name) tuple for symbol_ranks
    let symbol_ranks: HashMap<(Arc<str>, Arc<str>), f64> = function_ranks
        .into_iter()
        .map(|(func_id, rank)| ((func_id.file, func_id.name), rank))
        .collect();

    if cli.verbose && !symbol_ranks.is_empty() {
        eprintln!(
            "✓ Computed function-level PageRank for {} symbols",
            symbol_ranks.len()
        );
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Stage 4.6: Test↔Source Coupling Detection
    // ══════════════════════════════════════════════════════════════════════════
    // Codex optimization identified "path-aware test↔crate coupling edges" as
    // a missing architectural feature. Detect test files and link them to their
    // corresponding source files, adding synthetic edges for graph expansion.
    use ripmap::ranking::{FocusResolver, TestCouplingDetector};

    let coupling_detector =
        TestCouplingDetector::new().with_min_confidence(config.test_coupling_min_confidence);

    // Detect test↔source couplings from file list
    let file_paths: Vec<_> = files
        .iter()
        .map(|f| f.strip_prefix(&root).unwrap_or(f).to_path_buf())
        .collect();

    let test_couplings = coupling_detector.detect_couplings(&file_paths);
    let test_coupling_edges = coupling_detector.as_symbol_edges(&test_couplings);

    if cli.verbose && !test_couplings.is_empty() {
        eprintln!("✓ Detected {} test↔source couplings", test_couplings.len());
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Stage 4.7: Focus Expansion via Call Graph
    // ══════════════════════════════════════════════════════════════════════════
    // When user provides --focus, we BFS through call relationships to find
    // related functions. This surfaces the call chain around focused symbols.

    let focus_expansion_weights = if cli.focus.is_some() {
        let focus_start = Instant::now();
        let focus_resolver = FocusResolver::new(&root);

        // Parse focus targets from query
        let focus_targets: Vec<String> = cli
            .focus
            .as_ref()
            .map(|f| f.split_whitespace().map(String::from).collect())
            .unwrap_or_default();

        // Resolve focus targets to matched symbols
        let (_matched_files, matched_idents) =
            focus_resolver.resolve(&focus_targets, &tags_by_file);

        // Convert matched idents to the format expand_via_graph expects
        let matched_set: HashSet<String> = matched_idents;

        // Get call graph edges for BFS expansion
        // Combine call graph edges with test↔source coupling edges
        let mut symbol_edges = call_graph.as_symbol_edges();
        symbol_edges.extend(test_coupling_edges.clone());

        // Expand through call relationships: callers and callees of focused functions
        // Uses configurable max_hops and decay from RankingConfig
        let expanded = focus_resolver.expand_via_graph(
            &matched_set,
            &symbol_edges,
            config.focus_expansion_max_hops,
            config.focus_expansion_decay,
        );

        if cli.verbose && !expanded.is_empty() {
            eprintln!(
                "✓ Focus expansion: {} symbols via call graph ({:.2?})",
                expanded.len(),
                focus_start.elapsed()
            );
        }

        Some(expanded)
    } else {
        None
    };

    // ══════════════════════════════════════════════════════════════════════════
    // Stage 5: Apply Contextual Boosts
    // ══════════════════════════════════════════════════════════════════════════
    let boost_start = Instant::now();
    let boost_calculator = BoostCalculator::new(config);

    let chat_fnames_set: HashSet<String> = chat_fnames.into_iter().collect();
    let mentioned_fnames: HashSet<String> = cli.files.iter().cloned().collect();

    // Extract identifiers from focus query
    let mentioned_idents: HashSet<String> = cli
        .focus
        .as_ref()
        .map(|f| f.split_whitespace().map(String::from).collect())
        .unwrap_or_default();

    // Get caller weights from call graph - functions called by many others
    // are likely important API surfaces.
    // Aggregate symbol-level caller counts to file-level weights.
    let symbol_caller_weights = call_graph.caller_weights();
    let mut caller_weights: HashMap<String, f64> = HashMap::new();
    for ((file, _symbol), count) in &symbol_caller_weights {
        *caller_weights.entry(file.to_string()).or_insert(0.0) += *count as f64;
    }
    // Normalize by applying log scale (many callers = high importance, but diminishing returns)
    for weight in caller_weights.values_mut() {
        *weight = 1.0 + (*weight).ln().max(0.0);
    }

    let ranked_tags = boost_calculator.apply_boosts(
        &tags_by_file,
        &file_ranks,
        if symbol_ranks.is_empty() {
            None
        } else {
            Some(&symbol_ranks)
        },
        &chat_fnames_set,
        &mentioned_fnames,
        &mentioned_idents,
        &HashSet::new(), // temporal_boost_files - TODO with git
        None,            // git_weights - TODO
        if caller_weights.is_empty() {
            None
        } else {
            Some(&caller_weights)
        },
        focus_expansion_weights.as_ref(),
    );

    if cli.verbose {
        eprintln!(
            "✓ Applied boosts to {} tags ({:.2?})",
            ranked_tags.len(),
            boost_start.elapsed()
        );
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Stage 6: Rendering
    // ══════════════════════════════════════════════════════════════════════════
    let render_start = Instant::now();

    // Simple token counter (4 chars ≈ 1 token)
    let token_counter = |s: &str| s.len() / 4;
    let renderer = DirectoryRenderer::new(Box::new(token_counter));

    // Choose detail level based on token budget
    let detail = if cli.tokens >= 16384 {
        DetailLevel::High
    } else if cli.tokens >= 4096 {
        DetailLevel::Medium
    } else {
        DetailLevel::Low
    };

    // Render output (with call graph visualization if --calls enabled)
    let output = renderer.render_with_calls(
        &ranked_tags,
        detail,
        &HashMap::new(), // badges - TODO with git
        &HashMap::new(), // temporal_mates - TODO
        if cli.calls { Some(&call_graph) } else { None },
    );

    if cli.verbose {
        eprintln!("✓ Rendered output ({:.2?})", render_start.elapsed());
        eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        eprintln!("Total time: {:.2?}", start.elapsed());
    }

    // Prepend header
    let header = format!(
        "# Ranking: {} | {} symbols | ~{} tokens\n",
        if ranked_tags.len() > 100 {
            "high (dense)"
        } else {
            "low (sparse)"
        },
        ranked_tags.len(),
        token_counter(&output)
    );

    if cli.stats {
        let stats = format!(
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\
             ## Statistics\n\
             Files discovered: {}\n\
             Files with tags: {}\n\
             Tags extracted: {}\n\
             Ranked symbols: {}\n\
             Total time: {:.2?}\n",
            files.len(),
            tags_by_file.len(),
            total_tags,
            ranked_tags.len(),
            start.elapsed()
        );
        Ok(format!("{}{}{}", header, output, stats))
    } else {
        Ok(format!("{}{}", header, output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parse_minimal() {
        // Test minimal invocation
        let cli = Cli::parse_from(&["ripmap"]);
        assert_eq!(cli.files.len(), 0);
        assert_eq!(cli.tokens, 8192);
        assert!(cli.directory);
        assert!(cli.color);
    }

    #[test]
    fn test_cli_parse_with_files() {
        let cli = Cli::parse_from(&["ripmap", "src/lib.rs", "src/main.rs"]);
        assert_eq!(cli.files, vec!["src/lib.rs", "src/main.rs"]);
    }

    #[test]
    fn test_cli_parse_focus() {
        let cli = Cli::parse_from(&["ripmap", "--focus", "auth parser"]);
        assert_eq!(cli.focus, Some("auth parser".into()));
    }

    #[test]
    fn test_cli_parse_tokens() {
        let cli = Cli::parse_from(&["ripmap", "--tokens", "4096"]);
        assert_eq!(cli.tokens, 4096);
    }

    #[test]
    fn test_cli_parse_flags() {
        let cli = Cli::parse_from(&[
            "ripmap",
            "--refresh",
            "--stats",
            "--git-weight",
            "--verbose",
            "--diagnose",
        ]);
        assert!(cli.refresh);
        assert!(cli.stats);
        assert!(cli.git_weight);
        assert!(cli.verbose);
        assert!(cli.diagnose);
    }

    #[test]
    fn test_cli_parse_root() {
        let cli = Cli::parse_from(&["ripmap", "--root", "/tmp/test"]);
        assert_eq!(cli.root, PathBuf::from("/tmp/test"));
    }

    #[test]
    fn test_run_on_ripmap_itself() -> Result<()> {
        // Test running on the ripmap codebase itself
        let cli = Cli {
            files: vec![],
            focus: None,
            other: vec![],
            tokens: 8192,
            refresh: false,
            color: true,
            directory: true,
            stats: false,
            git_weight: false,
            diagnose: false,
            calls: false,
            root: PathBuf::from("."),
            verbose: false,
            join: false,
            extensions: vec![],
            no_color: false,
        };

        let output = run(&cli)?;

        // Output should contain ranking header and symbols
        assert!(output.contains("# Ranking:"), "Missing ranking header");
        assert!(output.contains("symbols"), "Missing symbols count");
        assert!(output.contains("tokens"), "Missing tokens estimate");

        Ok(())
    }

    #[test]
    fn test_cli_parse_join_mode() {
        let cli = Cli::parse_from(&["ripmap", "--join", "-e", "rs"]);
        assert!(cli.join);
        assert_eq!(cli.extensions, vec!["rs"]);
    }

    #[test]
    fn test_cli_parse_multiple_extensions() {
        let cli = Cli::parse_from(&["ripmap", "-e", "py", "-e", "pyi", "--ext", "rs"]);
        assert_eq!(cli.extensions, vec!["py", "pyi", "rs"]);
    }

    #[test]
    fn test_cli_parse_no_color() {
        let cli = Cli::parse_from(&["ripmap", "--no-color"]);
        assert!(cli.no_color);
    }
}
