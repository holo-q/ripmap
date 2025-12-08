//! Demonstration of the boost calculator in action.
//!
//! This example shows how contextual boosts dramatically affect symbol ranking.
//! We'll create a simple scenario where multiple signals align to boost a symbol.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use ripmap::ranking::BoostCalculator;
use ripmap::types::{RankingConfig, Tag, TagKind};

fn main() {
    println!("🚀 Boost Calculator Demo\n");

    // Setup configuration with default boost multipliers
    let config = RankingConfig::default();
    let calculator = BoostCalculator::new(config.clone());

    // Create some sample tags
    let mut tags_by_file = HashMap::new();

    // File: src/auth.rs
    tags_by_file.insert(
        "/src/auth.rs".to_string(),
        vec![
            make_tag("src/auth.rs", "authenticate", TagKind::Def),
            make_tag("src/auth.rs", "validate_token", TagKind::Def),
        ],
    );

    // File: src/main.rs
    tags_by_file.insert(
        "/src/main.rs".to_string(),
        vec![
            make_tag("src/main.rs", "main", TagKind::Def),
            make_tag("src/main.rs", "handle_request", TagKind::Def),
        ],
    );

    // Base PageRank scores (from graph structure)
    let mut file_ranks = HashMap::new();
    file_ranks.insert("src/auth.rs".to_string(), 0.3);
    file_ranks.insert("src/main.rs".to_string(), 0.5);

    let mut symbol_ranks = HashMap::new();
    symbol_ranks.insert((Arc::from("src/auth.rs"), Arc::from("authenticate")), 0.4);
    symbol_ranks.insert((Arc::from("src/auth.rs"), Arc::from("validate_token")), 0.2);
    symbol_ranks.insert((Arc::from("src/main.rs"), Arc::from("main")), 0.6);
    symbol_ranks.insert((Arc::from("src/main.rs"), Arc::from("handle_request")), 0.5);

    println!("📊 Base PageRank Scores:");
    println!("  authenticate:     0.4");
    println!("  validate_token:   0.2");
    println!("  main:             0.6");
    println!("  handle_request:   0.5\n");

    // Scenario 1: No contextual signals
    println!("🔵 Scenario 1: No Contextual Signals");
    println!("   (Just base PageRank)\n");

    let result1 = calculator.apply_boosts(
        &tags_by_file,
        &file_ranks,
        Some(&symbol_ranks),
        &HashSet::new(),
        &HashSet::new(),
        &HashSet::new(),
        &HashSet::new(),
        None,
        None,
        None,
    );

    print_results(&result1);

    // Scenario 2: User mentions "authenticate" in query
    println!("\n🟢 Scenario 2: User Asks About 'authenticate'");
    println!("   Boost: 10x (mentioned_ident)\n");

    let mut mentioned_idents = HashSet::new();
    mentioned_idents.insert("authenticate".to_string());

    let result2 = calculator.apply_boosts(
        &tags_by_file,
        &file_ranks,
        Some(&symbol_ranks),
        &HashSet::new(),
        &HashSet::new(),
        &mentioned_idents,
        &HashSet::new(),
        None,
        None,
        None,
    );

    print_results(&result2);

    // Scenario 3: Editing auth.rs, mentions "authenticate"
    println!("\n🟡 Scenario 3: Editing auth.rs + Mentions 'authenticate'");
    println!("   Boosts: 10x (mentioned_ident) × 20x (chat_file) = 200x!\n");

    let mut chat_fnames = HashSet::new();
    chat_fnames.insert("/src/auth.rs".to_string());

    let result3 = calculator.apply_boosts(
        &tags_by_file,
        &file_ranks,
        Some(&symbol_ranks),
        &chat_fnames,
        &HashSet::new(),
        &mentioned_idents,
        &HashSet::new(),
        None,
        None,
        None,
    );

    print_results(&result3);

    // Scenario 4: All signals align!
    println!("\n🔴 Scenario 4: ALL SIGNALS ALIGN!");
    println!("   - Editing auth.rs (20x chat_file)");
    println!("   - Mentions 'authenticate' (10x mentioned_ident)");
    println!("   - Mentions auth.rs (5x mentioned_file)");
    println!("   - auth.rs co-changes with chat (3x temporal_coupling)");
    println!("   - Recent git activity (2x git_weight)");
    println!("   - Caller analysis boost (1.5x caller_weight)");
    println!("   Total: 20 × 10 × 5 × 3 × 2 × 1.5 = 9000x BASE RANK!\n");

    let mut mentioned_fnames = HashSet::new();
    mentioned_fnames.insert("src/auth.rs".to_string());

    let mut temporal_boost_files = HashSet::new();
    temporal_boost_files.insert("src/auth.rs".to_string());

    let mut git_weights = HashMap::new();
    git_weights.insert("src/auth.rs".to_string(), 2.0);

    let mut caller_weights = HashMap::new();
    caller_weights.insert("src/auth.rs".to_string(), 1.5);

    let result4 = calculator.apply_boosts(
        &tags_by_file,
        &file_ranks,
        Some(&symbol_ranks),
        &chat_fnames,
        &mentioned_fnames,
        &mentioned_idents,
        &temporal_boost_files,
        Some(&git_weights),
        Some(&caller_weights),
        None,
    );

    print_results(&result4);

    println!("\n✨ Key Insight:");
    println!("   When multiple contextual signals align, the boost system creates");
    println!("   EXPLOSIVE amplification, ensuring the most relevant symbol rises");
    println!("   to the top regardless of its structural PageRank.");
    println!("\n   This is the magic of multiplicative boosts! 🎯");
}

/// Helper to create a test tag
fn make_tag(rel_fname: &str, name: &str, kind: TagKind) -> Tag {
    Tag {
        rel_fname: Arc::from(rel_fname),
        fname: Arc::from(format!("/{}", rel_fname)),
        line: 1,
        name: Arc::from(name),
        kind,
        node_type: Arc::from("function"),
        parent_name: None,
        parent_line: None,
        signature: None,
        fields: None,
    }
}

/// Pretty-print ranked results
fn print_results(results: &[ripmap::types::RankedTag]) {
    println!("   Ranking:");
    for (i, ranked) in results.iter().enumerate() {
        println!(
            "   {}. {:20} (rank: {:.2})",
            i + 1,
            ranked.tag.name.as_ref(),
            ranked.rank
        );
    }
}
