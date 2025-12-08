use ripmap::ranking::PageRanker;
use ripmap::types::{RankingConfig, Tag, TagKind};
use std::collections::HashMap;
use std::sync::Arc;

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
        metadata: None,
    }
}

fn main() {
    let config = RankingConfig::default();
    let ranker = PageRanker::new(config);

    // Create a simple graph:
    // a.rs defines "foo"
    // b.rs references "foo" (b -> a)
    // c.rs references "foo" (c -> a)
    // a.rs should have highest rank (referenced by both)

    let mut tags_by_file = HashMap::new();
    tags_by_file.insert(
        "/a.rs".to_string(),
        vec![make_tag("a.rs", "foo", TagKind::Def)],
    );
    tags_by_file.insert(
        "/b.rs".to_string(),
        vec![make_tag("b.rs", "foo", TagKind::Ref)],
    );
    tags_by_file.insert(
        "/c.rs".to_string(),
        vec![make_tag("c.rs", "foo", TagKind::Ref)],
    );

    let chat_fnames = vec![];
    let ranks = ranker.compute_ranks(&tags_by_file, &chat_fnames);

    println!("PageRank results:");
    for (file, rank) in &ranks {
        println!("  {}: {:.6}", file, rank);
    }

    let total: f64 = ranks.values().sum();
    println!("\nTotal rank: {:.6}", total);
    println!("Expected: ~1.0");

    // Verify rankings
    if ranks["a.rs"] > ranks["b.rs"] && ranks["a.rs"] > ranks["c.rs"] {
        println!("\n✓ a.rs has highest rank (as expected)");
    } else {
        println!("\n✗ ERROR: a.rs should have highest rank!");
    }

    if (total - 1.0).abs() < 0.01 {
        println!("✓ Total rank sums to 1.0 (as expected)");
    } else {
        println!("✗ ERROR: Total rank should sum to 1.0!");
    }
}
