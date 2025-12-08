//! Benchmark the extraction performance.

use ripmap::extraction::{Parser, extract_tags};
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let parser = Parser::new();

    // Collect all Rust files in src/
    let files: Vec<PathBuf> = walkdir::WalkDir::new("src")
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("rs"))
        .map(|e| e.path().to_path_buf())
        .collect();

    println!("Found {} Rust files in src/", files.len());
    println!("Starting extraction training...\n");

    let start = Instant::now();
    let mut total_tags = 0;

    for path in &files {
        if let Ok(tags) = extract_tags(path, path.to_str().unwrap(), &parser) {
            total_tags += tags.len();
        }
    }

    let elapsed = start.elapsed();

    println!("Results:");
    println!("  Files processed: {}", files.len());
    println!("  Total tags extracted: {}", total_tags);
    println!("  Time elapsed: {:.2?}", elapsed);
    println!("  Avg per file: {:.2?}", elapsed / files.len() as u32);
    println!(
        "  Tags per second: {:.0}",
        total_tags as f64 / elapsed.as_secs_f64()
    );
}
