//! Example demonstrating tag extraction from source files.
//!
//! This shows how to use the extraction pipeline to parse files
//! and extract symbol definitions using regex-based patterns.

use ripmap::extraction::{Parser, extract_tags};
use std::path::Path;

fn main() {
    let parser = Parser::new();
    
    // Example 1: Extract from a Python file
    println!("=== Python Example ===");
    if let Ok(tags) = extract_tags(Path::new("examples/sample.py"), "examples/sample.py", &parser) {
        for tag in &tags {
            println!("  {}:{} - {} ({})", tag.rel_fname, tag.line, tag.name, tag.node_type);
        }
        println!("Total tags: {}\n", tags.len());
    }
    
    // Example 2: Extract from a Rust file
    println!("=== Rust Example (this file) ===");
    if let Ok(tags) = extract_tags(Path::new("examples/extract_tags.rs"), "examples/extract_tags.rs", &parser) {
        for tag in &tags {
            println!("  {}:{} - {} ({})", tag.rel_fname, tag.line, tag.name, tag.node_type);
        }
        println!("Total tags: {}\n", tags.len());
    }
    
    // Example 3: Extract from JavaScript file
    println!("=== JavaScript Example ===");
    if let Ok(tags) = extract_tags(Path::new("examples/sample.js"), "examples/sample.js", &parser) {
        for tag in &tags {
            println!("  {}:{} - {} ({})", tag.rel_fname, tag.line, tag.name, tag.node_type);
        }
        println!("Total tags: {}", tags.len());
    }
}
