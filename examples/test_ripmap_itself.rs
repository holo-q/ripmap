//! Test extraction on the ripmap codebase itself.

use ripmap::extraction::{Parser, extract_tags};
use std::path::Path;

fn main() {
    let parser = Parser::new();
    
    println!("=== Extracting from ripmap's own types.rs ===\n");
    
    if let Ok(tags) = extract_tags(
        Path::new("src/types.rs"),
        "src/types.rs",
        &parser
    ) {
        // Group by node type
        let mut by_type: std::collections::HashMap<&str, Vec<_>> = std::collections::HashMap::new();
        for tag in &tags {
            by_type.entry(tag.node_type.as_ref()).or_default().push(tag);
        }
        
        // Display by category
        for (node_type, tags) in by_type.iter() {
            println!("{} (count: {}):", node_type, tags.len());
            for tag in tags.iter().take(5) {
                println!("  - {} (line {})", tag.name, tag.line);
            }
            if tags.len() > 5 {
                println!("  ... and {} more", tags.len() - 5);
            }
            println!();
        }
        
        println!("Total symbols extracted: {}", tags.len());
    }
}
