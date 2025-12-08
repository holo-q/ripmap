//! Demo of the file discovery module.

use ripmap::discovery::find_source_files;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let cwd = Path::new(".");

    println!("🔍 Discovering source files in ripmap codebase...\n");

    let files = find_source_files(cwd, false)?;

    println!("✓ Found {} source files\n", files.len());

    // Group by extension
    let mut by_ext = std::collections::HashMap::new();
    for file in &files {
        if let Some(ext) = file.extension() {
            if let Some(ext_str) = ext.to_str() {
                *by_ext.entry(ext_str.to_string()).or_insert(0) += 1;
            }
        }
    }

    println!("Files by extension:");
    let mut exts: Vec<_> = by_ext.iter().collect();
    exts.sort_by_key(|(_, count)| std::cmp::Reverse(**count));
    for (ext, count) in exts {
        println!("  .{}: {}", ext, count);
    }

    println!("\nFirst 15 files:");
    for (i, file) in files.iter().take(15).enumerate() {
        println!(
            "  {}. {}",
            i + 1,
            file.strip_prefix(cwd).unwrap_or(file).display()
        );
    }

    if files.len() > 15 {
        println!("  ... and {} more", files.len() - 15);
    }

    // Verify gitignore is working
    println!("\n🔒 Gitignore verification:");
    let has_target = files
        .iter()
        .any(|f| f.to_string_lossy().contains("target/"));
    let has_lock = files
        .iter()
        .any(|f| f.to_string_lossy().ends_with("Cargo.lock"));
    println!("  Excludes target/: {}", !has_target);
    println!("  Excludes Cargo.lock: {}", !has_lock);

    Ok(())
}
