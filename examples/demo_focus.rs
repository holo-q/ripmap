//! Demo: Focus resolver - fuzzy matching and graph expansion
//!
//! This example demonstrates how the FocusResolver enables targeted
//! exploration of large codebases through natural queries.
//!
//! Run with: cargo run --example demo_focus

use std::collections::HashMap;
use std::sync::Arc;

use ripmap::ranking::FocusResolver;
use ripmap::types::{Tag, TagKind};

fn main() {
    println!("=== Focus Resolver Demo ===\n");

    // Create a mock codebase with authentication and parsing modules
    let (tags_by_file, symbol_graph) = create_mock_codebase();

    let resolver = FocusResolver::new("/project");

    // Demo 1: Fuzzy symbol matching
    println!("1. Fuzzy Symbol Matching");
    println!("   Query: 'auth'");
    let (_files, idents) = resolver.resolve(&["auth".to_string()], &tags_by_file);
    println!("   Matched identifiers:");
    for ident in &idents {
        println!("     - {}", ident);
    }
    println!();

    // Demo 2: File matching
    println!("2. File Matching");
    println!("   Query: 'auth.rs'");
    let (files, _idents) = resolver.resolve(&["auth.rs".to_string()], &tags_by_file);
    println!("   Matched files:");
    for file in &files {
        println!("     - {}", file);
    }
    println!();

    // Demo 3: Comma-separated targets
    println!("3. Multi-target Query");
    println!("   Query: 'parse,config.rs'");
    let (files, idents) = resolver.resolve(&["parse,config.rs".to_string()], &tags_by_file);
    println!("   Matched files: {}", files.len());
    println!("   Matched identifiers: {}", idents.len());
    for ident in &idents {
        println!("     - {}", ident);
    }
    println!();

    // Demo 4: Graph expansion
    println!("4. Graph Expansion (BFS with decay)");
    println!("   Starting from: 'authenticate'");
    let mut seed_idents = std::collections::HashSet::new();
    seed_idents.insert("authenticate".to_string());

    let expanded = resolver.expand_via_graph(&seed_idents, &symbol_graph, 2, 0.5);

    println!("   Expanded symbols (weight = decay^hop_distance):");
    let mut expanded_vec: Vec<_> = expanded.iter().collect();
    expanded_vec.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for ((file, symbol), weight) in expanded_vec {
        println!("     - {} in {} (weight: {:.2})", symbol, file, weight);
    }
    println!();

    // Demo 5: Typo tolerance
    println!("5. Typo Tolerance");
    println!("   Query: 'authentcate' (missing 'i')");
    let (_files, idents) = resolver.resolve(&["authentcate".to_string()], &tags_by_file);
    println!("   Matched identifiers:");
    for ident in &idents {
        println!("     - {}", ident);
    }
    println!();

    // Demo 6: Stem matching
    println!("6. Stem Matching (morphological variants)");
    println!("   Query: 'valid'");
    let (_files, idents) = resolver.resolve(&["valid".to_string()], &tags_by_file);
    println!("   Matched identifiers:");
    for ident in &idents {
        println!("     - {}", ident);
    }
    println!();

    println!("=== Demo Complete ===");
}

/// Create a mock codebase for demonstration purposes.
///
/// Returns:
/// - tags_by_file: Map of file paths to tags
/// - symbol_graph: Call graph edges (from_file, from_sym, to_file, to_sym)
fn create_mock_codebase() -> (
    HashMap<String, Vec<Tag>>,
    Vec<(Arc<str>, Arc<str>, Arc<str>, Arc<str>)>,
) {
    let mut tags_by_file = HashMap::new();

    // auth.rs - Authentication module
    tags_by_file.insert(
        "/project/src/auth.rs".to_string(),
        vec![
            Tag {
                rel_fname: "src/auth.rs".into(),
                fname: "/project/src/auth.rs".into(),
                line: 10,
                name: "authenticate".into(),
                kind: TagKind::Def,
                node_type: "function".into(),
                parent_name: None,
                parent_line: None,
                signature: None,
                fields: None,
                metadata: None,
            },
            Tag {
                rel_fname: "src/auth.rs".into(),
                fname: "/project/src/auth.rs".into(),
                line: 20,
                name: "authorize".into(),
                kind: TagKind::Def,
                node_type: "function".into(),
                parent_name: None,
                parent_line: None,
                signature: None,
                fields: None,
                metadata: None,
            },
            Tag {
                rel_fname: "src/auth.rs".into(),
                fname: "/project/src/auth.rs".into(),
                line: 30,
                name: "validate_token".into(),
                kind: TagKind::Def,
                node_type: "function".into(),
                parent_name: None,
                parent_line: None,
                signature: None,
                fields: None,
                metadata: None,
            },
        ],
    );

    // parser.rs - Parsing module
    tags_by_file.insert(
        "/project/src/parser.rs".to_string(),
        vec![
            Tag {
                rel_fname: "src/parser.rs".into(),
                fname: "/project/src/parser.rs".into(),
                line: 10,
                name: "parse".into(),
                kind: TagKind::Def,
                node_type: "function".into(),
                parent_name: None,
                parent_line: None,
                signature: None,
                fields: None,
                metadata: None,
            },
            Tag {
                rel_fname: "src/parser.rs".into(),
                fname: "/project/src/parser.rs".into(),
                line: 20,
                name: "parser_error".into(),
                kind: TagKind::Def,
                node_type: "function".into(),
                parent_name: None,
                parent_line: None,
                signature: None,
                fields: None,
                metadata: None,
            },
        ],
    );

    // config.rs - Configuration module
    tags_by_file.insert(
        "/project/src/config.rs".to_string(),
        vec![
            Tag {
                rel_fname: "src/config.rs".into(),
                fname: "/project/src/config.rs".into(),
                line: 10,
                name: "load_config".into(),
                kind: TagKind::Def,
                node_type: "function".into(),
                parent_name: None,
                parent_line: None,
                signature: None,
                fields: None,
                metadata: None,
            },
            Tag {
                rel_fname: "src/config.rs".into(),
                fname: "/project/src/config.rs".into(),
                line: 20,
                name: "validate_config".into(),
                kind: TagKind::Def,
                node_type: "function".into(),
                parent_name: None,
                parent_line: None,
                signature: None,
                fields: None,
                metadata: None,
            },
        ],
    );

    // validator.rs - Validation module
    tags_by_file.insert(
        "/project/src/validator.rs".to_string(),
        vec![Tag {
            rel_fname: "src/validator.rs".into(),
            fname: "/project/src/validator.rs".into(),
            line: 10,
            name: "validate".into(),
            kind: TagKind::Def,
            node_type: "function".into(),
            parent_name: None,
            parent_line: None,
            signature: None,
            fields: None,
            metadata: None,
        }],
    );

    // Build symbol graph - who calls whom?
    // authenticate -> validate_token -> validate
    // load_config -> validate_config -> validate
    // parse -> parser_error
    let symbol_graph = vec![
        // authenticate calls validate_token
        (
            Arc::from("/project/src/auth.rs"),
            Arc::from("authenticate"),
            Arc::from("/project/src/auth.rs"),
            Arc::from("validate_token"),
        ),
        // validate_token calls validate
        (
            Arc::from("/project/src/auth.rs"),
            Arc::from("validate_token"),
            Arc::from("/project/src/validator.rs"),
            Arc::from("validate"),
        ),
        // load_config calls validate_config
        (
            Arc::from("/project/src/config.rs"),
            Arc::from("load_config"),
            Arc::from("/project/src/config.rs"),
            Arc::from("validate_config"),
        ),
        // validate_config calls validate
        (
            Arc::from("/project/src/config.rs"),
            Arc::from("validate_config"),
            Arc::from("/project/src/validator.rs"),
            Arc::from("validate"),
        ),
        // parse calls parser_error
        (
            Arc::from("/project/src/parser.rs"),
            Arc::from("parse"),
            Arc::from("/project/src/parser.rs"),
            Arc::from("parser_error"),
        ),
    ];

    (tags_by_file, symbol_graph)
}
