//! Promptgram: treating prompts as structured programs.
//!
//! A promptgram is not a blob of text but a structured program with sections
//! that can be independently evolved by the outer loop.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// A promptgram: a structured prompt treated as a program.
///
/// Each section serves a specific role and can be independently modified
/// by the outer loop optimizer.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Promptgram {
    /// Unique identifier for this promptgram
    pub id: String,

    /// Parent promptgram ID (if this was derived from another)
    #[serde(default)]
    pub parent_id: Option<String>,

    /// Version number (increments with each edit)
    pub version: usize,

    /// The structured sections of the prompt
    pub sections: HashMap<String, PromptSection>,

    /// Metadata about this promptgram
    pub metadata: PromptgramMetadata,
}

/// A section of a promptgram.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptSection {
    /// Section name (Role, Policy, Heuristics, etc.)
    pub name: String,

    /// Section content (markdown/text)
    pub content: String,

    /// Is this section immutable (cannot be edited by L2)?
    #[serde(default)]
    pub immutable: bool,

    /// Tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Metadata about a promptgram.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PromptgramMetadata {
    /// When this promptgram was created
    pub created_at: i64,

    /// Last modified timestamp
    pub modified_at: i64,

    /// Best NDCG achieved with this promptgram
    pub best_ndcg: f64,

    /// Number of inner runs using this promptgram
    pub run_count: usize,

    /// Human-readable description
    pub description: String,

    /// Lineage: IDs of ancestor promptgrams
    #[serde(default)]
    pub lineage: Vec<String>,
}

/// Standard section names for inner promptgrams (L1).
pub mod sections {
    pub const ROLE: &str = "Role";
    pub const API_CONTRACT: &str = "API_contract";
    pub const POLICY: &str = "Policy";
    pub const HEURISTICS: &str = "Heuristics";
    pub const CURRICULUM: &str = "Curriculum";
    pub const OUTPUT_SCHEMA: &str = "Output_schema";
    pub const STYLE: &str = "Style";
}

impl Promptgram {
    /// Create a new promptgram with the given ID.
    pub fn new(id: impl Into<String>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        Promptgram {
            id: id.into(),
            parent_id: None,
            version: 1,
            sections: HashMap::new(),
            metadata: PromptgramMetadata {
                created_at: now,
                modified_at: now,
                ..Default::default()
            },
        }
    }

    /// Add or update a section.
    pub fn with_section(mut self, name: &str, content: &str, immutable: bool) -> Self {
        self.sections.insert(
            name.to_string(),
            PromptSection {
                name: name.to_string(),
                content: content.to_string(),
                immutable,
                tags: vec![],
            },
        );
        self
    }

    /// Get a section by name.
    pub fn get_section(&self, name: &str) -> Option<&PromptSection> {
        self.sections.get(name)
    }

    /// Render the promptgram to a single prompt string.
    ///
    /// Sections are rendered in a canonical order with markdown headers.
    pub fn render(&self) -> String {
        let section_order = [
            sections::ROLE,
            sections::API_CONTRACT,
            sections::POLICY,
            sections::HEURISTICS,
            sections::CURRICULUM,
            sections::OUTPUT_SCHEMA,
            sections::STYLE,
        ];

        let mut output = String::new();

        // Render sections in canonical order
        for section_name in &section_order {
            if let Some(section) = self.sections.get(*section_name) {
                output.push_str(&format!("## {}\n\n", section.name));
                output.push_str(&section.content);
                output.push_str("\n\n");
            }
        }

        // Render any additional sections not in canonical order
        for (name, section) in &self.sections {
            if !section_order.contains(&name.as_str()) {
                output.push_str(&format!("## {}\n\n", section.name));
                output.push_str(&section.content);
                output.push_str("\n\n");
            }
        }

        output.trim().to_string()
    }

    /// Create a child promptgram (fork with new ID, linked lineage).
    pub fn fork(&self, new_id: impl Into<String>) -> Self {
        let new_id = new_id.into();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        let mut lineage = self.metadata.lineage.clone();
        lineage.push(self.id.clone());

        Promptgram {
            id: new_id,
            parent_id: Some(self.id.clone()),
            version: 1,
            sections: self.sections.clone(),
            metadata: PromptgramMetadata {
                created_at: now,
                modified_at: now,
                best_ndcg: 0.0,
                run_count: 0,
                description: format!("Fork of {}", self.id),
                lineage,
            },
        }
    }

    /// Apply an edit to a section.
    ///
    /// Returns Err if the section is immutable or doesn't exist (for replace/delete).
    pub fn apply_edit(&mut self, edit: &super::PromptEdit) -> Result<(), String> {
        let section = self
            .sections
            .get_mut(&edit.section)
            .ok_or_else(|| format!("Section '{}' not found", edit.section))?;

        if section.immutable {
            return Err(format!("Section '{}' is immutable", edit.section));
        }

        match edit.edit_type.as_str() {
            "append" => {
                section.content.push_str("\n\n");
                section.content.push_str(&edit.content);
            }
            "replace" => {
                let target = edit.target.as_deref().unwrap_or("");
                if target.is_empty() {
                    // Replace entire section
                    section.content = edit.content.clone();
                } else {
                    // Replace specific target text
                    section.content = section.content.replace(target, &edit.content);
                }
            }
            "delete" => {
                let target = edit.target.as_deref().unwrap_or("");
                section.content = section.content.replace(target, "");
            }
            _ => return Err(format!("Unknown edit type: {}", edit.edit_type)),
        }

        self.version += 1;
        self.metadata.modified_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        Ok(())
    }

    /// Load a promptgram from a TOML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| format!("Failed to read promptgram: {}", e))?;
        toml::from_str(&content).map_err(|e| format!("Failed to parse promptgram: {}", e))
    }

    /// Save the promptgram to a TOML file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize promptgram: {}", e))?;
        std::fs::write(path.as_ref(), content)
            .map_err(|e| format!("Failed to write promptgram: {}", e))
    }

    /// Load from markdown with section headers.
    ///
    /// Format:
    /// ```markdown
    /// ## Role
    /// Content here...
    ///
    /// ## Policy
    /// More content...
    /// ```
    ///
    /// Architecture note: For inner promptgrams (L1), only "Role" is immutable.
    /// Protocol sections (API_contract, Output_schema) are injected at runtime
    /// by reasoning.rs and should not be in the markdown files that L2 evolves.
    pub fn from_markdown(id: &str, content: &str) -> Self {
        let mut promptgram = Self::new(id);
        let mut current_section: Option<String> = None;
        let mut current_content = String::new();

        for line in content.lines() {
            if line.starts_with("## ") {
                // Save previous section if any
                if let Some(section_name) = current_section.take() {
                    // For inner promptgrams (L1): only Role is immutable
                    // Protocol sections (API_contract, Output_schema) are injected at runtime
                    // and kept immutable to prevent L2 from modifying the protocol itself
                    let immutable = matches!(
                        section_name.as_str(),
                        "Role" | "API_contract" | "Output_schema"
                    );
                    promptgram =
                        promptgram.with_section(&section_name, current_content.trim(), immutable);
                    current_content.clear();
                }

                // Start new section
                current_section = Some(line[3..].trim().to_string());
            } else if current_section.is_some() {
                current_content.push_str(line);
                current_content.push('\n');
            }
        }

        // Save last section
        if let Some(section_name) = current_section {
            let immutable = matches!(
                section_name.as_str(),
                "Role" | "API_contract" | "Output_schema"
            );
            promptgram = promptgram.with_section(&section_name, current_content.trim(), immutable);
        }

        promptgram
    }
}

/// Diff two promptgrams section by section.
///
/// Returns a list of differences for L2 to understand what changed.
/// This helps L2 reason about which mutations were effective.
pub fn diff_prompts(a: &Promptgram, b: &Promptgram) -> Vec<PromptDiff> {
    let mut diffs = Vec::new();

    // Check all sections in A
    for (name, section_a) in &a.sections {
        match b.sections.get(name) {
            Some(section_b) => {
                if section_a.content != section_b.content {
                    diffs.push(PromptDiff {
                        section: name.clone(),
                        diff_type: DiffType::Modified,
                        before: Some(section_a.content.clone()),
                        after: Some(section_b.content.clone()),
                        lines_added: count_lines(&section_b.content)
                            .saturating_sub(count_lines(&section_a.content)),
                        lines_removed: count_lines(&section_a.content)
                            .saturating_sub(count_lines(&section_b.content)),
                    });
                }
            }
            None => {
                diffs.push(PromptDiff {
                    section: name.clone(),
                    diff_type: DiffType::Removed,
                    before: Some(section_a.content.clone()),
                    after: None,
                    lines_added: 0,
                    lines_removed: count_lines(&section_a.content),
                });
            }
        }
    }

    // Check for sections in B but not in A
    for (name, section_b) in &b.sections {
        if !a.sections.contains_key(name) {
            diffs.push(PromptDiff {
                section: name.clone(),
                diff_type: DiffType::Added,
                before: None,
                after: Some(section_b.content.clone()),
                lines_added: count_lines(&section_b.content),
                lines_removed: 0,
            });
        }
    }

    diffs
}

/// A difference between two promptgram sections.
#[derive(Debug, Clone)]
pub struct PromptDiff {
    pub section: String,
    pub diff_type: DiffType,
    pub before: Option<String>,
    pub after: Option<String>,
    pub lines_added: usize,
    pub lines_removed: usize,
}

impl PromptDiff {
    /// Format as a compact summary string.
    pub fn summary(&self) -> String {
        match self.diff_type {
            DiffType::Added => format!(
                "[+{}] {} (+{} lines)",
                self.section, "added", self.lines_added
            ),
            DiffType::Removed => format!(
                "[-{}] {} (-{} lines)",
                self.section, "removed", self.lines_removed
            ),
            DiffType::Modified => format!(
                "[~{}] modified (+{}/-{})",
                self.section, self.lines_added, self.lines_removed
            ),
        }
    }
}

/// Type of difference between sections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffType {
    Added,
    Removed,
    Modified,
}

fn count_lines(s: &str) -> usize {
    s.lines().count()
}

/// Create baseline inner promptgram (L1 v001).
///
/// This is the seed promptgram - L2 will mutate it over time.
/// Version numbers track lineage, not archetypes.
///
/// NOTE: Protocol sections (API_contract, Output_schema) are now injected
/// at runtime by reasoning.rs from training/prompts/protocol/inner_output_schema.md.
/// The baseline only contains the policy sections that L2 can evolve:
/// Role, Policy, Heuristics, and Style.
pub fn baseline_promptgram() -> Promptgram {
    Promptgram::new("inner_v001")
        // Role: immutable - defines the agent's fundamental identity
        .with_section(
            sections::ROLE,
            r#"You approximate the gradient in concept space.
Given failures and trajectory, propose parameter changes."#,
            true,
        )
        // Policy: mutable - high-level approach to the problem
        .with_section(
            sections::POLICY,
            r#"Analyze trajectory state:
- Improving: continue direction
- Degrading: revert or reverse
- Plateaued: orthogonal move

Analyze failures:
- Missing signal vs overwhelming signal
- Parameter interactions"#,
            false,
        )
        // Heuristics: mutable - specific rules and patterns discovered
        .with_section(
            sections::HEURISTICS,
            r#"- NDCG drop >5% = collapse signal
- Temporal and structural signals compete
- High boosts cause tunnel vision
- Low alpha localizes, high alpha globalizes
- Depth penalties break monorepos"#,
            false,
        )
        // Style: mutable - tone and presentation guidance
        .with_section(
            sections::STYLE,
            r#"Analytical. Specific. Reference concrete failures."#,
            false,
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_promptgram_render() {
        let pg = Promptgram::new("test")
            .with_section(sections::ROLE, "You are a test", true)
            .with_section(sections::POLICY, "Do the thing", false);

        let rendered = pg.render();
        assert!(rendered.contains("## Role"));
        assert!(rendered.contains("You are a test"));
        assert!(rendered.contains("## Policy"));
    }

    #[test]
    fn test_promptgram_fork() {
        let parent = Promptgram::new("parent").with_section(sections::POLICY, "Original", false);

        let child = parent.fork("child");

        assert_eq!(child.parent_id, Some("parent".to_string()));
        assert_eq!(child.metadata.lineage, vec!["parent"]);
        assert!(child.get_section(sections::POLICY).is_some());
    }

    #[test]
    fn test_promptgram_edit_immutable() {
        let mut pg = Promptgram::new("test").with_section(sections::ROLE, "Original", true);

        let edit = super::super::PromptEdit {
            section: sections::ROLE.to_string(),
            edit_type: "replace".to_string(),
            target: Some(String::new()),
            content: "Modified".to_string(),
            rationale: "test".to_string(),
        };

        assert!(pg.apply_edit(&edit).is_err());
    }

    #[test]
    fn test_from_markdown() {
        let md = r#"## Role
You are a test optimizer.

## Policy
Do smart things.
Make good choices.

## Style
Be brief.
"#;

        let pg = Promptgram::from_markdown("test", md);
        assert!(pg.get_section("Role").is_some());
        assert!(pg.get_section("Policy").is_some());
        assert!(pg.get_section("Style").is_some());
    }
}
