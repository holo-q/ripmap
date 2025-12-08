//! ANSI color utilities and badge rendering for ripmap.
//!
//! Badge system reveals structural and temporal patterns:
//! - Structural: [bridge], [api] - what role it plays in the graph
//! - Temporal: [recent], [high-churn] - what's happening now
//! - Lifecycle: [crystal], [rotting], [emergent], [evolving] - where it is in its arc
//!
//! Color scheme optimized for both light and dark terminals:
//! - High contrast for critical info (file headers, class names)
//! - Muted colors for metadata (badges, decorators)
//! - Semantic colors for syntax (functions=green, types=cyan)

use owo_colors::{OwoColorize, Style};
use std::fmt;

/// Badge types for file and symbol annotations.
/// Each badge reveals different information about structure and evolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Badge {
    // Structural badges - role in the graph
    /// Load-bearing component - high betweenness centrality
    /// Removing this disconnects the graph
    Bridge,

    /// Public API surface - called from many places
    /// Entry points and facades
    Api,

    // Temporal badges - recent activity
    /// Modified in the last 7 days (configurable)
    Recent,

    /// High commit frequency (10+ commits, configurable)
    HighChurn,

    // Lifecycle phase badges - maturity arc
    /// Old (180+ days) and stable (30+ days quiet)
    /// Settled code, safe to rely on
    Crystal,

    /// Old (90+ days) but recently churning
    /// Tech debt surfacing, refactor candidate
    Rotting,

    /// New file (< 30 days)
    /// Still finding its shape
    Emergent,

    /// Normal development state
    /// Actively being worked on
    Evolving,
}

impl Badge {
    /// Get the badge label for display
    pub fn label(&self) -> &'static str {
        match self {
            Badge::Bridge => "bridge",
            Badge::Api => "api",
            Badge::Recent => "recent",
            Badge::HighChurn => "high-churn",
            Badge::Crystal => "crystal",
            Badge::Rotting => "rotting",
            Badge::Emergent => "emergent",
            Badge::Evolving => "evolving",
        }
    }

    /// Get the badge's display color/style
    pub fn style(&self) -> Style {
        match self {
            // Structural badges - bright to catch attention
            Badge::Bridge => Style::new().bright_red().bold(),
            Badge::Api => Style::new().bright_blue().bold(),

            // Temporal badges - yellow/orange tones
            Badge::Recent => Style::new().yellow(),
            Badge::HighChurn => Style::new().bright_yellow(),

            // Lifecycle badges - muted, informational
            Badge::Crystal => Style::new().bright_cyan().dimmed(),
            Badge::Rotting => Style::new().bright_magenta(),
            Badge::Emergent => Style::new().green(),
            Badge::Evolving => Style::new().dimmed(),
        }
    }

    /// Render the badge with color
    pub fn render(&self) -> String {
        format!("[{}]", self.label().style(self.style()))
    }
}

impl fmt::Display for Badge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.render())
    }
}

/// Colorize different symbol types for directory rendering.
/// Provides semantic highlighting similar to LSP-based editors.
pub struct Colorizer;

impl Colorizer {
    /// Colorize a file path (bold blue for headers)
    pub fn file_path(s: &str) -> String {
        s.bright_blue().bold().to_string()
    }

    /// Colorize a class/struct name (magenta)
    pub fn class_name(s: &str) -> String {
        s.magenta().to_string()
    }

    /// Colorize a function/method name (green)
    pub fn function_name(s: &str) -> String {
        s.green().to_string()
    }

    /// Colorize a constant name (bright cyan)
    pub fn constant_name(s: &str) -> String {
        s.bright_cyan().to_string()
    }

    /// Colorize a type annotation (cyan)
    pub fn type_name(s: &str) -> String {
        s.cyan().to_string()
    }

    /// Colorize field names (normal white/default)
    pub fn field_name(s: &str) -> String {
        s.to_string()
    }

    /// Colorize decorators/attributes (dimmed yellow)
    pub fn decorator(s: &str) -> String {
        s.yellow().dimmed().to_string()
    }

    /// Colorize badges (dim yellow for low visual weight)
    pub fn badge_group(badges: &[Badge]) -> String {
        if badges.is_empty() {
            return String::new();
        }

        let rendered: Vec<_> = badges.iter().map(|b| b.render()).collect();
        rendered.join(" ")
    }

    /// Colorize temporal coupling info (e.g., "⇄ changes with:")
    pub fn coupling_label() -> String {
        "⇄ changes with:".dimmed().to_string()
    }

    /// Colorize a coupling entry (file name + percentage)
    pub fn coupling_entry(file: &str, percentage: f64) -> String {
        format!("{}({}%)", file.cyan(), (percentage * 100.0) as u32)
    }

    /// Dim text for secondary information (call relationships, metadata)
    pub fn dim(s: &str) -> String {
        s.dimmed().to_string()
    }
}

/// Convenience function for colorizing based on node type.
/// Maps AST node types to appropriate color schemes.
pub fn colorize(node_type: &str, name: &str) -> String {
    match node_type {
        "class" | "class_definition" | "struct" | "struct_item" => Colorizer::class_name(name),
        "function" | "function_definition" | "method" | "method_definition" => {
            Colorizer::function_name(name)
        }
        "constant" | "const_item" => Colorizer::constant_name(name),
        "type" | "type_alias" | "type_definition" => Colorizer::type_name(name),
        "decorator" | "attribute" => Colorizer::decorator(name),
        _ => name.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_badge_labels() {
        assert_eq!(Badge::Bridge.label(), "bridge");
        assert_eq!(Badge::Api.label(), "api");
        assert_eq!(Badge::Recent.label(), "recent");
        assert_eq!(Badge::HighChurn.label(), "high-churn");
        assert_eq!(Badge::Crystal.label(), "crystal");
        assert_eq!(Badge::Rotting.label(), "rotting");
        assert_eq!(Badge::Emergent.label(), "emergent");
        assert_eq!(Badge::Evolving.label(), "evolving");
    }

    #[test]
    fn test_badge_render() {
        // Just ensure they don't panic
        let badges = vec![
            Badge::Bridge,
            Badge::Api,
            Badge::Recent,
            Badge::HighChurn,
            Badge::Crystal,
            Badge::Rotting,
            Badge::Emergent,
            Badge::Evolving,
        ];

        for badge in badges {
            let rendered = badge.render();
            assert!(rendered.contains(badge.label()));
        }
    }

    #[test]
    fn test_colorize_node_types() {
        // These should apply different colors (we can't test actual ANSI codes easily)
        // but we verify the function doesn't panic
        colorize("class", "MyClass");
        colorize("function", "my_func");
        colorize("constant", "MAX_SIZE");
        colorize("type", "UserId");
        colorize("unknown", "something");
    }
}
