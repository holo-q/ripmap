//! Configuration loading from project config files.
//!
//! Ripmap auto-detects config from your existing tooling. Just run `ripmap` and it
//! will find your project's include/exclude patterns automatically.
//!
//! ## Supported config files (in priority order)
//!
//! ### Language-specific
//! | File | Sections checked |
//! |------|-----------------|
//! | `ripmap.toml` | root (native config) |
//! | `pyproject.toml` | `[tool.ripmap]` → `[tool.ty.environment]` → `[tool.ruff]` → `[tool.pyright]` → `[tool.mypy]` → `[tool.black]` → `[tool.isort]` → `[tool.pytest]` |
//! | `package.json` | `ripmap` → `eslintConfig` |
//! | `tsconfig.json` / `jsconfig.json` | `include`/`exclude` |
//! | `biome.json` | `files.include`/`files.ignore` |
//! | `deno.json` | `include`/`exclude` + `lint`/`fmt` sections |
//! | `jest.config.json` | `roots`, `testMatch`, `testPathIgnorePatterns` |
//! | `Cargo.toml` | `[workspace]` members/exclude |
//! | `composer.json` | `autoload.psr-4` directories |
//!
//! ### Monorepo tools
//! | File | What's extracted |
//! |------|-----------------|
//! | `nx.json` | `workspaceLayout.appsDir`, `workspaceLayout.libsDir` |
//! | `turbo.json` | `globalDependencies` |
//!
//! ### Gitignore-style (fallback)
//! | File | Notes |
//! |------|-------|
//! | `.ripmapignore` | Native ignore file |
//! | `.eslintignore` | ESLint (legacy, flat config preferred) |
//! | `.prettierignore` | Prettier (true gitignore syntax) |
//! | `.stylelintignore` | Stylelint (uses node-ignore) |
//!
//! Note: `.dockerignore` is intentionally unsupported - its pattern matching
//! has inverted anchoring behavior compared to `.gitignore`.
//!
//! ## Example ripmap.toml
//!
//! ```toml
//! include = ["src/**", "lib/**"]
//! exclude = ["**/generated/**"]
//! extend-exclude = ["**/vendor/**"]
//! src = ["src", "lib"]
//! ```

use owo_colors::OwoColorize;
use serde::Deserialize;
use serde_json::Value as JsonValue;
use std::path::{Path, PathBuf};

/// Default exclude patterns (common non-source directories).
pub const DEFAULT_EXCLUDES: &[&str] = &[
    "**/node_modules/**",
    "**/.git/**",
    "**/target/**",
    "**/build/**",
    "**/dist/**",
    "**/__pycache__/**",
    "**/.venv/**",
    "**/venv/**",
    "**/.tox/**",
    "**/.mypy_cache/**",
    "**/.pytest_cache/**",
    "**/.ruff_cache/**",
    "**/vendor/**",
    "**/third_party/**",
    "**/.next/**",
    "**/.nuxt/**",
];

/// Ripmap configuration.
#[derive(Debug, Clone, Default)]
pub struct Config {
    /// Source file for this config (for display), with tool name if inherited.
    pub source: Option<String>,

    /// Glob patterns for files to include. If empty, include all source files.
    pub include: Vec<String>,

    /// Glob patterns for files to exclude. Replaces defaults if set.
    pub exclude: Vec<String>,

    /// Additional exclude patterns (extends defaults).
    pub extend_exclude: Vec<String>,

    /// Source root directories (affects depth weighting).
    pub src: Vec<PathBuf>,
}

// ============================================================================
// TOOL ADAPTER FRAMEWORK
// ============================================================================
// Declarative config extraction from any project config file.
// Each adapter knows how to find include/exclude patterns in its format.

/// Extracted configuration from a tool adapter.
#[derive(Debug, Default)]
struct ExtractedConfig {
    include: Vec<String>,
    exclude: Vec<String>,
    extend_exclude: Vec<String>,
    src: Vec<String>,
}

impl ExtractedConfig {
    fn is_empty(&self) -> bool {
        self.include.is_empty() && self.exclude.is_empty() && self.extend_exclude.is_empty()
    }
}

/// A tool adapter knows how to extract config from a specific tool's section.
struct ToolAdapter {
    /// Tool name for display (e.g., "ruff", "eslint")
    name: &'static str,
    /// Optional deprecation snark (shown when this fallback is used)
    snark: Option<fn()>,
    /// Extract config from parsed JSON/TOML value
    extract: fn(&JsonValue) -> Option<ExtractedConfig>,
}

/// Generate the schizo sparkle prefix/suffix for deprecated tools
fn sparkle_banner(msg: &str, highlight: &str) {
    let sparkle_chars = ['~', '*', '`', '.', '+', '^'];
    let colors: Vec<fn(&str) -> String> = vec![
        |s| s.bright_magenta().to_string(),
        |s| s.bright_cyan().to_string(),
        |s| s.bright_yellow().to_string(),
        |s| s.bright_green().to_string(),
        |s| s.bright_red().to_string(),
        |s| s.bright_blue().to_string(),
    ];
    let prefix: String = (0..12)
        .map(|i| colors[i % colors.len()](&sparkle_chars[i % sparkle_chars.len()].to_string()))
        .collect();
    let suffix: String = (0..12)
        .rev()
        .map(|i| {
            colors[(i + 2) % colors.len()](
                &sparkle_chars[(i + 3) % sparkle_chars.len()].to_string(),
            )
        })
        .collect();
    eprintln!(
        "   {} {} use {} instead {}",
        prefix,
        msg.bright_red(),
        highlight.bright_cyan().bold().underline(),
        suffix
    );
}

// ============================================================================
// PYPROJECT.TOML ADAPTERS (Python ecosystem)
// ============================================================================

fn extract_ripmap(v: &JsonValue) -> Option<ExtractedConfig> {
    let tool = v.get("tool")?.get("ripmap")?;
    Some(ExtractedConfig {
        include: json_string_array(tool.get("include")),
        exclude: json_string_array(tool.get("exclude")),
        extend_exclude: json_string_array(tool.get("extend-exclude")),
        src: json_string_array(tool.get("src")),
    })
}

fn extract_ty(v: &JsonValue) -> Option<ExtractedConfig> {
    let ty = v.get("tool")?.get("ty")?;

    // ty uses environment.root for project paths (not include/exclude)
    // See: https://docs.astral.sh/ty/reference/configuration/
    let mut src = Vec::new();
    if let Some(env) = ty.get("environment") {
        src = json_string_array(env.get("root"));
    }

    // Also check legacy src.include/exclude for backwards compat
    let mut include = Vec::new();
    let mut exclude = Vec::new();
    if let Some(src_section) = ty.get("src") {
        include = json_string_array(src_section.get("include"));
        exclude = json_string_array(src_section.get("exclude"));
    }

    if src.is_empty() && include.is_empty() && exclude.is_empty() {
        None
    } else {
        Some(ExtractedConfig { include, exclude, src, ..Default::default() })
    }
}

fn extract_ruff(v: &JsonValue) -> Option<ExtractedConfig> {
    let ruff = v.get("tool")?.get("ruff")?;
    let cfg = ExtractedConfig {
        include: json_string_array(ruff.get("include")),
        exclude: json_string_array(ruff.get("exclude")),
        extend_exclude: json_string_array(ruff.get("extend-exclude")),
        src: json_string_array(ruff.get("src")),
    };
    if cfg.is_empty() { None } else { Some(cfg) }
}

fn extract_pyright(v: &JsonValue) -> Option<ExtractedConfig> {
    let pyright = v.get("tool")?.get("pyright")?;
    let cfg = ExtractedConfig {
        include: json_string_array(pyright.get("include")),
        exclude: json_string_array(pyright.get("exclude")),
        ..Default::default()
    };
    if cfg.is_empty() { None } else { Some(cfg) }
}

fn extract_mypy(v: &JsonValue) -> Option<ExtractedConfig> {
    let mypy = v.get("tool")?.get("mypy")?;
    let cfg = ExtractedConfig {
        exclude: json_string_array(mypy.get("exclude")),
        ..Default::default()
    };
    if cfg.is_empty() { None } else { Some(cfg) }
}

fn extract_black(v: &JsonValue) -> Option<ExtractedConfig> {
    let black = v.get("tool")?.get("black")?;
    let cfg = ExtractedConfig {
        include: json_string_array(black.get("include"))
            .into_iter()
            .chain(std::iter::once(black.get("include")?.as_str()?.to_string()))
            .filter(|s| !s.is_empty())
            .collect(),
        exclude: json_string_array(black.get("exclude")),
        extend_exclude: json_string_array(black.get("extend-exclude")),
        ..Default::default()
    };
    if cfg.is_empty() { None } else { Some(cfg) }
}

fn extract_isort(v: &JsonValue) -> Option<ExtractedConfig> {
    let isort = v.get("tool")?.get("isort")?;
    let cfg = ExtractedConfig {
        exclude: json_string_array(isort.get("skip")),
        extend_exclude: json_string_array(isort.get("skip_glob")),
        src: json_string_array(isort.get("src_paths")),
        ..Default::default()
    };
    if cfg.is_empty() { None } else { Some(cfg) }
}

fn extract_pytest(v: &JsonValue) -> Option<ExtractedConfig> {
    let pytest = v.get("tool")?.get("pytest")?.get("ini_options")?;
    let cfg = ExtractedConfig {
        include: json_string_array(pytest.get("testpaths")),
        exclude: json_string_array(pytest.get("norecursedirs")),
        ..Default::default()
    };
    if cfg.is_empty() { None } else { Some(cfg) }
}

// ============================================================================
// PACKAGE.JSON ADAPTERS (Node/TypeScript ecosystem)
// ============================================================================

fn extract_pkg_ripmap(v: &JsonValue) -> Option<ExtractedConfig> {
    let ripmap = v.get("ripmap")?;
    Some(ExtractedConfig {
        include: json_string_array(ripmap.get("include")),
        exclude: json_string_array(ripmap.get("exclude")),
        ..Default::default()
    })
}

fn extract_pkg_eslint(v: &JsonValue) -> Option<ExtractedConfig> {
    let eslint = v.get("eslintConfig")?;
    let cfg = ExtractedConfig {
        exclude: json_string_array(eslint.get("ignorePatterns")),
        ..Default::default()
    };
    if cfg.is_empty() { None } else { Some(cfg) }
}

fn extract_pkg_prettier(v: &JsonValue) -> Option<ExtractedConfig> {
    // prettier in package.json doesn't typically have include/exclude
    // but we check for overrides patterns
    let _prettier = v.get("prettier")?;
    None
}

// ============================================================================
// TSCONFIG.JSON ADAPTER
// ============================================================================

fn extract_tsconfig(v: &JsonValue) -> Option<ExtractedConfig> {
    let cfg = ExtractedConfig {
        include: json_string_array(v.get("include")),
        exclude: json_string_array(v.get("exclude")),
        ..Default::default()
    };
    if cfg.is_empty() { None } else { Some(cfg) }
}

// ============================================================================
// BIOME.JSON ADAPTER (modern JS/TS tooling)
// ============================================================================

fn extract_biome(v: &JsonValue) -> Option<ExtractedConfig> {
    let files = v.get("files")?;
    let cfg = ExtractedConfig {
        include: json_string_array(files.get("include")),
        exclude: json_string_array(files.get("ignore")),
        ..Default::default()
    };
    if cfg.is_empty() { None } else { Some(cfg) }
}

// ============================================================================
// DENO.JSON ADAPTER
// ============================================================================

fn extract_deno(v: &JsonValue) -> Option<ExtractedConfig> {
    // Check lint, fmt, or root level
    let lint = v.get("lint");
    let fmt = v.get("fmt");

    let mut include = json_string_array(v.get("include"));
    let mut exclude = json_string_array(v.get("exclude"));

    if let Some(l) = lint {
        include.extend(json_string_array(l.get("include")));
        exclude.extend(json_string_array(l.get("exclude")));
    }
    if let Some(f) = fmt {
        include.extend(json_string_array(f.get("include")));
        exclude.extend(json_string_array(f.get("exclude")));
    }

    let cfg = ExtractedConfig {
        include,
        exclude,
        ..Default::default()
    };
    if cfg.is_empty() { None } else { Some(cfg) }
}

// ============================================================================
// CARGO.TOML ADAPTER (Rust)
// ============================================================================

fn extract_cargo(v: &JsonValue) -> Option<ExtractedConfig> {
    // Cargo doesn't have include/exclude in the same way, but we can extract
    // workspace members and exclude patterns
    let workspace = v.get("workspace")?;
    let cfg = ExtractedConfig {
        include: json_string_array(workspace.get("members")),
        exclude: json_string_array(workspace.get("exclude")),
        ..Default::default()
    };
    if cfg.is_empty() { None } else { Some(cfg) }
}

// ============================================================================
// ESLINT FLAT CONFIG (eslint.config.js/mjs) - parsed as JSON export
// ============================================================================

fn extract_eslint_flat(v: &JsonValue) -> Option<ExtractedConfig> {
    // eslint.config.js exports an array, we look for ignores in the objects
    let arr = v.as_array()?;
    let mut exclude = Vec::new();
    for item in arr {
        if let Some(ignores) = item.get("ignores").and_then(|v| v.as_array()) {
            for ig in ignores {
                if let Some(s) = ig.as_str() {
                    exclude.push(s.to_string());
                }
            }
        }
    }
    if exclude.is_empty() {
        None
    } else {
        Some(ExtractedConfig { exclude, ..Default::default() })
    }
}

// ============================================================================
// JEST CONFIG (jest.config.js/json)
// ============================================================================

fn extract_jest(v: &JsonValue) -> Option<ExtractedConfig> {
    let cfg = ExtractedConfig {
        include: json_string_array(v.get("roots"))
            .into_iter()
            .chain(json_string_array(v.get("testMatch")))
            .collect(),
        exclude: json_string_array(v.get("testPathIgnorePatterns"))
            .into_iter()
            .chain(json_string_array(v.get("modulePathIgnorePatterns")))
            .chain(json_string_array(v.get("coveragePathIgnorePatterns")))
            .collect(),
        ..Default::default()
    };
    if cfg.is_empty() { None } else { Some(cfg) }
}

// ============================================================================
// VITEST CONFIG (vitest.config.ts exports)
// ============================================================================

fn extract_vitest(v: &JsonValue) -> Option<ExtractedConfig> {
    let test = v.get("test")?;
    let cfg = ExtractedConfig {
        include: json_string_array(test.get("include")),
        exclude: json_string_array(test.get("exclude")),
        ..Default::default()
    };
    if cfg.is_empty() { None } else { Some(cfg) }
}

// ============================================================================
// NX WORKSPACE (nx.json)
// ============================================================================

fn extract_nx(v: &JsonValue) -> Option<ExtractedConfig> {
    // nx.json has implicitDependencies and targetDefaults
    // We extract from workspaceLayout if present
    let layout = v.get("workspaceLayout")?;
    let cfg = ExtractedConfig {
        src: vec![
            layout.get("appsDir").and_then(|v| v.as_str()).unwrap_or("apps").to_string(),
            layout.get("libsDir").and_then(|v| v.as_str()).unwrap_or("libs").to_string(),
        ],
        ..Default::default()
    };
    Some(cfg)
}

// ============================================================================
// TURBO (turbo.json)
// ============================================================================

fn extract_turbo(v: &JsonValue) -> Option<ExtractedConfig> {
    // turbo.json has globalDependencies and pipeline
    let deps = json_string_array(v.get("globalDependencies"));
    if deps.is_empty() {
        None
    } else {
        Some(ExtractedConfig {
            include: deps,
            ..Default::default()
        })
    }
}

// ============================================================================
// VITE CONFIG (vite.config.ts)
// ============================================================================

fn extract_vite(v: &JsonValue) -> Option<ExtractedConfig> {
    let build = v.get("build")?;
    let rollup = build.get("rollupOptions")?;
    let external = json_string_array(rollup.get("external"));
    if external.is_empty() {
        None
    } else {
        Some(ExtractedConfig { exclude: external, ..Default::default() })
    }
}

// ============================================================================
// WEBPACK CONFIG
// ============================================================================

fn extract_webpack(v: &JsonValue) -> Option<ExtractedConfig> {
    let resolve = v.get("resolve")?;
    let modules = json_string_array(resolve.get("modules"));
    if modules.is_empty() {
        None
    } else {
        Some(ExtractedConfig { src: modules, ..Default::default() })
    }
}

// ============================================================================
// GO.MOD / GO.WORK (Go ecosystem)
// ============================================================================

fn extract_go_work(v: &JsonValue) -> Option<ExtractedConfig> {
    // go.work has "use" directives listing module directories
    let uses = json_string_array(v.get("use"));
    if uses.is_empty() {
        None
    } else {
        Some(ExtractedConfig { include: uses, ..Default::default() })
    }
}

// ============================================================================
// COMPOSER.JSON (PHP)
// ============================================================================

fn extract_composer(v: &JsonValue) -> Option<ExtractedConfig> {
    // composer.json has autoload.psr-4 mapping namespaces to directories
    let mut src = Vec::new();
    if let Some(autoload) = v.get("autoload") {
        if let Some(psr4) = autoload.get("psr-4").and_then(|v| v.as_object()) {
            for path in psr4.values() {
                if let Some(s) = path.as_str() {
                    src.push(s.to_string());
                } else if let Some(arr) = path.as_array() {
                    for p in arr {
                        if let Some(s) = p.as_str() {
                            src.push(s.to_string());
                        }
                    }
                }
            }
        }
    }
    if src.is_empty() {
        None
    } else {
        Some(ExtractedConfig { src, ..Default::default() })
    }
}

// ============================================================================
// PHPSTAN.NEON (PHP static analysis)
// ============================================================================
// Note: NEON format is similar to YAML, needs separate parser

// ============================================================================
// RUBOCOP.YML (Ruby)
// ============================================================================
// Note: YAML format, needs separate parser - use serde_yaml if needed

// ============================================================================
// GRADLE BUILD (build.gradle.kts as JSON export)
// ============================================================================

fn extract_gradle(v: &JsonValue) -> Option<ExtractedConfig> {
    // Gradle JSON export might have sourceSets
    let source_sets = v.get("sourceSets")?;
    let main = source_sets.get("main")?;
    let dirs = json_string_array(main.get("srcDirs"));
    if dirs.is_empty() {
        None
    } else {
        Some(ExtractedConfig { src: dirs, ..Default::default() })
    }
}

// ============================================================================
// MAVEN POM.XML (Java) - needs XML parsing
// ============================================================================
// XML files are handled separately via load_xml_config

// ============================================================================
// .NET PROJECT FILES (.csproj, Directory.Build.props) - XML
// ============================================================================
// XML files are handled separately via load_xml_config

// ============================================================================
// GITIGNORE-STYLE FILES
// ============================================================================

/// Parse gitignore-style file (one pattern per line, # comments, ! negation)
fn parse_gitignore_patterns(content: &str) -> Vec<String> {
    content
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            // Convert gitignore patterns to glob patterns
            let pattern = line.trim_start_matches('!'); // Handle negation (we ignore it for now)
            if pattern.starts_with('/') {
                // Anchored to root
                pattern[1..].to_string()
            } else if pattern.ends_with('/') {
                // Directory pattern
                format!("**/{}", pattern.trim_end_matches('/'))
            } else if !pattern.contains('/') {
                // Simple name - match anywhere
                format!("**/{}", pattern)
            } else {
                pattern.to_string()
            }
        })
        .collect()
}

/// Load patterns from a gitignore-style file
fn load_ignore_file(path: &Path) -> Option<ExtractedConfig> {
    let content = std::fs::read_to_string(path).ok()?;
    let patterns = parse_gitignore_patterns(&content);
    if patterns.is_empty() {
        None
    } else {
        Some(ExtractedConfig { exclude: patterns, ..Default::default() })
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn json_string_array(v: Option<&JsonValue>) -> Vec<String> {
    v.and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// Parse TOML into serde_json::Value for uniform handling
fn toml_to_json(content: &str) -> Option<JsonValue> {
    let toml_value: toml::Value = toml::from_str(content).ok()?;
    // Convert toml::Value to serde_json::Value
    serde_json::to_value(toml_value).ok()
}

// ============================================================================
// ADAPTER REGISTRIES
// ============================================================================

/// Adapters for pyproject.toml (checked in order)
const PYPROJECT_ADAPTERS: &[ToolAdapter] = &[
    ToolAdapter {
        name: "ripmap",
        snark: None,
        extract: extract_ripmap,
    },
    ToolAdapter {
        name: "ty",
        snark: None,
        extract: extract_ty,
    },
    ToolAdapter {
        name: "ruff",
        snark: None,
        extract: extract_ruff,
    },
    ToolAdapter {
        name: "pyright",
        snark: Some(|| sparkle_banner("pyright is now banned under international law.", "ty")),
        extract: extract_pyright,
    },
    ToolAdapter {
        name: "mypy",
        snark: None,
        extract: extract_mypy,
    },
    ToolAdapter {
        name: "black",
        snark: None,
        extract: extract_black,
    },
    ToolAdapter {
        name: "isort",
        snark: None,
        extract: extract_isort,
    },
    ToolAdapter {
        name: "pytest",
        snark: None,
        extract: extract_pytest,
    },
];

/// Adapters for package.json
const PACKAGE_JSON_ADAPTERS: &[ToolAdapter] = &[
    ToolAdapter {
        name: "ripmap",
        snark: None,
        extract: extract_pkg_ripmap,
    },
    ToolAdapter {
        name: "eslint",
        snark: None,
        extract: extract_pkg_eslint,
    },
    ToolAdapter {
        name: "prettier",
        snark: None,
        extract: extract_pkg_prettier,
    },
];

/// Config files to check (in priority order)
/// Priority: explicit ripmap config > language-specific tools > monorepo tools > ignore files
const CONFIG_FILES: &[(&str, &[ToolAdapter])] = &[
    // === EXPLICIT RIPMAP CONFIG ===
    ("ripmap.toml", &[]),  // Special case: direct config

    // === PYTHON ECOSYSTEM ===
    ("pyproject.toml", PYPROJECT_ADAPTERS),

    // === JAVASCRIPT/TYPESCRIPT ECOSYSTEM ===
    ("package.json", PACKAGE_JSON_ADAPTERS),
    ("tsconfig.json", &[ToolAdapter { name: "tsconfig", snark: None, extract: extract_tsconfig }]),
    ("jsconfig.json", &[ToolAdapter { name: "jsconfig", snark: None, extract: extract_tsconfig }]),  // Same format as tsconfig
    ("biome.json", &[ToolAdapter { name: "biome", snark: None, extract: extract_biome }]),
    ("biome.jsonc", &[ToolAdapter { name: "biome", snark: None, extract: extract_biome }]),
    ("deno.json", &[ToolAdapter { name: "deno", snark: None, extract: extract_deno }]),
    ("deno.jsonc", &[ToolAdapter { name: "deno", snark: None, extract: extract_deno }]),
    ("jest.config.json", &[ToolAdapter { name: "jest", snark: None, extract: extract_jest }]),

    // === MONOREPO / BUILD TOOLS ===
    ("nx.json", &[ToolAdapter { name: "nx", snark: None, extract: extract_nx }]),
    ("turbo.json", &[ToolAdapter { name: "turbo", snark: None, extract: extract_turbo }]),

    // === RUST ECOSYSTEM ===
    ("Cargo.toml", &[ToolAdapter { name: "cargo", snark: None, extract: extract_cargo }]),

    // === PHP ECOSYSTEM ===
    ("composer.json", &[ToolAdapter { name: "composer", snark: None, extract: extract_composer }]),
];

/// Gitignore-style files to check as fallback (in priority order)
/// These are checked AFTER all structured config files.
/// Note: .dockerignore is intentionally excluded - it has inverted anchoring
/// behavior where simple patterns match root only (opposite of .gitignore).
const IGNORE_FILES: &[&str] = &[
    ".ripmapignore",     // Our own ignore file (gitignore syntax)
    ".eslintignore",     // ESLint (gitignore syntax, legacy - flat config preferred)
    ".prettierignore",   // Prettier (true gitignore syntax)
    ".stylelintignore",  // Stylelint (true gitignore syntax, uses node-ignore)
];

impl Config {
    /// Load configuration from the given directory.
    ///
    /// Checks config files in priority order, walking up the directory tree.
    /// See module docs for supported config files and their sections.
    pub fn load(directory: &Path) -> Self {
        // Try each config file in the current directory first
        if let Some(config) = Self::try_load_from_dir(directory) {
            return config;
        }

        // Walk up the directory tree
        let mut current = directory.to_path_buf();
        while let Some(parent) = current.parent() {
            if let Some(config) = Self::try_load_from_dir(parent) {
                return config;
            }
            current = parent.to_path_buf();
        }

        // Default config
        Self::default()
    }

    /// Try to load config from a single directory, checking all config files.
    fn try_load_from_dir(dir: &Path) -> Option<Self> {
        // First: check structured config files
        for (filename, adapters) in CONFIG_FILES {
            let path = dir.join(filename);
            if !path.exists() {
                continue;
            }

            // Special case: ripmap.toml is direct config, no adapters
            if *filename == "ripmap.toml" {
                if let Some(config) = Self::load_ripmap_toml(&path) {
                    return Some(config);
                }
                continue;
            }

            // Load and parse the config file
            let content = match std::fs::read_to_string(&path) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let json = match Self::parse_config_file(&content, filename) {
                Some(j) => j,
                None => continue,
            };

            // Try each adapter in order
            for adapter in *adapters {
                if let Some(extracted) = (adapter.extract)(&json) {
                    // Fire the snark if this tool has one
                    if let Some(snark) = adapter.snark {
                        snark();
                    }

                    return Some(Self::from_extracted(
                        extracted,
                        format!("{} [{}]", path.display(), adapter.name),
                    ));
                }
            }
        }

        // Fallback: check gitignore-style files
        for filename in IGNORE_FILES {
            let path = dir.join(filename);
            if path.exists() {
                if let Some(extracted) = load_ignore_file(&path) {
                    return Some(Self::from_extracted(
                        extracted,
                        format!("{}", path.display()),
                    ));
                }
            }
        }

        None
    }

    /// Parse config file content based on filename extension
    fn parse_config_file(content: &str, filename: &str) -> Option<JsonValue> {
        if filename.ends_with(".toml") {
            toml_to_json(content)
        } else if filename.ends_with(".json") || filename.ends_with(".jsonc") {
            // Strip comments for jsonc
            let clean = if filename.ends_with(".jsonc") {
                Self::strip_json_comments(content)
            } else {
                content.to_string()
            };
            serde_json::from_str(&clean).ok()
        } else {
            None
        }
    }

    /// Strip // and /* */ comments from JSON (for .jsonc files)
    fn strip_json_comments(content: &str) -> String {
        let mut result = String::new();
        let mut chars = content.chars().peekable();
        let mut in_string = false;
        let mut escape = false;

        while let Some(c) = chars.next() {
            if escape {
                result.push(c);
                escape = false;
                continue;
            }

            if c == '\\' && in_string {
                result.push(c);
                escape = true;
                continue;
            }

            if c == '"' {
                in_string = !in_string;
                result.push(c);
                continue;
            }

            if !in_string && c == '/' {
                if chars.peek() == Some(&'/') {
                    // Line comment - skip to end of line
                    for ch in chars.by_ref() {
                        if ch == '\n' {
                            result.push('\n');
                            break;
                        }
                    }
                    continue;
                } else if chars.peek() == Some(&'*') {
                    // Block comment - skip to */
                    chars.next(); // consume *
                    while let Some(ch) = chars.next() {
                        if ch == '*' && chars.peek() == Some(&'/') {
                            chars.next();
                            break;
                        }
                    }
                    continue;
                }
            }

            result.push(c);
        }
        result
    }

    fn load_ripmap_toml(path: &Path) -> Option<Self> {
        let content = std::fs::read_to_string(path).ok()?;
        let json = toml_to_json(&content)?;

        Some(Self {
            source: Some(path.display().to_string()),
            include: json_string_array(json.get("include")),
            exclude: json_string_array(json.get("exclude")),
            extend_exclude: json_string_array(json.get("extend-exclude")),
            src: json_string_array(json.get("src"))
                .into_iter()
                .map(PathBuf::from)
                .collect(),
        })
    }

    fn from_extracted(extracted: ExtractedConfig, source: String) -> Self {
        Self {
            source: Some(source),
            include: extracted.include,
            exclude: extracted.exclude,
            extend_exclude: extracted.extend_exclude,
            src: extracted.src.into_iter().map(PathBuf::from).collect(),
        }
    }

    /// Get effective exclude patterns (defaults + extend-exclude, or custom exclude).
    pub fn effective_excludes(&self) -> Vec<String> {
        if !self.exclude.is_empty() {
            // Custom exclude replaces defaults
            self.exclude.clone()
        } else {
            // Use defaults + extend-exclude
            let mut patterns: Vec<String> =
                DEFAULT_EXCLUDES.iter().map(|s| s.to_string()).collect();
            patterns.extend(self.extend_exclude.clone());
            patterns
        }
    }

    /// Check if a path matches any include pattern.
    /// Returns true if no include patterns (include all), or if path matches any pattern.
    /// Handles both glob patterns (with *, ?, [) and directory prefixes (e.g., "src").
    pub fn matches_include(&self, path: &Path) -> bool {
        if self.include.is_empty() {
            return true;
        }
        let path_str = path.to_string_lossy();
        self.include
            .iter()
            .any(|pattern| Self::matches_pattern(pattern, &path_str))
    }

    /// Check if a path matches any exclude pattern.
    /// Handles both glob patterns and directory prefixes.
    pub fn matches_exclude(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();
        self.effective_excludes()
            .iter()
            .any(|pattern| Self::matches_pattern(pattern, &path_str))
    }

    /// Match a pattern against a path, handling both globs and directory prefixes.
    fn matches_pattern(pattern: &str, path: &str) -> bool {
        // If pattern contains glob characters, use glob matching
        if pattern.contains('*') || pattern.contains('?') || pattern.contains('[') {
            glob_match::glob_match(pattern, path)
        } else {
            // Treat as directory prefix: "src" matches "src/foo.py", "src/bar/baz.rs"
            // Also handle trailing slash: "src/" same as "src"
            let prefix = pattern.trim_end_matches('/');
            path == prefix || path.starts_with(&format!("{}/", prefix))
        }
    }

    /// Check if a path should be included (matches include AND not exclude).
    pub fn should_include(&self, path: &Path) -> bool {
        self.matches_include(path) && !self.matches_exclude(path)
    }

    /// Format config for verbose display.
    pub fn display_summary(&self) -> String {
        let mut lines = Vec::new();

        if let Some(ref source) = self.source {
            lines.push(format!("   Config: {}", source));
        } else {
            lines.push("   Config: (defaults)".to_string());
        }

        if !self.include.is_empty() {
            lines.push(format!("   Include: {}", self.include.join(", ")));
        }

        let excludes = self.effective_excludes();
        if !excludes.is_empty() {
            // Show first few, then count
            if excludes.len() <= 3 {
                lines.push(format!("   Exclude: {}", excludes.join(", ")));
            } else {
                lines.push(format!(
                    "   Exclude: {}, ... (+{} more)",
                    excludes[..2].join(", "),
                    excludes.len() - 2
                ));
            }
        }

        if !self.src.is_empty() {
            let src_strs: Vec<_> = self.src.iter().map(|p| p.display().to_string()).collect();
            lines.push(format!("   Src roots: {}", src_strs.join(", ")));
        }

        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_excludes() {
        let config = Config::default();
        assert!(config.matches_exclude(Path::new("foo/node_modules/bar.js")));
        assert!(config.matches_exclude(Path::new("project/.git/config")));
        assert!(config.matches_exclude(Path::new("src/__pycache__/mod.pyc")));
        assert!(!config.matches_exclude(Path::new("src/main.py")));
    }

    #[test]
    fn test_include_patterns() {
        let config = Config {
            include: vec!["src/**".to_string(), "lib/**".to_string()],
            ..Default::default()
        };
        assert!(config.matches_include(Path::new("src/main.py")));
        assert!(config.matches_include(Path::new("lib/utils.py")));
        assert!(!config.matches_include(Path::new("tests/test_main.py")));
    }

    #[test]
    fn test_extend_exclude() {
        let config = Config {
            extend_exclude: vec!["**/generated/**".to_string()],
            ..Default::default()
        };
        // Should still have defaults
        assert!(config.matches_exclude(Path::new("node_modules/foo.js")));
        // Plus the extension
        assert!(config.matches_exclude(Path::new("src/generated/schema.py")));
    }

    #[test]
    fn test_directory_prefix_patterns() {
        // Test include with directory prefix (no glob chars)
        let config = Config {
            include: vec!["src".to_string()],
            ..Default::default()
        };
        assert!(config.matches_include(Path::new("src/main.py")));
        assert!(config.matches_include(Path::new("src/lib/utils.py")));
        assert!(!config.matches_include(Path::new("tests/test_main.py")));
        assert!(!config.matches_include(Path::new("srcfoo/bar.py"))); // "srcfoo" != "src/"

        // Test exclude with directory prefix
        let config = Config {
            exclude: vec!["vendor".to_string(), "src/gui/old/".to_string()],
            ..Default::default()
        };
        assert!(config.matches_exclude(Path::new("vendor/lib.py")));
        assert!(config.matches_exclude(Path::new("src/gui/old/widget.py")));
        assert!(!config.matches_exclude(Path::new("src/gui/new/widget.py")));
    }
}
