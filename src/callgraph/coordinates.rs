//! Trainable coordinates for call graph resolution.
//!
//! Following the Dissolved Decision Trees philosophy (docs/8):
//! - No categorical modes, only continuous coordinates
//! - Every magic number becomes a trainable parameter
//! - L1 can tune these via `{param: [direction, magnitude, rationale]}`
//!
//! # Coordinate Categories
//!
//! - **Strategy Weights**: Relative trust in each resolution strategy
//! - **Acceptance Physics**: How candidates pass through the gate
//! - **Blending Dynamics**: How multiple strategies combine

/// Trainable coordinates for strategy weighting and candidate selection.
///
/// These coordinates replace the hard-coded confidence values and boolean toggles
/// in the resolver. L1 can tune them to shift between regimes (14% heuristic → 80% LSP).
#[derive(Debug, Clone)]
pub struct StrategyCoordinates {
    // ==========================================================================
    // Strategy Weights (relative trust in each signal)
    // ==========================================================================

    /// Weight multiplier for same-file strategy [0.5, 1.5]
    /// Higher = trust same-file matches more
    /// In 80% LSP regime, may drop to 0.6 to avoid shadowing bugs
    pub weight_same_file: f64,

    /// Weight multiplier for type-hint strategy [0.5, 1.5]
    /// In LSP regime, might increase since LSP confirms type info
    pub weight_type_hint: f64,

    /// Weight multiplier for import strategy [0.5, 1.5]
    pub weight_import: f64,

    /// Weight multiplier for name-match strategy [0.1, 1.0]
    /// Toxic noise in 80% regime - L1 should suppress to ~0.2
    pub weight_name_match: f64,

    /// Weight multiplier for LSP-resolved candidates [0.8, 2.0]
    /// When LSP provides ground truth, this should dominate
    pub weight_lsp: f64,

    // ==========================================================================
    // Acceptance Gate (sigmoid replaces threshold cliff)
    // ==========================================================================

    /// Sigmoid center point for acceptance gate [0.2, 0.6]
    /// Candidates near this confidence get 50% acceptance probability
    /// Replaces hard min_confidence threshold
    pub acceptance_bias: f64,

    /// Sigmoid slope (steepness) [2.0, 20.0]
    /// Higher = sharper cutoff (approaches hard threshold)
    /// Lower = gentler transition (more uncertainty tolerance)
    pub acceptance_slope: f64,

    // ==========================================================================
    // Selection Dynamics (softmax replaces argmax)
    // ==========================================================================

    /// Temperature for candidate selection softmax [0.01, 2.0]
    /// Low = winner-take-all (approaches argmax)
    /// High = more exploration, ensemble behavior
    pub selection_temperature: f64,

    /// Evidence accumulation mode [0.0, 1.0]
    /// 0.0 = take best candidate only
    /// 1.0 = accumulate evidence from all candidates
    pub evidence_accumulation: f64,

    // ==========================================================================
    // Proximity Dynamics
    // ==========================================================================

    /// Boost for same-directory matches [0.0, 0.3]
    /// Replaces hard-coded proximity_boost: 0.1
    pub proximity_boost: f64,
}

impl Default for StrategyCoordinates {
    fn default() -> Self {
        // Conservative defaults that approximate current behavior
        // L1 will tune these toward optimal regime
        Self {
            // Strategy weights start at "neutral" (1.0 = no change)
            weight_same_file: 1.0,
            weight_type_hint: 1.0,
            weight_import: 1.0,
            weight_name_match: 1.0,
            weight_lsp: 1.2,  // Slight LSP preference by default

            // Acceptance gate approximates old min_confidence: 0.3
            acceptance_bias: 0.3,
            acceptance_slope: 10.0,  // Fairly sharp by default

            // Selection starts conservative (near argmax)
            selection_temperature: 0.1,
            evidence_accumulation: 0.0,

            // Proximity boost matches old default
            proximity_boost: 0.1,
        }
    }
}

impl StrategyCoordinates {
    /// Compute acceptance probability for a candidate.
    /// Sigmoid: 1 / (1 + exp(-slope * (confidence - bias)))
    pub fn acceptance_probability(&self, raw_confidence: f64) -> f64 {
        let x = self.acceptance_slope * (raw_confidence - self.acceptance_bias);
        1.0 / (1.0 + (-x).exp())
    }

    /// Apply weight multiplier to a strategy's raw confidence.
    pub fn weighted_confidence(&self, strategy: &str, raw: f64) -> f64 {
        let weight = match strategy {
            "same_file" => self.weight_same_file,
            "type_hint" => self.weight_type_hint,
            "import" => self.weight_import,
            "name_match" => self.weight_name_match,
            "lsp" => self.weight_lsp,
            _ => 1.0,
        };
        raw * weight
    }

    /// Validate coordinates are within expected ranges.
    pub fn validate(&self) -> Result<(), String> {
        let checks = [
            ("weight_same_file", self.weight_same_file, 0.1, 2.0),
            ("weight_type_hint", self.weight_type_hint, 0.1, 2.0),
            ("weight_import", self.weight_import, 0.1, 2.0),
            ("weight_name_match", self.weight_name_match, 0.0, 1.5),
            ("weight_lsp", self.weight_lsp, 0.5, 3.0),
            ("acceptance_bias", self.acceptance_bias, 0.0, 1.0),
            ("acceptance_slope", self.acceptance_slope, 0.5, 50.0),
            ("selection_temperature", self.selection_temperature, 0.001, 5.0),
            ("evidence_accumulation", self.evidence_accumulation, 0.0, 1.0),
            ("proximity_boost", self.proximity_boost, 0.0, 0.5),
        ];

        for (name, value, min, max) in checks {
            if value < min || value > max {
                return Err(format!(
                    "{} = {} is outside valid range [{}, {}]",
                    name, value, min, max
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_coordinates_valid() {
        let coords = StrategyCoordinates::default();
        assert!(coords.validate().is_ok());
    }

    #[test]
    fn test_acceptance_sigmoid() {
        let coords = StrategyCoordinates::default();

        // At bias point, probability should be ~0.5
        let at_bias = coords.acceptance_probability(0.3);
        assert!((at_bias - 0.5).abs() < 0.01);

        // High confidence should be accepted
        let high = coords.acceptance_probability(0.9);
        assert!(high > 0.99);

        // Low confidence should be rejected
        let low = coords.acceptance_probability(0.1);
        assert!(low < 0.2);
    }

    #[test]
    fn test_weighted_confidence() {
        let mut coords = StrategyCoordinates::default();
        coords.weight_name_match = 0.5;  // Suppress name matching

        let raw = 0.6;
        let weighted = coords.weighted_confidence("name_match", raw);
        assert_eq!(weighted, 0.3);
    }

    #[test]
    fn test_validation_bounds() {
        let mut coords = StrategyCoordinates::default();
        coords.weight_same_file = 5.0;  // Out of bounds
        assert!(coords.validate().is_err());
    }
}
