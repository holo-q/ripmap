## API_contract

Input:
- NDCG, episode number, trajectory history
- 17 hyperparameters
- Up to 5 ranking failures with context

Output JSON:
- strategy_capsule: intent encoding
- diagnosis: analysis summary
- param_interactions: discovered couplings
- proposed_changes: {param: [direction, magnitude, rationale]}
- structural_insights: beyond-tuning observations
- confidence: 0.0-1.0

## Output_schema

Reason freely first, then output JSON in this EXACT format:

REASONING:
[Your analysis here - be specific about trajectory patterns and causal hypotheses]

JSON:
{
  "strategy_capsule": "1-2 sentence description of your intent (exploring, testing counterfactual, reverting, etc.)",
  "diagnosis": "summary of your analysis",
  "param_interactions": ["any interactions discovered"],
  "proposed_changes": {
    "param_name": ["increase|decrease", "small|medium|large", "rationale"]
  },
  "structural_insights": ["insights beyond parameter tuning"],
  "confidence": 0.7
}
