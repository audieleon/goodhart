# Roadmap

## Current (v0.1.0)

40 composable rules, 23 presets, 57 examples, 92 LEAN 4 theorems (zero sorry), `@reward_function` decorator, CLI, Python API, MCP server.

## Next

### Web interface
Zero-install trial at goodhart.dev. Paste a YAML config or use a preset, get instant analysis with ProofStrength badges. Shareable URLs for team review.

### LLM reward generation filter
Pre-screen LLM-generated reward candidates (Eureka, CARD) in milliseconds. Reject structurally doomed candidates before training. Highest-leverage use case for the tool.

### Training framework callbacks
Lightweight callbacks for SB3, CleanRL, and Sample Factory that run goodhart before training starts. Reads training config from the framework, reward structure from `@reward_function` decorator or config file.

### Expanded formal proofs
- Prove the aggregation idle trap for general ratio objectives
- Strengthen remaining GROUNDED proofs (exploration_threshold, budget_sufficiency, reward_dominance_imbalance)
- Target: 20+ rules with formal proofs (currently 17, of which 9 VERIFIED)

## Quality constraints (non-negotiable)

1. Every claim is verifiable. LEAN proofs compile. Preset tests pass.
2. ProofStrength is always honest. VERIFIED/GROUNDED/MOTIVATED, never inflated.
3. Conservative by default. False positives are annoying; false negatives are dangerous.
4. The tool says what it can't do. Blind-spot advisories, not silent omissions.
