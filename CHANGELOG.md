# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- **Reward Failure Dataset**: 212 structured encodings from 133 published
  papers across 18 domains (manipulation, game AI, locomotion, driving,
  control, multi-agent, navigation, energy, finance, healthcare, safety,
  RLHF, chip design, fusion, industrial, and more).
- `--fields` and `--field NAME`: plain-language field reference in CLI.
- `--strict` (`-s`): treat warnings as errors (exit code 1).
- `--ignore RULES`: suppress specific rules by name (comma-separated).
- `--format compact`: one-line-per-finding output for grep and log scanning.
- `--doctor --json`: machine-readable fix suggestions.
- `--config -`: read YAML/JSON config from stdin.
- `-j` short flag for `--json`, `-s` for `--strict`.
- Plain-language docstrings on all RewardSource and EnvironmentModel fields.
- 4 new rules: `intrinsic_dominance`, `discount_horizon_mismatch`,
  `negative_only_reward`, `reward_delay_horizon` (all Verified, LEAN).
- 146 evaluation entries encoded independently of rule development.
- Total theorems increased from 92 to 105.
- Proved rules increased from 17 to 24.
- LLM baseline comparison: 4 frontier LLMs tested (May 2026).
- Datasheet for the Reward Failure Dataset (Gebru et al. format).

### Removed
- Presets system (`goodhart/presets.py`, `--preset` CLI flag).
  Examples are now self-contained with inline model definitions.

### Changed
- Examples no longer depend on presets; each builds its own
  EnvironmentModel with full METADATA and paper provenance.
- Quick-check (`--goal/--penalty/--steps`) now routes through
  `_output_analysis` for consistent `--strict`/`--ignore` support.

### Fixed
- `.gitignore` no longer ignores `evaluation/sources/papers/`.
- 4 tool bugs found during evaluation encoding (multiplicative
  modifiers, tracking-controller severity, advisory scope,
  state-dependent penalty handling).

## [0.1.0] - 2026-04-26

Initial release.

### Added
- 40 composable analysis rules across four categories:
  - 15 reward structure rules (penalty traps, idle exploits, shaping loops,
    proxy hackability, staged plateaus, respawning exploits, dominance imbalance,
    exponential saturation)
  - 13 training hyperparameter rules (PPO, DQN, SAC, DDPG, TD3, IMPALA) (learning rate, entropy, clipping,
    batch size, memory capacity)
  - 4 architecture rules (embedding capacity, routing floor, recurrence,
    actor count)
  - 8 blind-spot advisories (physics exploits, misgeneralization, credit
    assignment, constrained RL, non-stationarity, learned rewards, missing
    constraints, aggregation traps)
- `@reward_function` decorator for annotating Python reward functions
- `--verbose` mode with educational explanations for each finding
- `--explain <rule>` for standalone rule deep-dives with examples and references
- Contradiction detection across rule recommendations
- 103 LEAN 4 theorems across 10 files, zero sorry
- 24 rules linked to formal proofs via ProofStrength hierarchy
  (13 verified, 7 grounded, 4 motivated)
- Ng 1999 Theorem 1 formalization (sufficiency, necessity, general policy
  version, undiscounted extension)
- Skalse 2022 Theorems 1-3 formalization (impossibility, existence,
  simplification, |Pi|=2 edge case)
- MDP infrastructure (FiniteMDP, Bellman contraction via Banach fixed point)
- 23 presets from published papers
- 66 cookbook examples spanning 40+ papers from 1983-2025
- Explanations database with learn_more context for all 40 rules
- CLI with preset, config, check, rules, examples, explain, about, doctor,
  detect, json, quiet, verbose modes
- Python API (check, analyze, reward_function, analyze_function)
- MCP server for AI assistant integration
- YAML, JSON, TOML config file support
- Isaac Gym and highway-env native config adapters
- Gymnasium auto-detect from rollout episodes
- 276 tests

[Unreleased]: https://github.com/audieleon/goodhart/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/audieleon/goodhart/releases/tag/v0.1.0
