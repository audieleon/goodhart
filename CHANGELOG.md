# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

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
- 92 LEAN 4 theorems across 10 files, zero sorry
- 17 rules linked to formal proofs via ProofStrength hierarchy
  (9 verified, 3 grounded, 5 motivated)
- Ng 1999 Theorem 1 formalization (sufficiency, necessity, general policy
  version, undiscounted extension)
- Skalse 2022 Theorems 1-3 formalization (impossibility, existence,
  simplification, |Pi|=2 edge case)
- MDP infrastructure (FiniteMDP, Bellman contraction via Banach fixed point)
- 23 presets from published papers
- 57 cookbook examples spanning 40+ papers from 1983-2025
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
