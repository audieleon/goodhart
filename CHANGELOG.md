# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- `intrinsic_dominance` rule: flags when accumulated intrinsic reward
  exceeds the terminal goal (Verified, LEAN proof).
- `discount_horizon_mismatch` rule: episode exceeds discount horizon
  with sparse rewards (Verified, LEAN proof).
- `negative_only_reward` rule: all components non-positive, no learning
  signal (Verified, LEAN proof).
- `reward_delay_horizon` rule: terminal goal discounted below noise
  floor (Verified, LEAN proof).
- 9 new examples (6 intrinsic dominance + 3 new rules).
- LEAN CI uses leanprover/lean-action (precompiled Mathlib).
- LEAN CI runs on PRs, not just push to main.
- Branch protection on main (PR + CI required).
- Total theorems increased from 92 to 103.
- Proved rules increased from 17 to 24 (13 verified, 7 grounded, 4 motivated).
- Examples increased from 57 to 66.
- Upgraded `shaping_loop_exploit` and `shaping_not_potential_based` from
  Motivated to Grounded.
- Added FormalBasis to `critic_lr_ratio`, `batch_size_interaction`,
  `parallelism_effect`.

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
