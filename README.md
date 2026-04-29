# goodhart

[![CI](https://github.com/audieleon/goodhart/actions/workflows/ci.yml/badge.svg)](https://github.com/audieleon/goodhart/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Paper:** [Catching Goodhart's Law Before Training: Static Reward Analysis with Formal Guarantees](https://github.com/audieleon/goodhart/blob/main/CITATION.cff) (Sheridan, 2026)

> "When a measure becomes a target, it ceases to be a good measure."
> -- Charles Goodhart (1975), generalized by Marilyn Strathern (1997)

**Catch reward traps before training.** Goodhart runs 44 composable analysis rules on your RL reward configuration and reports degenerate equilibria, perverse incentives, and exploitable reward structures -- before you spend compute. 24 rules are backed by machine-verified LEAN 4 proofs (zero sorry), including formalizations of Ng 1999 and Skalse 2022.

## Installation

```bash
pip install goodhart

# Or install from source
pip install git+https://github.com/audieleon/goodhart.git

# Optional: visualization and Gymnasium auto-detection
pip install goodhart[all]
```

## Quick Start

```bash
# Check a sparse reward config
goodhart --goal 1.0 --penalty -0.01 --steps 500
# -> CRITICAL: death beats survival by 9.6x

# Try a preset from a published paper
goodhart --preset coast-runners
# -> CRITICAL: loop EV (+800) beats goal (+100)

# List all available presets
goodhart --preset

# Interactive mode (asks questions)
goodhart
```

## Usage

### CLI

```bash
# Quick check with training params
goodhart --goal 1.0 --penalty -0.001 --steps 400 --gamma 0.999 \
  --actors 64 --budget 10000000 --lr 1e-4 --specialists 3 --floor 0.10

# From a config file (YAML, JSON, or TOML)
goodhart --config my_experiment.yaml

# From an annotated Python reward function
goodhart --check my_env.py:compute_reward

# With educational explanations
goodhart --preset humanoid --verbose

# Deep-dive on a specific rule
goodhart --explain idle_exploit

# Diagnose and suggest fixes
goodhart --doctor --goal 1.0 --penalty -0.01 --steps 500

# CI integration (exit code 1 on critical issues)
goodhart --quiet --exit-on-critical --config experiment.yaml
```

### Python API

```python
# Quick check (prints report, returns bool)
from goodhart import check
passed = check(goal=1.0, penalty=-0.01, max_steps=500)  # False if criticals

# Programmatic analysis (no printing, returns typed Result)
from goodhart import analyze
result = analyze(goal=1.0, penalty=-0.01, max_steps=500, gamma=0.999)
print(result.passed)       # True/False
print(result.criticals)    # list of Verdict objects
print(result.to_dict())    # JSON-serializable dict
```

### Decorator (annotate a Python reward function)

```python
from goodhart import reward_function, RewardSource, RewardType

ALIVE_BONUS = 1.0
VELOCITY_SCALE = 0.5
CTRL_COST = -0.001

@reward_function(
    max_steps=1000, gamma=0.99, n_actions=8, action_type="continuous",
    sources=[
        RewardSource("alive", RewardType.PER_STEP, ALIVE_BONUS,
                     requires_action=False, intentional=True),
        RewardSource("velocity", RewardType.PER_STEP, VELOCITY_SCALE,
                     intentional=True, state_dependent=True),
        RewardSource("ctrl", RewardType.PER_STEP, CTRL_COST,
                     requires_action=True),
    ],
)
def compute_reward(obs, action, info):
    return ALIVE_BONUS + obs["velocity"] * VELOCITY_SCALE + CTRL_COST * sum(a**2 for a in action)

# The function works normally AND carries analysis metadata
compute_reward(obs, action, info)        # returns reward
compute_reward.goodhart_check()          # prints full report
assert compute_reward.goodhart_passed()  # CI gate
```

Constants are defined once and shared between the decorator and the function body -- no duplication, no drift.

### AI Assistant (Claude Code, Cursor)

If you use an AI coding assistant, goodhart can run automatically when you discuss reward design. Add to your MCP config (one-time setup):

```json
{
  "mcpServers": {
    "goodhart": {
      "command": "python",
      "args": ["-m", "goodhart.mcp_server"]
    }
  }
}
```

**Claude Code:** add to `~/.claude/settings.json`
**Cursor:** add to `.cursor/mcp.json`

Then just describe your reward in conversation — the assistant calls `goodhart_check` automatically and explains the findings. 8 tools available: check, doctor, explain rules, browse presets and examples.

### YAML Configuration

```yaml
# my_experiment.yaml
environment:
  name: "MiniHack-Navigation"
  max_steps: 500
  gamma: 0.999
  reward_sources:
    - name: goal
      type: terminal
      value: 1.0
      discovery_probability: 0.05
    - name: step penalty
      type: per_step
      value: -0.001

training:
  algorithm: APPO
  lr: 0.0002
  entropy_coeff: 0.0001
  num_envs: 256
  total_steps: 10000000
```

## Presets

23 presets from published papers, with hyperparameters sourced from the original publications:

```bash
goodhart --preset              # list all presets
goodhart --preset coast-runners  # run CoastRunners (loop exploit)
goodhart --preset humanoid       # run Humanoid (idle exploit)
goodhart --preset cartpole       # run CartPole (clean pass)
```

## Rules

44 composable rules in four categories:

```bash
goodhart --rules      # list all with descriptions
goodhart --explain X  # deep-dive on rule X
```

- **19 reward rules**: penalty dominance, death incentive, idle exploit, exploration threshold, respawning exploit, death reset, shaping loops, shaping safety (Ng 1999), proxy hackability (Skalse 2022), intrinsic sufficiency, budget sufficiency, compound traps, staged plateaus, reward dominance, exponential saturation, intrinsic dominance, discount horizon mismatch, negative-only reward, reward delay horizon
- **13 training rules**: learning rate regime (all algorithms), critic LR ratio, entropy regime, clip fraction risk (PPO), expert collapse, batch size interaction, parallelism effect, memory capacity, replay buffer ratio (off-policy), target network update (DQN), epsilon schedule (DQN), soft update rate (SAC/DDPG/TD3), SAC alpha
- **4 architecture rules**: embedding capacity, routing floor necessity, recurrence type, actor count effect
- **8 blind-spot advisories**: pattern-based hints about failure modes static analysis cannot detect (physics exploits, goal misgeneralization, credit assignment depth, constrained RL, non-stationarity, learned rewards, missing constraints, aggregation traps)

Reward structure rules (19) are algorithm-agnostic — they analyze the MDP reward regardless of training algorithm. Training rules (13) cover PPO, APPO, DQN, SAC, DDPG, TD3, IMPALA, and A2C with algorithm-specific thresholds and checks.

## What it catches vs. what it can't

**Catches** (from configuration alone):
- Degenerate equilibria (standing still, dying fast)
- Respawning reward loops (CoastRunners, YouTube watch time)
- Death-as-reset exploits (Road Runner level replay)
- Shaping reward cycles vs. potential-based shaping (Ng 1999)
- Reward deserts (no gradient signal, e.g., Mountain Car)
- Proxy reward hackability (Skalse 2022)
- Expert collapse, entropy issues, budget insufficiency

**Cannot catch** (emits advisory hints when config patterns match):
- Physics engine exploits (box surfing, leg hooking)
- Goal misgeneralization (CoinRun "go right")
- Learned reward model gaming (RLHF overoptimization)
- Missing reward terms (tokamak coil balance)
- Non-stationarity in self-play
- Episode-level aggregation traps (Sharpe ratio)

## Examples

66 cookbook examples spanning 40+ published papers from 1983-2025:

```bash
goodhart --examples              # list all
goodhart --example coast_runners # run one
```

Examples include documented failures (CoastRunners, Humanoid, Mountain Car), positive design patterns (Pendulum, CartPole, Breakout), industrial applications (YouTube, data center cooling, tokamak plasma, sepsis treatment), and honest limitation cases showing what static analysis cannot detect.

## Formal Proofs

24 rules link to machine-verified LEAN 4 theorems (103 theorems, zero sorry). Each link has a strength level:

- **VERIFIED** (13 rules): The Python check is a direct instance of the theorem.
- **GROUNDED** (7 rules): The theorem proves the core. Python extends with discounting and thresholds.
- **MOTIVATED** (4 rules): The theorem proves WHY the issue matters. Python checks a structural heuristic.

Key formalizations:

- **Ng 1999 Theorem 1**: Potential-based reward shaping preserves V* (sufficiency, necessity, general policy version, undiscounted extension). Full MDP with Bellman contraction via Banach fixed point theorem.
- **Skalse 2022 Theorems 1-3**: Hackability impossibility on open sets, existence of unhackable pairs, simplification characterization. Includes a machine-verified proof that Theorem 2's non-trivial witness construction requires |Pi| >= 3; for |Pi| = 2 only trivial witnesses exist (documented edge case, see proofs/GoodhartProofs/Skalse.lean).

```bash
cd proofs
lake build  # requires LEAN 4 + Mathlib
# Should complete with zero sorry, zero errors
```

## Auto-Detection

Automatically detect reward structure from a Gymnasium environment:

```bash
pip install goodhart[detect]
goodhart --detect CartPole-v1
goodhart --detect MountainCar-v0
```

## License

Apache 2.0
