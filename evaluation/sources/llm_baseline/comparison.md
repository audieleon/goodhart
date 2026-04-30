# Comparison: goodhart vs Claude Sonnet 4.6

## Results on 23 presets

| Preset | goodhart | Claude | Known Issue? | Notes |
|--------|----------|--------|-------------|-------|
| anymal | FAIL | PASS | Yes (idle) | goodhart catches idle exploit; Claude misses |
| atari | WARN | WARN | No | Agree |
| bipedal-walker | WARN | WARN | No | Agree |
| cartpole | PASS | PASS | No | Agree |
| coast-runners | FAIL | FAIL | Yes (loop) | Both catch turbo loop |
| coinrun | WARN | WARN | No | Agree |
| dense-survival | WARN | PASS | No | goodhart warns; Claude clean |
| football | WARN | PASS | No | goodhart warns; Claude clean |
| hand-manipulation | FAIL | WARN | Yes (explore) | goodhart CRITICAL; Claude only WARN |
| highway-env | WARN | FAIL | No | Claude overcalls |
| humanoid | FAIL | FAIL | Yes (idle) | Both catch it |
| legged-gym | FAIL | WARN | Yes (idle) | goodhart catches idle; Claude only warns |
| lunar-lander | FAIL | PASS | Debatable | goodhart overcalls |
| metadrive | WARN | WARN | No | Agree |
| minihack-nav | FAIL | PASS | Debatable | goodhart CRITICAL; Claude clean |
| minihack-skill | FAIL | WARN | Yes (penalty) | goodhart catches penalty dominance |
| mountain-car | FAIL | FAIL | Yes (desert) | Both catch it |
| mujoco-locomotion | WARN | PASS | No | goodhart warns; Claude clean |
| mujoco-manipulation | WARN | PASS | No | goodhart warns; Claude clean |
| robosuite | WARN | WARN | No | Agree |
| smac | FAIL | PASS | Yes (death) | goodhart catches death_beats_survival |
| sparse-goal | WARN | WARN | No | Agree |
| taxi | FAIL | FAIL | Yes (penalty) | Both catch it |

## Summary on 11 presets with known structural issues

| Metric | goodhart | Claude |
|--------|----------|--------|
| True positives | 10 | 5 |
| False negatives | 1 (debatable) | 6 |
| Detection rate | 91% | 45% |

## What Claude misses

Claude fails to detect traps requiring expected-value computation:
- **Idle exploits** with moderate ratios (ANYmal 0.2:1, Legged Gym 1:1.5)
- **Death incentives** in multi-agent settings (SMAC)
- **Penalty dominance** with small penalties (MiniHack-skill)
- **Exploration thresholds** for sparse goals (Hand manipulation)

## What Claude catches that goodhart doesn't

- highway-env conservative-driving incentive (goodhart only warns, Claude calls FAIL)
- Qualitative reasoning about risk-aversion dynamics
