"""Example: Robosuite staged rewards (Zhu et al. 2020, CoRL).

Pick-and-place with 4 prerequisite-gated stages:
  grasp -> lift -> hover -> place

Each stage only activates after the previous one succeeds,
creating learning plateaus with zero gradient signal for
later stages until earlier ones are achieved.

Demonstrates the staged_reward_plateau rule — the first
example using prerequisite-gated reward sources.

Source: Zhu et al. 2020, "robosuite: A Modular Simulation Framework
  and Benchmark for Robot Learning" (CoRL)
"""

from goodhart.presets import PRESETS
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    model, config = PRESETS["robosuite-pick-place"]
    engine = TrainingAnalysisEngine().add_all_rules()

    print("Robosuite Pick-and-Place — Staged Rewards")
    print("=" * 50)
    print("Source: Zhu et al. 2020 (CoRL), robosuite library")
    print()
    print("Reward chain:")
    for s in model.reward_sources:
        prereq = f" (requires: {s.prerequisite})" if s.prerequisite else ""
        print(f"  {s.name:10s} {s.value:+.2f}  p(discover)={s.discovery_probability:.2f}{prereq}")
    print()

    engine.print_report(model, config)

    print()
    print("The staged structure means:")
    print("  - Agent gets ZERO signal for 'lift' until 'grasp' succeeds")
    print("  - Agent gets ZERO signal for 'hover' until 'lift' succeeds")
    print("  - A 4-stage chain with low per-stage discovery creates")
    print("    compounding plateaus")
    print()
    print("Fix: Add distance-based shaping between stages,")
    print("or use curriculum learning to unlock stages gradually.")


if __name__ == "__main__":
    run_example()
