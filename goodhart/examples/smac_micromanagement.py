"""Example: SMAC StarCraft Micromanagement (Samvelyan et al. 2019).

Multi-agent combat with asymmetric reward scaling:
  enemy killed = +10.0, ally killed = -5.0 (0.5x negative scale)

This means killing enemies is valued 2x over protecting allies,
incentivizing hyper-aggressive strategies that sacrifice units.

The passive damage_received penalty creates a death-beats-survival
incentive: a dead ally stops accumulating damage penalty.

Source: Samvelyan et al. 2019, "The StarCraft Multi-Agent Challenge"
  (AAMAS); reward from Section 3.2
"""

from goodhart.presets import PRESETS
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    model, config = PRESETS["smac"]
    engine = TrainingAnalysisEngine().add_all_rules()

    print("SMAC StarCraft Micromanagement")
    print("=" * 50)
    print("Source: Samvelyan et al. 2019 (NeurIPS), oxwhirl/smac")
    print()
    print("Reward components:")
    for s in model.reward_sources:
        print(f"  {s.name:20s} {s.value:+6.1f} ({s.reward_type.value})")
    print()
    print("Note: ally_killed at -5.0 vs enemy_killed at +10.0")
    print("means killing is 2x more rewarding than protecting.")
    print()

    engine.print_report(model, config)


if __name__ == "__main__":
    run_example()
