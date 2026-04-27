"""Example: EME Exploration — held-out evaluation.

Held-out paper not used during tool development. Tests whether
the tool correctly passes a well-calibrated intrinsic motivation
setup where the bonus is small relative to the task reward.

Source: EME Exploration (NeurIPS 2024) — intrinsic motivation
  for hard exploration problems

Expected result: PASS (clean, or INFO-level notes only).
  The intrinsic bonus is well-calibrated relative to the
  task reward (0.1 vs 1.0).
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("EME Exploration — held-out evaluation")
    print("=" * 70)
    print()
    print("Source: EME Exploration, NeurIPS 2024")
    print("Intrinsic bonus dominance test: 0.1 vs 1.0 task reward.")
    print("Well-calibrated — tool should note at INFO level at most.")
    print()

    model = EnvironmentModel(
        name="EME Exploration (Held-Out)",
        max_steps=1000,
        gamma=0.99,
        n_actions=18,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="task_reward",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="intrinsic_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.1,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("The intrinsic bonus (0.1) is 10% of the task reward (1.0),")
    print("which is well within safe bounds. The tool should pass clean")
    print("or note the intrinsic bonus at INFO level without flagging")
    print("any structural problems.")


if __name__ == "__main__":
    run_example()
