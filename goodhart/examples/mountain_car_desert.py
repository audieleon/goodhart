"""Example: Mountain Car — the canonical reward desert.

Every step costs -1. The goal (hilltop) gives 0. There is no
positive reward anywhere. The agent gets -1 whether it's building
momentum or standing still. Learning is extremely slow because
only random goal discovery provides any gradient.

Source: Moore 1990, "Efficient Memory-based Learning for Robot
  Control." Also Sutton & Barto 2018, Section 10.1.
Tool should catch: negative_only_reward (CRITICAL)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Mountain Car — reward desert (all negative)")
    print("=" * 70)
    print()
    print("Source: Moore 1990, Sutton & Barto 2018")
    print("-1 per step, 0 at goal. No positive signal anywhere.")
    print()

    model = EnvironmentModel(
        name="Mountain Car (reward desert)",
        max_steps=200,
        gamma=1.0,
        n_actions=3,
        action_type="discrete",
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="step_penalty",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="goal",
        reward_type=RewardType.TERMINAL,
        value=0.0,
        requires_action=True,
        intentional=True,
        discovery_probability=0.01,
    ))

    engine.print_report(model)

    print()
    print("The agent earns -1 per step regardless of behavior.")
    print("No positive reward exists. The optimal strategy is")
    print("whichever accumulates the least total penalty.")


if __name__ == "__main__":
    run_example()
