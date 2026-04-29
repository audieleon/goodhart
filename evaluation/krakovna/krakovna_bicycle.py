"""Krakovna entry: Bicycle riding in circles.

Source: Randlov & Alstrom 1998
Failure: Agent rides in tight circles instead of reaching the goal.
Mechanism: Progress reward (distance decrease) without regression penalty.
  The agent can repeatedly harvest shaping reward by orbiting the goal,
  decreasing distance on approach and increasing on retreat, with only
  the decrease rewarded.
Domain: Control / navigation

Also in goodhart/examples/bicycle_circles.py
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Bicycle riding in circles (Randlov & Alstrom 1998)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Bicycle (Randlov & Alstrom 1998)",
        max_steps=1000,
        gamma=0.99,
        n_actions=4,
        action_type="continuous",
    )

    # Sparse terminal goal: reach the destination
    model.add_reward_source(RewardSource(
        name="reach_goal",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        discovery_probability=0.05,
        intentional=True,
    ))

    # Distance-decrease shaping -- the trap
    # Only rewards decrease, no penalty for increase, so orbiting works.
    model.add_reward_source(RewardSource(
        name="distance_decrease",
        reward_type=RewardType.SHAPING,
        value=0.1,
        can_loop=True,
        loop_period=4,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
