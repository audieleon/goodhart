"""Example: Eureka Shadow Hand Bottle Cap — GPT-4 generated reward.

Eureka uses GPT-4 to write reward functions for Isaac Gym tasks.
The Bottle Cap reward has a passive hand-distance component: hands
start near the bottle/cap, so exp(-dist/50) is near 1.0 at idle.
The cap-goal distance reward requires action but is not temperature-
scaled, making it linearly negative and hard to optimize.

Source: Ma et al. 2024 (ICLR), Eureka project — GPT-4 generated reward
Tool should catch: passive hand_distance_transformed_reward creates
  idle floor (critical), raw negative cap_goal_distance_reward
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Bottle Cap — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Shadow Hand Bottle Cap.")
    print("Passive hand distance reward creates idle exploit.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Bottle Cap",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="hand_distance_transformed_reward",
        reward_type=RewardType.PER_STEP,
        value=0.5,  # active: hands move, exp decays
        value_range=(0.0, 1.0),  # idle: hands start near objects, ~1.0
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="cap_goal_distance_reward",
        reward_type=RewardType.PER_STEP,
        value=-0.5,  # negative raw distance, no exp transform
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("The hand_distance_transformed_reward is passive: hands start")
    print("near the bottle/cap, so exp(-dist/50) is near 1.0 at idle.")
    print("The cap_goal_distance_reward is raw negative distance with no")
    print("exponential scaling, making it hard to balance against the")
    print("passive component. Idle strategy dominates.")


if __name__ == "__main__":
    run_example()
