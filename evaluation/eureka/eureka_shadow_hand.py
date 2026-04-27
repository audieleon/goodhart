"""Example: Eureka Shadow Hand — GPT-4 generated dexterous manipulation.

Eureka uses GPT-4 to write reward functions for Isaac Gym tasks.
The Shadow Hand reward has a passive distance_reward: fingertips
start near the object, so the agent earns reward by doing nothing.
The idle expected value (87) exceeds the active value (43), creating
a floor that discourages exploration.

Source: Ma et al. 2024 (ICLR), Eureka project — GPT-4 generated reward
Tool should catch: passive distance reward creates idle floor
  (critical), EV 87 idle vs 43 active
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand — GPT-4 generated manipulation reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Shadow Hand task.")
    print("Passive distance reward creates idle exploit.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand",
        max_steps=1000,
        gamma=0.99,
        n_actions=20,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="distance_reward",
        reward_type=RewardType.PER_STEP,
        value=0.2,  # active: fingertips move away from object
        value_range=(0.0, 1.0),  # idle: fingertips stay near object (1.0)
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="rotation_reward",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="fingertip_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.2,
        requires_action=True,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("The distance_reward is passive (requires_action=False) because")
    print("fingertips start near the object. Idle: value_range max (1.0/step)")
    print("gives EV ~100. Active manipulation moves fingertips away, reducing")
    print("distance_reward to ~0.2/step while gaining 0.5/step rotation at")
    print("explore_fraction=0.0. Explore EV ~40. The idle floor exceeds the")
    print("active ceiling, making inaction the dominant strategy.")


if __name__ == "__main__":
    run_example()
