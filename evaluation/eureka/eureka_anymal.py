"""Example: Eureka ANYmal — GPT-4 generated quadruped locomotion.

Eureka uses GPT-4 to write reward functions for Isaac Gym tasks.
The ANYmal reward is clean: velocity tracking is intentional and
the energy penalty is small and requires action. Both components
align with the locomotion objective.

Source: Ma et al. 2024 (ICLR), Eureka project — GPT-4 generated reward
Tool should catch: no critical issues (PASS)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka ANYmal — GPT-4 generated quadruped locomotion")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym ANYmal task.")
    print("Velocity tracking with small energy penalty.")
    print()

    model = EnvironmentModel(
        name="Eureka ANYmal",
        max_steps=1000,
        gamma=0.99,
        n_actions=12,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="velocity_tracking",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=True,
        intentional=True,
        explore_fraction=0.3,  # random movement produces some velocity
    ))
    model.add_reward_source(RewardSource(
        name="energy_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.01,
        requires_action=True,
    ))

    engine.print_report(model)

    print()
    print("Well-structured reward: velocity tracking directly measures")
    print("the locomotion objective and the energy penalty (-0.01) is")
    print("tiny relative to the tracking reward (1.0), so it regularizes")
    print("without creating an idle exploit. Both terms require action.")


if __name__ == "__main__":
    run_example()
