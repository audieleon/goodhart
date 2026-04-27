"""Example: Eureka Franka Cabinet — GPT-4 generated cabinet opening.

Eureka uses GPT-4 to write reward functions for Isaac Gym tasks.
The Franka Cabinet reward is clean: cabinet opening and gripper
bonus both require action and are intentional. The velocity penalty
is small and correctly structured (requires action).

Source: Ma et al. 2024 (ICLR), Eureka project — GPT-4 generated reward
Tool should catch: no critical issues (PASS)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Franka Cabinet — GPT-4 generated cabinet opening")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Franka Cabinet task.")
    print("Velocity penalty correctly structured with requires_action.")
    print()

    model = EnvironmentModel(
        name="Eureka Franka Cabinet",
        max_steps=500,
        gamma=0.99,
        n_actions=9,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="cabinet_open",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=True,
        intentional=True,
        explore_fraction=0.2,  # random arm movement partially opens cabinet
    ))
    model.add_reward_source(RewardSource(
        name="gripper_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.3,
        requires_action=True,
        intentional=True,
        explore_fraction=0.2,  # random gripper actuation earns partial credit
    ))
    model.add_reward_source(RewardSource(
        name="velocity_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.05,
        requires_action=True,
    ))

    engine.print_report(model)

    print()
    print("Well-designed reward: both positive components directly measure")
    print("the task objective (opening the cabinet, gripping the handle).")
    print("The velocity penalty (-0.05) is small relative to the task")
    print("rewards (1.0 + 0.3) and requires action, so it cannot create")
    print("an idle exploit. This is one of Eureka's better outputs.")


if __name__ == "__main__":
    run_example()
