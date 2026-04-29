"""Example: Eureka Shadow Hand Re-Orientation — GPT-4 generated reward.

Eureka uses GPT-4 to write reward functions for Isaac Gym tasks.
The Re-Orientation reward is well-structured: purely rotation-based
with two objects that need to match goal orientations. Success
bonuses provide near-terminal incentive. No passive components
since rotation rewards require active manipulation.

Source: Ma et al. 2024 (ICLR), Eureka project — GPT-4 generated reward
Tool should catch: no critical issues (PASS), clean rotation reward
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Re-Orientation — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Shadow Hand Re-Orientation.")
    print("Clean rotation-based reward with success bonuses.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Re-Orientation",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="object_rotation_reward",
        reward_type=RewardType.PER_STEP,
        value=0.5,  # exp(-8*rot_distance)
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="object_another_rotation_reward",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # exp(-5*rot_distance)
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="object_orientation_success_bonus",
        reward_type=RewardType.PER_STEP,
        value=1.0,  # 5*(1-rot_dist) when rot_dist < 0.02
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="object_another_orientation_success_bonus",
        reward_type=RewardType.PER_STEP,
        value=1.0,  # 5*(1-rot_dist) when rot_dist < 0.02
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("Well-structured reward: all components directly measure the")
    print("task objective (object orientation matching goal). The exp")
    print("transforms with temperatures 8.0 and 5.0 create appropriate")
    print("gradients. Success bonuses (5x scale) fire when rotation")
    print("distance < 0.02, providing strong near-goal incentive. No")
    print("passive components — rotation always requires active control.")


if __name__ == "__main__":
    run_example()
