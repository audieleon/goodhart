"""Example: Eureka Shadow Hand Door Close Outward — GPT-4 generated reward.

Eureka uses GPT-4 to write reward functions for Isaac Gym tasks.
The Door Close Outward reward uses negative exp(dist/temp) for hand
approach (penalizing distance) and positive alignment rewards for
door handle orientation matching goal. The hand approach rewards use
an inverted exponential that grows unbounded with distance, which is
unusual. The alignment rewards are passive if handles start aligned.

Source: Ma et al. 2024 (ICLR), Eureka project — GPT-4 generated reward
Tool should catch: passive alignment rewards if door starts aligned,
  unbounded negative approach penalties
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Door Close Outward — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Shadow Hand Door Close Outward.")
    print("Inverted exp penalties and passive alignment rewards.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Door Close Outward",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="left_hand_approach_reward",
        reward_type=RewardType.PER_STEP,
        value=-0.3,  # -exp(dist/10), negative and grows with distance
        requires_action=True,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="right_hand_approach_reward",
        reward_type=RewardType.PER_STEP,
        value=-0.3,  # -exp(dist/10), negative and grows with distance
        requires_action=True,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="alignment_reward_left",
        reward_type=RewardType.PER_STEP,
        value=0.5,  # dot product * 5, passive if aligned at start
        value_range=(0.0, 1.0),
        requires_action=False,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="alignment_reward_right",
        reward_type=RewardType.PER_STEP,
        value=0.5,  # dot product * 5, passive if aligned at start
        value_range=(0.0, 1.0),
        requires_action=False,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("The hand approach rewards use -exp(dist/10), an inverted")
    print("exponential that penalizes distance but grows unbounded.")
    print("The alignment rewards measure quaternion dot product with")
    print("goal rotation, which may be high at start if the door is")
    print("already partially aligned. The approach penalties encourage")
    print("reaching the handle but the alignment rewards can be earned")
    print("passively if the door starts near the closed position.")


if __name__ == "__main__":
    run_example()
