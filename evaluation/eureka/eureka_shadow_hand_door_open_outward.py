"""Example: Eureka Shadow Hand Door Open Outward — GPT-4 generated reward.

Eureka uses GPT-4 to write reward functions for Isaac Gym tasks.
The Door Open Outward reward combines a handle reaching reward with
a door rotation reward measuring quaternion distance from closed
position. The handle_reward is passive if the hand starts near the
handle. The door_position_reward and door_orientation_reward both
measure how far the door is from closed, which starts at zero.

Source: Ma et al. 2024 (ICLR), Eureka project — GPT-4 generated reward
Tool should catch: passive handle_reward (critical), redundant
  door position and orientation rewards measuring same thing
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Door Open Outward — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Shadow Hand Door Open Outward.")
    print("Passive handle reward and redundant door rotation metrics.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Door Open Outward",
        max_steps=1000,
        gamma=0.99,
        n_actions=20,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="handle_reward",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # active: hand moves away
        value_range=(0.0, 1.0),  # idle: hand starts near handle
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="door_position_reward",
        reward_type=RewardType.PER_STEP,
        value=0.5,  # exp(-5*(1 - quat_dot)), high when door closed
        value_range=(0.0, 1.0),  # idle: door starts closed = max reward
        requires_action=False,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="door_orientation_reward",
        reward_type=RewardType.PER_STEP,
        value=0.5,  # exp(-0.5*angle_diff), high when door closed
        value_range=(0.0, 1.0),  # idle: door starts closed = max reward
        requires_action=False,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("The handle_reward is passive: exp(-0.5*dist) is high when the")
    print("hand starts near the handle. Critically, both door_position_")
    print("reward and door_orientation_reward measure distance from the")
    print("closed position using quaternion dot product and angle diff.")
    print("Since the task is to OPEN the door, these rewards are inverted:")
    print("they reward the door being CLOSED (the initial state). The")
    print("agent is rewarded for keeping the door closed, which directly")
    print("opposes the task objective.")


if __name__ == "__main__":
    run_example()
