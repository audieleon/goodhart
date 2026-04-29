"""Example: Eureka Shadow Hand Swing Cup — GPT-4 generated reward.

Eureka uses GPT-4 to write reward functions for Isaac Gym tasks.
The Swing Cup reward has passive grasp rewards: hands start near
cup handles. The object_goal_distance_reward is negative (penalty
form), creating a confusing signal. The cup_orientation_reward and
cup_linvel_penalty both discourage motion, rewarding stillness.
The touch_reward is redundant with grasp_reward.

Source: Ma et al. 2024 (ICLR), Eureka project — GPT-4 generated reward
Tool should catch: passive grasp_reward (critical), negative
  distance reward, velocity penalty rewards stillness
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Swing Cup — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Shadow Hand Swing Cup.")
    print("Passive grasp, negative distance, velocity penalty.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Swing Cup",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="grasp_reward",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # exp(-0.25*dist) for each handle, sum - 1.0
        value_range=(-1.0, 1.0),  # idle: hands near handles = ~1.0
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="object_goal_distance_reward",
        reward_type=RewardType.PER_STEP,
        value=-0.5,  # -exp(-0.1*dist)*2, always negative
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="cup_orientation_reward",
        reward_type=RewardType.PER_STEP,
        value=-0.1,  # -(1 - dot(rot,goal)^2)*0.25
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="cup_linvel_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.05,  # -||linvel||*0.1, zero at rest
        value_range=(-0.5, 0.0),  # idle: zero velocity = zero penalty
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="touch_reward",
        reward_type=RewardType.PER_STEP,
        value=0.1,  # exp(-0.25*dist) for handles, redundant with grasp
        value_range=(0.0, 0.125),  # idle: hands near handles
        requires_action=False,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("The grasp_reward is passive: hands start near cup handles,")
    print("giving high exp(-0.25*dist) at idle. The distance reward is")
    print("always negative (-exp(-0.1*dist)*2), so moving the cup toward")
    print("the goal reduces total reward. The velocity penalty is zero")
    print("at rest, rewarding stillness. touch_reward duplicates grasp.")
    print("The reward structure actively discourages task completion.")


if __name__ == "__main__":
    run_example()
