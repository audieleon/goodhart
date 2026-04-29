"""Example: Eureka Shadow Hand Grasp and Place — GPT-4 generated reward.

Eureka uses GPT-4 to write reward functions for Isaac Gym tasks.
The Grasp and Place reward has a grab_success sigmoid that rewards
hand proximity, and an instability_penalty that penalizes object
motion. The instability penalty rewards stillness. The distance and
rotation rewards are multiplied by grab_success and subtracted from
1.0, creating an unusual reward structure where higher distance/rot
rewards reduce total reward.

Source: Ma et al. 2024 (ICLR), Eureka project — GPT-4 generated reward
Tool should catch: instability penalty rewards idleness (critical),
  inverted distance/rot reward logic
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Grasp and Place — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Shadow Hand Grasp and Place.")
    print("Instability penalty rewards stillness, inverted reward logic.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Grasp and Place",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="hand_distance_reward",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # exp(-0.1*hand_dist)
        value_range=(0.0, 1.0),  # idle: hands start near object
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="grab_success",
        reward_type=RewardType.PER_STEP,
        value=0.5,  # sigmoid(5*(1-hand_dist))
        value_range=(0.0, 1.0),  # idle: hands near object = high
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="distance_reward",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # |exp(-0.05*dist)-1| * grab_success
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="rot_reward",
        reward_type=RewardType.PER_STEP,
        value=0.2,  # |exp(-2*rot_dist)-1| * grab_success
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="instability_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.3,  # -exp(-linvel), rewards zero velocity
        value_range=(-1.0, 0.0),  # idle: zero velocity = max penalty
        requires_action=False,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("The instability_penalty is -exp(-linvel), which is most")
    print("negative (penalty is largest) when velocity is zero. This")
    print("actually penalizes stillness. But hand_distance_reward and")
    print("grab_success are both passive, creating an idle floor.")
    print("The distance and rotation rewards use |exp(-x)-1|, which")
    print("approaches 0 as distance/rotation approach 0 (success),")
    print("creating an inverted signal that rewards being far from goal.")


if __name__ == "__main__":
    run_example()
