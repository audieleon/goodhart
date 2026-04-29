"""Example: Eureka Shadow Hand Lift Underarm — GPT-4 generated reward.

Eureka uses GPT-4 to write reward functions for Isaac Gym tasks.
The Lift Underarm reward has a passive grasping component: hands
start near pot handles, triggering the grasping threshold. The
object_goal_reward uses a very high temperature (15.0) making it
extremely sensitive to small distances. The grasping_reward is
binary (threshold-based) and can be earned passively.

Source: Ma et al. 2024 (ICLR), Eureka project — GPT-4 generated reward
Tool should catch: passive grasping_reward (critical), extreme
  temperature on goal reward
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Lift Underarm — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Shadow Hand Lift Underarm.")
    print("Passive grasping reward and extreme goal temperature.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Lift Underarm",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="object_goal_reward",
        reward_type=RewardType.PER_STEP,
        value=0.5,  # exp(-15*dist), very sharp
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="grasping_reward",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # binary: both hands within 0.1 of handles
        value_range=(0.0, 0.3),  # idle: hands start near handles
        requires_action=False,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("The grasping_reward is binary (threshold 0.1) and passive:")
    print("hands start near pot handles, so both left_hand_grasping and")
    print("right_hand_grasping trigger at init. The object_goal_reward")
    print("uses temperature 15.0, making it extremely sensitive — reward")
    print("drops to near-zero very quickly with distance. The passive")
    print("grasping reward (0.3/step) dominates early training.")


if __name__ == "__main__":
    run_example()
