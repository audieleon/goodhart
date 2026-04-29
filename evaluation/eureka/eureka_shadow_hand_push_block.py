"""Example: Eureka Shadow Hand Push Block — GPT-4 generated reward.

Eureka uses GPT-4 to write reward functions for Isaac Gym tasks.
The Push Block reward has passive hand-to-block proximity rewards:
hands start near the block, so exp(-0.1*dist) is high at idle.
The block-to-goal proximity rewards require action. All components
use equal weights (0.25), giving passive components significant
influence.

Source: Ma et al. 2024 (ICLR), Eureka project — GPT-4 generated reward
Tool should catch: passive hand proximity rewards (critical)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Push Block — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Shadow Hand Push Block.")
    print("Passive hand-to-block proximity rewards create idle exploit.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Push Block",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="left_proximity_reward",
        reward_type=RewardType.PER_STEP,
        value=0.25,  # exp(-0.5*dist_block_to_left_goal)
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="right_proximity_reward",
        reward_type=RewardType.PER_STEP,
        value=0.25,  # exp(-0.5*dist_block_to_right_goal)
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="left_hand_reward",
        reward_type=RewardType.PER_STEP,
        value=0.1,  # active: hand moves away
        value_range=(0.0, 0.25),  # idle: hand starts near block
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="right_hand_reward",
        reward_type=RewardType.PER_STEP,
        value=0.1,  # active: hand moves away
        value_range=(0.0, 0.25),  # idle: hand starts near block
        requires_action=False,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("The hand proximity rewards are passive: exp(-0.1*dist) is")
    print("high when hands start near the block. With temperature 0.1")
    print("the decay is very slow, so even moderate distances yield")
    print("significant reward. The block-to-goal rewards require pushing")
    print("the block but compete with the passive hand rewards. Equal")
    print("weighting (0.25 each) gives passive components 50% of total.")


if __name__ == "__main__":
    run_example()
