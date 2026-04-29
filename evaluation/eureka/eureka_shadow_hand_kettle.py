"""Example: Eureka Shadow Hand Kettle — GPT-4 generated reward.

Eureka uses GPT-4 to write reward functions for Isaac Gym tasks.
The Kettle reward has passive handle-grasping rewards: hands start
near kettle/bucket handles, so exp(-0.5*dist) is high at idle.
The spout-bucket alignment rewards require action. The no_spill
reward uses a hardcoded constant (0.5) instead of actual ball count,
making it a fixed bias term that adds nothing.

Source: Ma et al. 2024 (ICLR), Eureka project — GPT-4 generated reward
Tool should catch: passive handle rewards (critical), hardcoded
  no_spill constant is meaningless
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Kettle — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Shadow Hand Kettle.")
    print("Passive handle grasping rewards and hardcoded no_spill term.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Kettle",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="reward_kettle_handle",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # active: hand moves away
        value_range=(0.0, 1.0),  # idle: hand starts near handle
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="reward_bucket_handle",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # active: hand moves away
        value_range=(0.0, 1.0),  # idle: hand starts near handle
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="reward_spout_bucket",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # exp(-0.3*dist_spout_bucket)
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="reward_spout_bucket_z",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # exp(-2.5*z_diff)
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="reward_no_spill",
        reward_type=RewardType.PER_STEP,
        value=0.69,  # log(1+exp(-5*(0.5-0.5))) = log(2), constant
        requires_action=False,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("The kettle and bucket handle rewards are passive: hands start")
    print("near the handles, giving high exp(-0.5*dist) at idle. The")
    print("reward_no_spill uses a hardcoded constant (0.5) instead of")
    print("actual ball count, producing a fixed log(2)=0.69 every step.")
    print("This is pure bias that adds no training signal. Together the")
    print("passive components dominate the reward landscape.")


if __name__ == "__main__":
    run_example()
