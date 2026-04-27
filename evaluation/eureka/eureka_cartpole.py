"""Example: Eureka CartPole — GPT-4 generated balancing reward.

Eureka uses GPT-4 to write reward functions for Isaac Gym tasks.
The CartPole reward is clean: both components are intentional
Gaussian-shaped rewards that require action. No passive terms,
no exploitable shaping.

Source: Ma et al. 2024 (ICLR), Eureka project — GPT-4 generated reward
Tool should catch: no critical issues (PASS)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka CartPole — GPT-4 generated balancing reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym CartPole task.")
    print("Both components are intentional and Gaussian-shaped.")
    print()

    model = EnvironmentModel(
        name="Eureka CartPole",
        max_steps=500,
        gamma=0.99,
        n_actions=2,
        action_type="discrete",
    )
    model.add_reward_source(RewardSource(
        name="angle_reward",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="position_reward",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("This is one of Eureka's cleanest outputs. Both reward")
    print("components directly measure what we care about (pole angle")
    print("and cart position), both require action, and both are marked")
    print("intentional. No passive terms that could create idle exploits.")


if __name__ == "__main__":
    run_example()
