"""Example: Eureka Allegro Hand — GPT-4 generated dexterous manipulation.

Eureka uses GPT-4 to write reward functions for Isaac Gym tasks.
The Allegro Hand reward is clean: orientation reward and fingertip
bonus both require action and are intentional. No passive terms.

Source: Ma et al. 2024 (ICLR), Eureka project — GPT-4 generated reward
Tool should catch: no critical issues (PASS)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Allegro Hand — GPT-4 generated manipulation reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Allegro Hand task.")
    print("Both components require action and are intentional.")
    print()

    model = EnvironmentModel(
        name="Eureka Allegro Hand",
        max_steps=1000,
        gamma=0.99,
        n_actions=16,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="orientation_reward",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="fingertip_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.3,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("Another clean Eureka output. Both reward components directly")
    print("measure the manipulation objective: orienting the object and")
    print("maintaining fingertip contact. No passive terms to exploit.")


if __name__ == "__main__":
    run_example()
