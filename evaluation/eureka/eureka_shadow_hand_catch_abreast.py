"""Example: Eureka Shadow Hand Catch Abreast — GPT-4 generated reward.

Eureka uses GPT-4 to write reward functions for Isaac Gym tasks.
The Catch Abreast reward has a passive grasp_reward: hands start
near the object, so exp(-dist/0.08) is high at idle. The goal_bonus
provides a sparse terminal-like signal, but the grasp and contact
rewards dominate and can be earned passively.

Source: Ma et al. 2024 (ICLR), Eureka project — GPT-4 generated reward
Tool should catch: passive grasp_reward creates idle floor (critical)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Catch Abreast — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Shadow Hand Catch Abreast.")
    print("Passive grasp reward creates idle exploit.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Catch Abreast",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="grasp_reward",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # active: object moves away
        value_range=(0.0, 1.0),  # idle: hands start near object
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="goal_reward",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="smooth_contact_reward",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # exp(-|left_dist - right_dist|/0.12)
        value_range=(0.0, 1.0),  # idle: symmetric hand distances
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="goal_bonus",
        reward_type=RewardType.PER_STEP,
        value=1.0,  # sparse: only when dist < 0.03
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("The grasp_reward is passive: exp(-dist/0.08) is high when")
    print("hands start near the object. smooth_contact_reward is also")
    print("passive: symmetric hand positions at idle yield high reward.")
    print("Together these create an idle floor that may dominate the")
    print("goal_reward signal. The 25x goal_bonus helps but is sparse.")


if __name__ == "__main__":
    run_example()
