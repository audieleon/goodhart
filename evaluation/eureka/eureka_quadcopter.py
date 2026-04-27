"""Example: Eureka Quadcopter — GPT-4 generated flight reward.

Eureka uses GPT-4 to write reward functions for Isaac Gym tasks.
The Quadcopter reward has three critical issues: penalties total
12.0/step vs a sparse goal worth 15.0 (achievable ~1% of steps),
creating a massive idle exploit; the penalty structure also makes
exploration impossible since any movement incurs immediate cost.

Source: Ma et al. 2024 (ICLR), Eureka project — GPT-4 generated reward
Tool should catch: penalties 12x goal (critical), idle exploit
  from penalty avoidance (critical), exploration impossible (critical)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Quadcopter — GPT-4 generated flight reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Quadcopter task.")
    print("Penalties total 12.0/step vs sparse goal of 15.0.")
    print()

    model = EnvironmentModel(
        name="Eureka Quadcopter",
        max_steps=500,
        gamma=0.99,
        n_actions=4,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="distance_to_target",
        reward_type=RewardType.TERMINAL,
        value=15.0,
        requires_action=True,
        intentional=True,
        discovery_probability=0.01,
    ))
    model.add_reward_source(RewardSource(
        name="distance_penalty",
        reward_type=RewardType.PER_STEP,
        value=-6.0,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="linvel_penalty",
        reward_type=RewardType.PER_STEP,
        value=-4.0,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="angvel_penalty",
        reward_type=RewardType.PER_STEP,
        value=-2.0,
        requires_action=False,
    ))

    engine.print_report(model)

    print()
    print("Three simultaneous failures: (1) penalties total 12.0/step")
    print("while the goal is worth 15.0 and achievable ~1% of the time,")
    print("so doing nothing avoids 6000 penalty over 500 steps;")
    print("(2) all penalties are passive (requires_action=False), so")
    print("the agent minimizes them by staying still; (3) any exploratory")
    print("movement incurs immediate cost, preventing discovery of the")
    print("goal. GPT-4 added the penalties to encourage smooth flight,")
    print("but their scale makes stillness the dominant strategy.")


if __name__ == "__main__":
    run_example()
