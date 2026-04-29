"""Example: Eureka Shadow Hand Over — GPT-4 generated hand-over reward.

Eureka uses GPT-4 to write reward functions for Isaac Gym tasks.
The Hand Over reward combines toss, catch, and fingertip penalty.
The catch_reward uses an unusual formulation: distance between goal
position and object linear velocity, which is semantically wrong
(comparing position to velocity). The penalty subtracts from
catch_reward, but catch_reward itself is based on a meaningless
metric.

Source: Ma et al. 2024 (ICLR), Eureka project — GPT-4 generated reward
Tool should catch: catch_reward compares position to velocity
  (semantic error), penalty reduces catch_reward not total reward
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Over — GPT-4 generated hand-over reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Shadow Hand Over.")
    print("Semantically wrong catch_reward and unusual penalty structure.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Over",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="toss_reward",
        reward_type=RewardType.PER_STEP,
        value=0.5,  # exp(-10*pos_dist/1.5)
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="catch_reward",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # exp(-30*||goal_pos - linvel||/0.5), wrong semantics
        requires_action=True,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.2,  # sigmoid(fingertip_count * 10) * 0.5 * catch_reward
        requires_action=True,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("The catch_reward is exp(-30*||goal_pos - object_linvel||/0.5),")
    print("which compares a position vector to a velocity vector. These")
    print("have different units and semantics — this is a GPT-4 error.")
    print("The penalty is proportional to catch_reward * sigmoid of")
    print("fingertip contact count, creating a coupled structure that")
    print("penalizes catching when fingertips touch the object.")


if __name__ == "__main__":
    run_example()
