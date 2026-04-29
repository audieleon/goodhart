"""Example: Eureka Shadow Hand Scissors — GPT-4 generated reward.

Eureka uses GPT-4 to write reward functions for Isaac Gym tasks.
The Scissors reward has an opened_reward that grows exponentially
with handle separation beyond target distance (exp(5*(dist-0.3))).
This is unbounded and can be exploited by maximizing handle
separation far beyond the target. The reaching penalty is linear
and passive (hands start near handles).

Source: Ma et al. 2024 (ICLR), Eureka project — GPT-4 generated reward
Tool should catch: unbounded opened_reward (critical), passive
  reaching penalty component
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Scissors — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Shadow Hand Scissors.")
    print("Unbounded exponential opened_reward and passive reaching.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Scissors",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="opened_reward",
        reward_type=RewardType.PER_STEP,
        value=2.0,  # exp(5*(dist-0.3)), unbounded growth
        requires_action=True,
        intentional=True,
        can_loop=False,
    ))
    model.add_reward_source(RewardSource(
        name="handle_reaching_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.2,  # -0.5*(right_dist + left_dist)
        value_range=(-1.0, 0.0),  # idle: hands near handles = low penalty
        requires_action=False,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("The opened_reward uses exp(5*(handle_dist-0.3)), which grows")
    print("exponentially as handles separate beyond 0.3. This is")
    print("unbounded: the agent is incentivized to rip the scissors")
    print("apart as far as possible, not just open them to a functional")
    print("angle. The reaching penalty is linear and small (-0.2) so")
    print("it cannot counterbalance the exponential opened_reward.")


if __name__ == "__main__":
    run_example()
