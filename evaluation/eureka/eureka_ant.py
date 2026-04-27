"""Example: Eureka Ant — GPT-4 generated locomotion reward.

Eureka uses GPT-4 to write reward functions for Isaac Gym tasks.
The Ant reward combines forward velocity shaping with a height bonus.
The shaping term has no terminal goal, so the agent can cycle the
shaping signal indefinitely without ever reaching a destination.

Source: Ma et al. 2024 (ICLR), Eureka project — GPT-4 generated reward
Tool should catch: shaping with no terminal goal (critical),
  agent will cycle shaping forever
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Ant — GPT-4 generated locomotion reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Ant task.")
    print()

    model = EnvironmentModel(
        name="Eureka Ant",
        max_steps=1000,
        gamma=0.99,
        n_actions=8,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="forward_progress",
        reward_type=RewardType.SHAPING,
        value=1.0,
        requires_action=True,
        can_loop=False,
    ))
    model.add_reward_source(RewardSource(
        name="forward_velocity",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        value_type="exponential",
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="height_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        requires_action=False,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("The shaping term forward_progress has no terminal goal to")
    print("anchor it. The agent earns reward for moving forward but")
    print("has no incentive to reach any destination — it will cycle")
    print("the shaping signal forever. This is the encoding shown in")
    print("the Eureka paper appendix (Section: Encoding Workflow).")


if __name__ == "__main__":
    run_example()
