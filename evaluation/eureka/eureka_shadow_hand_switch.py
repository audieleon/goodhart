"""Example: Eureka Shadow Hand Switch — GPT-4 generated reward.

Eureka uses GPT-4 to write reward functions for Isaac Gym tasks.
The Switch reward uses only a touch_reward based on fingertip-to-
switch-handle distances. This is passive: fingertips start near the
switch handles. The reward uses pow(sum(exp(-dist)), 2) which
amplifies the passive signal. There is no task completion component
measuring whether the switch was actually toggled.

Source: Ma et al. 2024 (ICLR), Eureka project — GPT-4 generated reward
Tool should catch: passive touch_reward (critical), no task
  completion signal (missing intentional component)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Switch — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Shadow Hand Switch.")
    print("Passive touch reward with no task completion signal.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Switch",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="touch_reward",
        reward_type=RewardType.PER_STEP,
        value=1.0,  # pow(sum(exp(-dist)),2) * pow(sum(exp(-dist)),2)
        value_range=(0.0, 25.0),  # idle: all fingertips near handle
        requires_action=False,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("The touch_reward is the ONLY reward component. It measures")
    print("fingertip proximity to switch handles using pow(sum(exp(-d)),2)")
    print("for each hand, then multiplies them. This is entirely passive:")
    print("fingertips start near handles, giving high reward at idle.")
    print("Critically, there is NO component measuring whether the switch")
    print("was toggled — the task objective is completely absent from the")
    print("reward function.")


if __name__ == "__main__":
    run_example()
