"""Example: Eureka Humanoid — GPT-4 generated locomotion reward.

Eureka uses GPT-4 to write reward functions for Isaac Gym tasks.
The Humanoid reward includes an exp(x/0.01) time penalty that
creates an enormous effective penalty of ~20.0/step. Since the
alive bonus is only 2.0/step, dying beats surviving by 9.6x —
the agent learns to terminate episodes as fast as possible.

Source: Ma et al. 2024 (ICLR), Eureka project — GPT-4 generated reward
Tool should catch: time penalty dominates alive bonus (critical),
  dying is 9.6x better than surviving
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Humanoid — GPT-4 generated locomotion reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Humanoid task.")
    print("The exp(x/0.01) time penalty creates ~20.0/step effective cost.")
    print()

    model = EnvironmentModel(
        name="Eureka Humanoid",
        max_steps=1000,
        gamma=0.99,
        n_actions=21,
        action_type="continuous",
        death_probability=0.05,
    )
    model.add_reward_source(RewardSource(
        name="forward_velocity",
        reward_type=RewardType.PER_STEP,
        value=1.5,
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="alive_bonus",
        reward_type=RewardType.PER_STEP,
        value=2.0,
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="time_penalty",
        reward_type=RewardType.PER_STEP,
        value=-20.0,
        requires_action=False,
    ))

    engine.print_report(model)

    print()
    print("The exp(x/0.01) time penalty produces ~20.0/step cost,")
    print("dwarfing the alive bonus (2.0/step) and forward velocity")
    print("(1.5/step). Dying ends the episode and stops accumulating")
    print("penalty, making death 9.6x more rewarding than surviving.")
    print("This is a classic Goodhart failure: GPT-4 added the penalty")
    print("to encourage efficiency, but the scale makes suicide optimal.")


if __name__ == "__main__":
    run_example()
