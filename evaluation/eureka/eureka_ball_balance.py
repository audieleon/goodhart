"""Example: Eureka Ball Balance — GPT-4 generated ball balancing reward.

Eureka uses GPT-4 to write reward functions for Isaac Gym tasks.
The Ball Balance reward has all passive components: low velocity,
low force, and surface contact are all maximized by doing nothing.
Every term rewards inaction, making the idle policy globally optimal.

Source: Ma et al. 2024 (ICLR), Eureka project — GPT-4 generated reward
Tool should catch: all components passive (critical), doing nothing
  maximizes every reward term
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Ball Balance — GPT-4 generated ball balancing reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Ball Balance task.")
    print("All three components are passive — doing nothing is optimal.")
    print()

    model = EnvironmentModel(
        name="Eureka Ball Balance",
        max_steps=500,
        gamma=0.99,
        n_actions=6,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="velocity_reward",
        reward_type=RewardType.PER_STEP,
        value=0.2,  # active: any movement increases velocity → less reward
        value_range=(0.0, 1.0),  # idle: zero velocity → max reward
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="force_reward",
        reward_type=RewardType.PER_STEP,
        value=0.1,  # active: actuators apply force → less reward
        value_range=(0.0, 0.5),  # idle: zero force → max reward
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="surface_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.1,  # active: risk knocking ball off surface
        value_range=(0.0, 0.3),  # idle: ball stays on surface
        requires_action=False,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("Every component rewards passivity. Idle: value_range maxima")
    print("(1.0 + 0.5 + 0.3 = 1.8/step) give EV ~179. Any action")
    print("increases velocity (reducing velocity_reward to ~0.2),")
    print("applies force (reducing force_reward to ~0.1), and risks")
    print("knocking the ball off (reducing surface_bonus to ~0.1).")
    print("Active EV ~40. The ball starts at rest on the surface,")
    print("already saturating all three reward terms.")


if __name__ == "__main__":
    run_example()
