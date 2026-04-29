"""Example: Eureka Shadow Hand Door Open Inward — GPT-4 generated reward.

Eureka uses GPT-4 to write reward functions for Isaac Gym tasks.
The Door Open Inward reward has hand-to-handle distance and
orientation alignment components. The hand-handle distance rewards
are passive if hands start near handles. The orientation rewards
measure hand-handle rotation alignment, also potentially passive.

Source: Ma et al. 2024 (ICLR), Eureka project — GPT-4 generated reward
Tool should catch: passive hand_handle_dist rewards (critical),
  passive orientation rewards
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Door Open Inward — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Shadow Hand Door Open Inward.")
    print("Passive hand-handle distance and orientation rewards.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Door Open Inward",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="reward_hand_handle_dist_left",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # active: hand moves away
        value_range=(0.0, 0.9),  # idle: hand starts near handle
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="reward_hand_handle_dist_right",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # active: hand moves away
        value_range=(0.0, 0.9),  # idle: hand starts near handle
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="reward_door_handle_orientation_left",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        value_range=(0.0, 1.0),  # idle: may start aligned
        requires_action=False,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="reward_door_handle_orientation_right",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        value_range=(0.0, 1.0),  # idle: may start aligned
        requires_action=False,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("All four components are passive or partially passive: hand-")
    print("handle distances start small (high exp reward) and orientation")
    print("alignment may be high at initialization. The agent can earn")
    print("substantial reward by doing nothing. No terminal goal anchors")
    print("the overall reward structure.")


if __name__ == "__main__":
    run_example()
