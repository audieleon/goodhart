"""Example: Eureka Shadow Hand Door Close Inward — GPT-4 generated reward.

Eureka uses GPT-4 to write reward functions for Isaac Gym tasks.
The Door Close Inward reward has passive hand-distance components:
hands start near door handles, so exp(-dist) is high at idle. All
four components are distance-based with no terminal goal. The
handle-to-goal distance rewards are also passive if the door starts
near the goal position.

Source: Ma et al. 2024 (ICLR), Eureka project — GPT-4 generated reward
Tool should catch: passive hand_distance_rewards (critical),
  no terminal goal
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Door Close Inward — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Shadow Hand Door Close Inward.")
    print("Passive hand-to-handle distance rewards create idle exploit.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Door Close Inward",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="left_hand_distance_reward",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # active: hand moves
        value_range=(0.0, 1.0),  # idle: hand starts near handle
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="right_hand_distance_reward",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # active: hand moves
        value_range=(0.0, 1.0),  # idle: hand starts near handle
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="goal_left_handle_distance_reward",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="goal_right_handle_distance_reward",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("The hand distance rewards are passive: exp(-dist) is high when")
    print("hands start near handles. With two passive components each")
    print("yielding ~1.0/step at idle, the idle floor (2.0/step) competes")
    print("with the goal handle rewards. No terminal goal anchors the")
    print("shaping signals.")


if __name__ == "__main__":
    run_example()
