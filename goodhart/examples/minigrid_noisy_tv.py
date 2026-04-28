"""Example: MiniGrid 6-room with noisy TV — 20x exploration collapse.

MSE curiosity agents visit 100+ novel states without noise. With
a noisy TV present, they collapse to visiting only 5 novel states.
The TV's pixel prediction error dominates the exploration signal
and the agent stays near it instead of entering new rooms.

Source: Mavor-Parker et al. 2022, "How to Stay Curious while
  avoiding Noisy TVs using Aleatoric Uncertainty Estimation"
  (ICML, arXiv:2102.04399)
Tool should catch: intrinsic_dominance (curiosity 5x goal)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("MiniGrid 6-room + noisy TV — 20x exploration collapse")
    print("=" * 70)
    print()
    print("Source: Mavor-Parker et al. 2022 (ICML)")
    print("100+ novel states without noise, 5 with noisy TV.")
    print()

    model = EnvironmentModel(
        name="MiniGrid 6-Room + Noisy TV",
        max_steps=500,
        gamma=0.99,
        n_actions=7,
        action_type="discrete",
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="room_goal",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        requires_action=True,
        intentional=True,
        discovery_probability=0.02,
    ))
    model.add_reward_source(RewardSource(
        name="mse_curiosity",
        reward_type=RewardType.PER_STEP,
        value=0.05,
        respawn=RespawnBehavior.INFINITE,
        requires_action=True,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("The noisy TV produces irreducible prediction error.")
    print("MSE curiosity never decays for stochastic observations.")
    print("The agent earns more from watching TV than from exploring")
    print("new rooms, collapsing from 100+ to 5 novel states visited.")


if __name__ == "__main__":
    run_example()
