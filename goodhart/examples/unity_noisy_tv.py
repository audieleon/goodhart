"""Example: Unity 3D maze with noisy TV — canonical curiosity trap.

9-room maze with sparse +1.0 terminal goal. An in-world TV can be
channel-changed by the agent. Each new channel image is maximally
novel, generating constant high intrinsic reward. The agent discovers
the TV and stops exploring the maze entirely.

Source: Burda et al. 2019, "Large-Scale Study of Curiosity-Driven
  Learning" (ICLR, arXiv:1808.04355)
Tool should catch: intrinsic_dominance (curiosity competes with goal)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Unity 3D maze with noisy TV — the canonical curiosity trap")
    print("=" * 70)
    print()
    print("Source: Burda et al. 2019 (ICLR)")
    print("Sparse goal +1.0, curiosity coeff 0.01.")
    print("Agent watches TV instead of solving the maze.")
    print()

    model = EnvironmentModel(
        name="Unity Maze + Noisy TV",
        max_steps=1000,
        gamma=0.99,
        n_actions=5,  # forward, left, right, no-op, change_channel
        action_type="discrete",
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="maze_goal",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        requires_action=True,
        intentional=True,
        discovery_probability=0.05,
    ))
    model.add_reward_source(RewardSource(
        name="curiosity_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.01,
        respawn=RespawnBehavior.INFINITE,
        requires_action=True,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("The TV generates maximally novel observations on every")
    print("channel change. The curiosity bonus never decays for the TV")
    print("because each image is unique. The agent parks at the TV")
    print("and changes channels instead of exploring the maze.")


if __name__ == "__main__":
    run_example()
