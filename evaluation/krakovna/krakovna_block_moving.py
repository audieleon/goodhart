"""Krakovna entry: Block moving -- agent moves table instead of block.

Source: Chopra 2018
Failure: Agent moves the table the block is sitting on, rather than the
  block itself, to minimize distance-to-target.
Mechanism: Distance-to-target measured in the wrong reference frame.
  The proxy (block center distance) can be optimized by moving the
  surface the block rests on, which is mechanically easier than
  grasping and moving the block directly.
Domain: Robotics / manipulation
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Block moving (Chopra 2018)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Block moving (Chopra 2018)",
        max_steps=500,
        gamma=0.99,
        n_actions=6,
        action_type="continuous",
    )

    # True task: place block at target location
    model.add_reward_source(RewardSource(
        name="block_at_target",
        reward_type=RewardType.TERMINAL,
        value=10.0,
        discovery_probability=0.02,
        intentional=True,
    ))

    # Shaping: distance of block center to target -- the trap
    # Agent can reduce this by moving the table (which moves the block
    # indirectly), a mechanically easier strategy than grasping.
    model.add_reward_source(RewardSource(
        name="block_distance_to_target",
        reward_type=RewardType.SHAPING,
        value=1.0,
        can_loop=False,
        scales_with="distance",
        state_dependent=True,
        value_type="inverse",
        value_params={"eps": 0.01},
    ))

    # The key issue: table movement also decreases block distance
    # and is much easier to discover than grasping
    model.add_reward_source(RewardSource(
        name="table_push_shortcut",
        reward_type=RewardType.SHAPING,
        value=1.0,
        can_loop=False,
        requires_exploration=False,
        discovery_probability=0.8,
        scales_with="distance",
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
