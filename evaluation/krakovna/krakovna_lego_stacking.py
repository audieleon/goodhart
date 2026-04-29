"""Krakovna entry: Lego stacking -- agent flips block instead of stacking.

Source: Popov et al 2017
Failure: Agent flips the block upside-down instead of stacking it on
  top of another block.
Mechanism: Stacking proxy measures height of the bottom face of the
  block. Flipping the block achieves the target bottom-face height
  without performing actual stacking. The proxy conflates "bottom
  face is high" with "block is on top of another block."
Domain: Robotics / manipulation

Also in goodhart/examples/block_stacking.py
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Lego stacking (Popov et al 2017)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Lego stacking (Popov et al 2017)",
        max_steps=500,
        gamma=0.99,
        n_actions=6,
        action_type="continuous",
    )

    # True task: stack block on top of another
    model.add_reward_source(RewardSource(
        name="block_stacked",
        reward_type=RewardType.TERMINAL,
        value=10.0,
        discovery_probability=0.01,
        intentional=True,
    ))

    # Proxy: height of bottom face -- the trap
    # Flipping the block satisfies this without stacking.
    model.add_reward_source(RewardSource(
        name="bottom_face_height",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        scales_with="z_position",
        state_dependent=True,
        value_type="proportional",
        value_params={"scale": 1.0},
    ))

    # Grasp reward (shaping to encourage picking up block)
    model.add_reward_source(RewardSource(
        name="grasp_reward",
        reward_type=RewardType.ON_EVENT,
        value=0.5,
        max_occurrences=1,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
