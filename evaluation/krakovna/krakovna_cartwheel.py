"""Krakovna entry: Cartwheel Ant -- MuJoCo Ant does cartwheels.

Source: Jucys 2024
Failure: Ant does a cartwheel instead of jumping. The torso z-coordinate
  exceeds the threshold, but via rotation rather than vertical leap.
Mechanism: Z-coordinate threshold used as proxy for "jump height."
  The Ant's body geometry allows it to achieve high torso z by
  cartwheeling (rotating the body so limbs act as spokes), which
  satisfies the threshold without any actual jumping.
Domain: MuJoCo / locomotion
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Cartwheel Ant (Jucys 2024)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Cartwheel Ant (Jucys 2024)",
        max_steps=1000,
        gamma=0.99,
        n_actions=8,
        action_type="continuous",
    )

    # The intended behavior: jump (vertical displacement)
    # Not directly rewarded -- only the z-threshold proxy exists.

    # Z-coordinate threshold bonus -- the trap
    # Rewards torso z > threshold, but cartwheeling achieves this
    # more easily than jumping.
    model.add_reward_source(RewardSource(
        name="z_threshold_bonus",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=True,
        can_loop=True,
        loop_period=10,
        scales_with="z_position",
        state_dependent=True,
    ))

    # Alive bonus (standard MuJoCo)
    model.add_reward_source(RewardSource(
        name="alive_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.05,
        requires_action=False,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
