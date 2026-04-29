"""Krakovna entry: Wall sensor stack — tricking the sensor.

Source: Le Paine et al 2019
Failure: Agent is supposed to stack blocks to press a wall sensor,
  but instead discovers an environment bug that lets it activate the
  sensor directly by pressing it in a precise way, bypassing the
  intended stacking behavior.
Mechanism: Binary reward for sensor activation. The intended solution
  requires stacking blocks high enough to reach and press the sensor.
  The environment bug allows direct sensor activation without stacking.
Domain: Robotics / manipulation

Structural encoding: We encode the binary sensor activation as a
  terminal reward and a distance shaping signal toward the sensor.
  The reward structure itself is minimal and correct — activate the
  sensor. The exploit is an environment dynamics bug that provides
  an unintended shortcut.

Advisory: The exploit is an environment bug (sensor can be tricked
  without stacking). The reward function correctly specifies the goal
  (press the sensor). Structural analysis would not flag this — the
  reward is not misspecified, the environment dynamics are broken.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Wall sensor stack (Le Paine et al 2019)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Wall Sensor Stack (Le Paine et al 2019)",
        max_steps=500,
        gamma=0.99,
        n_actions=6,
        action_type="continuous",
    )

    # Binary sensor activation — the intended goal.
    model.add_reward_source(RewardSource(
        name="sensor_activation",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        intentional=True,
        discovery_probability=0.05,
    ))

    # Distance shaping toward sensor position
    model.add_reward_source(RewardSource(
        name="distance_to_sensor",
        reward_type=RewardType.SHAPING,
        value=0.1,
        scales_with="distance",
        state_dependent=True,
        requires_action=True,
    ))

    engine.print_report(model)

    print()
    print("Advisory: environment_bug — the sensor can be activated without")
    print("stacking blocks, bypassing the intended solution path. The")
    print("reward function (binary sensor activation) correctly specifies")
    print("the goal. The problem is that the environment provides an")
    print("unintended shortcut to achieve that goal, which structural")
    print("reward analysis cannot detect.")


if __name__ == "__main__":
    run_example()
