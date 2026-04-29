"""Krakovna entry: Running gaits -- unusual locomotion strategies.

Source: Kidzinski et al 2018 (Learning to Run challenge)
Failure: Agent develops bizarre, unnatural gaits (hopping, skipping,
  dragging limbs) instead of human-like running.
Mechanism: Distance/time reward without naturalness constraint.
  The objective is purely kinematic (maximize forward velocity),
  so the agent exploits simulator physics to find gaits that are
  efficient in simulation but physically impossible or unnatural.
Domain: MuJoCo / biomechanics
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Running gaits (Kidzinski et al 2018)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Running gaits (Kidzinski et al 2018)",
        max_steps=1000,
        gamma=0.99,
        n_actions=18,
        action_type="continuous",
    )

    # Forward velocity -- the only objective
    # Without naturalness constraints, any gait that maximizes
    # velocity is optimal, including unnatural ones.
    model.add_reward_source(RewardSource(
        name="forward_velocity",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=True,
        scales_with="velocity",
        state_dependent=True,
        intentional=True,
    ))

    # Alive bonus (standard)
    model.add_reward_source(RewardSource(
        name="alive_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.1,
        requires_action=False,
        intentional=True,
    ))

    # No naturalness penalty exists -- the structural gap.
    # A style or energy-efficiency penalty would constrain
    # the gait space to more realistic locomotion.

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
