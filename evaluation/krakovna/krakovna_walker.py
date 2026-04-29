"""Krakovna entry: Walker -- walks on one leg to match target speed.

Source: Lee et al 2021
Failure: Walker learns to hop on one leg instead of walking normally,
  because this matches the target velocity without naturalness.
Mechanism: Speed-matching reward without style or naturalness penalty.
  The agent finds that hopping on one leg is a simpler control
  strategy that achieves the target velocity. Without a penalty
  for unnatural postures, this degenerate gait dominates.
Domain: MuJoCo / locomotion
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Walker one-leg (Lee et al 2021)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Walker (Lee et al 2021)",
        max_steps=1000,
        gamma=0.99,
        n_actions=6,
        action_type="continuous",
    )

    # Speed-matching reward -- the trap
    # Reward is exp(-|v - v_target|^2 / sigma^2).
    # Matches target speed but does not constrain gait style.
    model.add_reward_source(RewardSource(
        name="speed_matching",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=True,
        scales_with="velocity",
        state_dependent=True,
        value_type="exponential",
        value_params={"sigma": 0.25},
        intentional=True,
    ))

    # Alive bonus
    model.add_reward_source(RewardSource(
        name="alive_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.1,
        requires_action=False,
        intentional=True,
    ))

    # No naturalness / symmetry penalty -- the structural gap.
    # Adding a gait symmetry term or motion-capture reference
    # would constrain the solution to natural bipedal walking.

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
