"""Krakovna entry: Half Cheetah spinning in place.

Source: Zhang et al 2021
Failure: Half Cheetah exploits MuJoCo contact overflow to achieve
  extreme velocity by spinning in place rather than running forward.
Mechanism: Per-step velocity reward with no directional constraint.
  The reward is proportional to velocity magnitude, so spinning
  (which accumulates angular velocity via physics overflow) scores
  higher than forward locomotion.
Domain: Continuous control / locomotion

Structural encoding: We encode the velocity reward as a per-step
  signal that scales with velocity. The structural issue IS encodable:
  the reward tracks velocity magnitude without constraining direction.
  A well-designed reward would use forward velocity (dot product with
  heading), not raw speed.

Advisory: The MuJoCo contact overflow that enables extreme magnitudes
  is a physics/simulator issue. But even without the overflow, the
  lack of directional constraint is a structural reward design flaw.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Half Cheetah spinning (Zhang et al 2021)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Half Cheetah Spinning (Zhang et al 2021)",
        max_steps=1000,
        gamma=0.99,
        n_actions=6,
        action_type="continuous",
    )

    # Per-step velocity reward — no directional constraint.
    # The agent is rewarded for speed, not forward progress.
    # Structural flaw: velocity magnitude instead of forward velocity.
    model.add_reward_source(RewardSource(
        name="velocity_reward",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        value_range=(0.0, 100.0),
        value_type="proportional",
        value_params={"scale": 1.0},
        scales_with="velocity",
        state_dependent=True,
        requires_action=True,
        intentional=True,
        max_occurrences=0,
    ))

    # Small control cost penalty (standard in MuJoCo envs)
    model.add_reward_source(RewardSource(
        name="control_cost",
        reward_type=RewardType.PER_STEP,
        value=-0.1,
        requires_action=True,
        max_occurrences=0,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("The structural flaw is that velocity reward has no directional")
    print("constraint. The MuJoCo overflow amplifies the problem, but the")
    print("reward would incentivize spinning even without the overflow —")
    print("any movement that achieves high speed is rewarded equally.")


if __name__ == "__main__":
    run_example()
