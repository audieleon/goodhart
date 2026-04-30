"""Pan et al. 2022: Traffic Merge -- Misweighting (acceleration).

Source: Pan, Bhatia, & Steinhardt, "The Effects of Reward
  Misspecification: Mapping and Mitigating Misaligned Models,"
  ICLR 2022.
Failure: Agent merges aggressively with unsafe acceleration because
  the acceleration penalty is underweighted (0.1x true weight).
Mechanism: Misweighting -- proxy uses the same reward components as
  the true reward but with incorrect coefficients. The acceleration
  penalty is 10x too small, so the agent treats hard braking and
  rapid acceleration as nearly free.
Domain: Autonomous driving / traffic merge (FLOW simulator)

Encoding: PROXY reward (what the agent optimizes).
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Pan et al. 2022: Traffic Merge -- Misweighting (acceleration)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Traffic Merge - Accel Misweight (Pan et al. 2022)",
        max_steps=1500,
        gamma=0.99,
        n_actions=1,
        action_type="continuous",
        death_probability=0.0,
    )

    # Commute time penalty (per-step cost for being in traffic)
    model.add_reward_source(RewardSource(
        name="commute_time_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.1,
        requires_action=False,
        max_occurrences=0,
        intentional=True,
    ))

    # Acceleration penalty -- MISWEIGHTED (0.1x true weight)
    # True weight would be -1.0; proxy uses -0.1
    model.add_reward_source(RewardSource(
        name="acceleration_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.1,
        requires_action=True,
        max_occurrences=0,
        state_dependent=True,
        scales_with="acceleration",
        value_type="proportional",
        value_params={"scale": 0.1},
    ))

    # Successful merge completion
    model.add_reward_source(RewardSource(
        name="merge_complete",
        reward_type=RewardType.TERMINAL,
        value=10.0,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
