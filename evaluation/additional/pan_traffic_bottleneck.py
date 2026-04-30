"""Pan et al. 2022: Traffic Bottleneck -- Misweighting (acceleration).

Source: Pan, Bhatia, & Steinhardt, "The Effects of Reward
  Misspecification: Mapping and Mitigating Misaligned Models,"
  ICLR 2022.
Failure: Agent accelerates aggressively near the bottleneck because
  the acceleration penalty is underweighted. This improves throughput
  locally but creates unsafe driving behavior.
Mechanism: Misweighting -- same components as the true reward, but
  the acceleration penalty coefficient is too small. The agent
  sacrifices safety for throughput in the bottleneck zone.
Domain: Autonomous driving / traffic bottleneck (FLOW simulator)

Encoding: PROXY reward (what the agent optimizes).
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Pan et al. 2022: Traffic Bottleneck -- Misweighting")
    print("=" * 70)

    model = EnvironmentModel(
        name="Traffic Bottleneck - Accel Misweight (Pan et al. 2022)",
        max_steps=1500,
        gamma=0.99,
        n_actions=1,
        action_type="continuous",
        death_probability=0.0,
    )

    # Throughput reward -- vehicles passing through bottleneck
    model.add_reward_source(RewardSource(
        name="throughput",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=True,
        max_occurrences=0,
        state_dependent=True,
        scales_with="velocity",
        value_type="proportional",
        value_params={"scale": 1.0},
        intentional=True,
    ))

    # Acceleration penalty -- MISWEIGHTED (underpenalized)
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

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
