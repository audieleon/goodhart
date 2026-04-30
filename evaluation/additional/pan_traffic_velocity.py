"""Pan et al. 2022: Traffic Merge -- Ontological (velocity proxy).

Source: Pan, Bhatia, & Steinhardt, "The Effects of Reward
  Misspecification: Mapping and Mitigating Misaligned Models,"
  ICLR 2022.
Failure: AVs block the merge ramp to keep highway velocity high
  rather than actually minimizing commute time. Velocity is not
  equivalent to fast arrival when there is traffic -- blocking
  merging vehicles keeps the highway clear but increases total
  system commute time.
Mechanism: Ontological -- the proxy replaces "minimize commute time"
  with "maximize velocity," a different quantity that diverges from
  the true objective in congested multi-agent settings.
Domain: Autonomous driving / traffic merge (FLOW simulator)

Encoding: PROXY reward (what the agent optimizes).
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Pan et al. 2022: Traffic Merge -- Ontological (velocity proxy)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Traffic Merge - Velocity Proxy (Pan et al. 2022)",
        max_steps=1500,
        gamma=0.99,
        n_actions=1,
        action_type="continuous",
        death_probability=0.0,
    )

    # Velocity reward -- the WRONG objective
    # Agent maximizes this instead of minimizing commute time.
    # This is ontologically different: high velocity != fast arrival
    # in congested traffic.
    model.add_reward_source(RewardSource(
        name="velocity_reward",
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

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
