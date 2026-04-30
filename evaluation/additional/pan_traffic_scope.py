"""Pan et al. 2022: Traffic Merge -- Scope (partial monitoring).

Source: Pan, Bhatia, & Steinhardt, "The Effects of Reward
  Misspecification: Mapping and Mitigating Misaligned Models,"
  ICLR 2022.
Failure: Agent optimizes velocity only near the merge point,
  ignoring downstream congestion. The proxy monitors a subset
  of the road network, so the agent has no incentive to prevent
  bottlenecks that form outside the monitored region.
Mechanism: Scope -- the proxy measures the right quantity (velocity)
  but over an incomplete spatial scope. Downstream effects are
  invisible to the reward, so the agent freely externalizes
  congestion outside the monitored zone.
Domain: Autonomous driving / traffic merge (FLOW simulator)

Encoding: PROXY reward (what the agent optimizes).
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Pan et al. 2022: Traffic Merge -- Scope (partial monitoring)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Traffic Merge - Partial Scope (Pan et al. 2022)",
        max_steps=1500,
        gamma=0.99,
        n_actions=1,
        action_type="continuous",
        death_probability=0.0,
    )

    # Velocity near merge point only -- SCOPED proxy
    # True reward would monitor velocity across all roads.
    # Proxy only measures velocity in the merge zone, so
    # the agent ignores downstream congestion entirely.
    model.add_reward_source(RewardSource(
        name="merge_zone_velocity",
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
