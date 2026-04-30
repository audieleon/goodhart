"""Pan et al. 2022: COVID Policy -- Ontological (missing political cost).

Source: Pan, Bhatia, & Steinhardt, "The Effects of Reward
  Misspecification: Mapping and Mitigating Misaligned Models,"
  ICLR 2022.
Failure: Agent imposes politically costly early lockdowns because
  the proxy omits political cost entirely. Without a term penalizing
  political backlash, the agent has no reason to avoid unpopular
  policies that are economically and epidemiologically rational but
  politically infeasible.
Mechanism: Ontological -- the proxy omits an entire reward dimension
  (political cost). The true reward is a three-way tradeoff
  (economic + health + political); the proxy is only two-way
  (economic + health), leading to policies that are technically
  optimal but politically unacceptable.
Domain: COVID-19 policy optimization (epidemic simulator)

Encoding: PROXY reward (what the agent optimizes).
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Pan et al. 2022: COVID Policy -- Ontological (missing political)")
    print("=" * 70)

    model = EnvironmentModel(
        name="COVID Missing Political (Pan et al. 2022)",
        max_steps=365,
        gamma=0.99,
        n_actions=5,
        action_type="continuous",
        death_probability=0.0,
    )

    # Economic cost -- penalizes lockdown measures
    model.add_reward_source(RewardSource(
        name="economic_cost",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
        requires_action=True,
        max_occurrences=0,
        state_dependent=True,
        scales_with="lockdown_severity",
        value_type="proportional",
        value_params={"scale": 1.0},
    ))

    # Health cost (deaths)
    model.add_reward_source(RewardSource(
        name="health_cost",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
        requires_action=False,
        max_occurrences=0,
        state_dependent=True,
        scales_with="deaths",
        value_type="proportional",
        value_params={"scale": 1.0},
    ))

    # NOTE: Political cost is MISSING from this proxy.
    # The true reward includes a third term for political backlash,
    # but the proxy omits it entirely. This is the ontological gap.

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
