"""Pan et al. 2022: COVID Policy -- Misweighting (health cost).

Source: Pan, Bhatia, & Steinhardt, "The Effects of Reward
  Misspecification: Mapping and Mitigating Misaligned Models,"
  ICLR 2022.
Failure: Agent delays lockdown, causing excess deaths, because
  health cost (deaths) is underpenalized relative to economic cost.
  The proxy treats deaths as cheap compared to economic disruption.
Mechanism: Misweighting -- the proxy includes economic, health, and
  political costs but underweights the health component. The agent
  optimizes for economic output at the expense of human lives.
Domain: COVID-19 policy optimization (epidemic simulator)

Encoding: PROXY reward (what the agent optimizes).
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Pan et al. 2022: COVID Policy -- Misweighting (health cost)")
    print("=" * 70)

    model = EnvironmentModel(
        name="COVID Health Misweight (Pan et al. 2022)",
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

    # Health cost (deaths) -- MISWEIGHTED (underpenalized)
    # True weight would be -1.0; proxy uses -0.1
    model.add_reward_source(RewardSource(
        name="health_cost",
        reward_type=RewardType.PER_STEP,
        value=-0.1,
        requires_action=False,
        max_occurrences=0,
        state_dependent=True,
        scales_with="deaths",
        value_type="proportional",
        value_params={"scale": 0.1},
    ))

    # Political cost -- backlash from extreme policies
    model.add_reward_source(RewardSource(
        name="political_cost",
        reward_type=RewardType.PER_STEP,
        value=-0.5,
        requires_action=True,
        max_occurrences=0,
        state_dependent=True,
        scales_with="policy_extremity",
        value_type="proportional",
        value_params={"scale": 0.5},
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
