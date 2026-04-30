"""Pan et al. 2022: Glucose Monitoring -- Ontological (risk function).

Source: Pan, Bhatia, & Steinhardt, "The Effects of Reward
  Misspecification: Mapping and Mitigating Misaligned Models,"
  ICLR 2022.
Failure: Agent over-administers insulin because the proxy risk
  function penalizes high glucose more aggressively than the true
  holistic health risk. The proxy's asymmetric risk function drives
  the agent to push glucose dangerously low (hypoglycemia) to avoid
  the heavily-penalized high-glucose region.
Mechanism: Ontological -- the proxy uses a different risk function
  than the true health objective. The shapes of the penalty curves
  differ: the proxy over-penalizes hyperglycemia relative to
  hypoglycemia, causing the agent to prefer dangerous insulin
  over-dosing.
Domain: Glucose monitoring / insulin delivery (medical simulator)

Encoding: PROXY reward (what the agent optimizes).
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Pan et al. 2022: Glucose Monitoring -- Ontological")
    print("=" * 70)

    model = EnvironmentModel(
        name="Glucose Monitoring (Pan et al. 2022)",
        max_steps=480,
        gamma=0.99,
        n_actions=1,
        action_type="continuous",
        death_probability=0.0,
    )

    # Glucose risk penalty -- ONTOLOGICALLY WRONG risk function
    # True reward uses a symmetric/holistic risk that equally penalizes
    # hypo- and hyperglycemia. The proxy uses an asymmetric function
    # that heavily penalizes high glucose, driving over-treatment.
    model.add_reward_source(RewardSource(
        name="glucose_risk",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
        requires_action=False,
        max_occurrences=0,
        state_dependent=True,
        scales_with="glucose_level",
        value_type="exponential",
        value_params={"sigma": 0.25},
        intentional=True,
    ))

    # Insulin action cost -- small penalty for administering insulin
    model.add_reward_source(RewardSource(
        name="insulin_cost",
        reward_type=RewardType.PER_STEP,
        value=-0.05,
        requires_action=True,
        max_occurrences=0,
        state_dependent=True,
        scales_with="insulin_dose",
        value_type="proportional",
        value_params={"scale": 0.05},
    ))

    # Target range bonus -- reward for glucose in safe range
    model.add_reward_source(RewardSource(
        name="target_range_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        requires_action=False,
        max_occurrences=0,
        state_dependent=True,
        scales_with="glucose_in_range",
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
