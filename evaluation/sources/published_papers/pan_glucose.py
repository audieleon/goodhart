"""Pan et al. 2022: Glucose monitoring ontological (risk vs cost).

Proxy uses Magni risk function (glycemic risk only) instead of the
true reward that includes both blood-glucose risk and insulin cost.
Agent over-administers insulin because the proxy ignores treatment cost.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "pan_glucose",
    "source_paper": (
        'Pan, Bhatia & Steinhardt, "The Effects of Reward'
        ' Misspecification: Mapping and Mitigating Misaligned'
        ' Models," ICLR 2022'
    ),
    "paper_url": "https://arxiv.org/abs/2201.03544",
    "source_code_url": "https://github.com/aypan17/reward-misspecification",
    "reward_location": "Table 1 row 9; code: glucose/rewards.py",
    "year": 2022,
    "domain": "healthcare",
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Glucose proxy uses Magni risk function"
        " R = -10*(3.5506*(ln(bg)^0.8353 - 3.7932))^2,"
        " omitting the insulin cost term. True reward is"
        " R = -10*fBG^2 - 25*insulin[-1]. Agent over-administers"
        " insulin because treatment cost is invisible."
    ),
    "documented_failure": (
        "Larger models administer more insulin, reducing glycemic"
        " risk but increasing treatment cost. The true reward"
        " (which includes insulin cost) is actually worse for"
        " larger models."
    ),
    "failure_mechanism": "ontological",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Use true reward: -10*fBG^2 - 25*insulin[-1], which"
        " includes both glycemic risk and insulin cost."
    ),
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "requires_action": (
            "Insulin administration is the action."
            " Glucose risk is state-dependent."
        ),
        "intentional": (
            "Magni risk function is the proxy objective"
            " (intentional but wrong -- omits insulin cost)."
        ),
        "can_loop": "No looping; sequential insulin decisions.",
        "value": (
            "Proxy: -10*(3.5506*(ln(bg)^0.8353 - 3.7932))^2."
            " True: -10*fBG^2 - 25*insulin[-1]."
            " Proxy omits the -25*insulin cost term."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Pan et al. 2022: Glucose Monitoring -- Ontological")
    print("=" * 70)

    # True (magni_bg_insulin_true): R = -10*fBG^2 - 25*insulin[-1]
    # Proxy (magni_reward): R = -10*(3.5506*(ln(bg)^0.8353 - 3.7932))^2
    #   (omits insulin cost entirely)
    # gamma=0.99, max_steps=2880 (10 days at 5-min intervals),
    # n_actions=1, continuous

    model = EnvironmentModel(
        name="Glucose Monitoring (Pan et al. 2022)",
        max_steps=2880,
        gamma=0.99,
        n_actions=1,
        action_type="continuous",
        death_probability=0.0,
    )

    # Magni glycemic risk -- proxy objective
    # Uses a nonlinear transformation of blood glucose:
    # R = -10 * (3.5506 * (ln(bg)^0.8353 - 3.7932))^2
    # This penalizes deviation from safe glucose range but
    # uses a different risk function than the true reward.
    model.add_reward_source(RewardSource(
        name="magni_glycemic_risk",
        reward_type=RewardType.PER_STEP,
        value=-10.0,
        requires_action=False,
        max_occurrences=0,
        state_dependent=True,
        scales_with="blood_glucose",
        value_type="exponential",
        value_params={"sigma": 0.25},
        intentional=True,
    ))

    # NOTE: True reward includes -25*insulin[-1] (insulin cost).
    # The proxy omits this term entirely, so the agent has no
    # incentive to minimize insulin administration.

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
