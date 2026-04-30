"""Warfarin dosing via deep RL (Zadeh et al. 2023, J. Biomed. Inf.).

Reward = negative squared INR deviation from therapeutic range.
Agent adjusts anticoagulant dose to keep INR in 2.0-3.0.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "warfarin_dosing",
    "source_paper": (
        'Zadeh, Street & Thomas, "Optimizing Warfarin Dosing using '
        'Deep Reinforcement Learning," J. Biomedical Informatics, 2023'
    ),
    "paper_url": "https://arxiv.org/abs/2202.03486",
    "source_code_url": None,
    "reward_location": (
        "Section 3.2, Eq. 3: r = -c * E[sum((mu_m - mu_t)^2)]. "
        "mu_m=2.5 (midpoint INR range 2.0-3.0). c=4 (normalized so "
        "penalty=1.0 at range borders). Sum over tau dosing days. "
        "Eq. 2: actions d in {0.0, 0.5, ..., 15.0} mg/day."
    ),
    "year": 2023,
    "domain": "healthcare",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "RL agent prescribes daily warfarin doses to maintain patient's "
        "INR in the therapeutic range (2.0-3.0). Reward is negative "
        "squared deviation from range midpoint. Overdose risks fatal "
        "bleeding; underdose risks thromboembolism."
    ),
    "documented_failure": (
        "None — well-designed reward. Outperforms clinical dosing "
        "protocols on virtual patients via PK/PD simulation. The "
        "reward directly measures the clinical objective (INR in range)."
    ),
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "low",
    "is_negative_example": True,
    "encoding_rationale": {
        "tracking_reward": (
            "r = -4 * sum((2.5 - INR_t)^2). All-negative tracking "
            "controller — same pattern as nuclear plant. The gradient "
            "IS informative (closer to 2.5 = less penalty). goodhart "
            "should fire negative_only_reward as WARNING, not CRITICAL."
        ),
        "life_or_death": (
            "INR > 4: bleeding risk (potentially fatal). "
            "INR < 2: thromboembolism risk. Narrow therapeutic index. "
            "This is a safety-critical healthcare application."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Eq. 3: r = -c * E[sum((mu_m - mu_t)^2)]
    # c=4, mu_m=2.5, discrete doses 0-15mg in 0.5mg steps

    model = EnvironmentModel(
        name="Warfarin Dosing (Zadeh et al. 2023)",
        max_steps=90,  # ~90 day dosing trial
        gamma=0.99,
        n_actions=31,  # doses: 0.0, 0.5, ..., 15.0 mg/day
        action_type="discrete",
        death_probability=0.01,  # serious adverse events possible
    )

    # INR deviation penalty (Eq. 3)
    # r = -4 * (2.5 - INR_t)^2
    # Normalized: penalty = 1.0 at INR = 2.0 or 3.0 (range borders)
    model.add_reward_source(RewardSource(
        name="inr_deviation_penalty",
        reward_type=RewardType.PER_STEP,
        value=-4.0,  # c=4 normalization factor
        requires_action=True,
        intentional=True,
        state_dependent=True,
        scales_with="distance",
        value_type="proportional",
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
