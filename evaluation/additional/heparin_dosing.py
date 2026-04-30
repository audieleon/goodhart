"""Heparin dosing via offline RL (Lim et al. 2025, IEEE Access).

Reward = double-sigmoid targeting aPTT therapeutic range 60-100s.
+1 in range, -1 outside. ICU anticoagulation from MIMIC-III data.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "heparin_dosing",
    "source_paper": (
        'Lim, Park & Lee, "Development and Validation of Heparin '
        'Dosing Policies Using an Offline Reinforcement Learning '
        'Algorithm," IEEE Access 13, 2025'
    ),
    "paper_url": "https://arxiv.org/abs/2409.15753",
    "source_code_url": None,
    "reward_location": (
        "Section 3.3, Eq. 2: R_t = 2/(1+exp(-(aPTT-60))) "
        "- 2/(1+exp(-(aPTT-100))) - 1. Fig. 4: reward shape. "
        "+1 in therapeutic range 60-100s, -1 outside. "
        "6 discrete dose categories (Section 3.2). "
        "17 features from MIMIC-III (Table 1). BCQ algorithm."
    ),
    "year": 2025,
    "domain": "healthcare",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Offline RL for ICU heparin dosing. Double-sigmoid reward "
        "targets aPTT therapeutic range (60-100 seconds). +1 when "
        "in range, -1 when outside. Wrong dose risks stroke (underdose) "
        "or bleeding (overdose). 7,000 deaths/year in US from errors."
    ),
    "documented_failure": (
        "None — well-designed reward function (from Nemati et al. 2025). "
        "BCQ outperforms standard DRL by constraining to clinician-like "
        "actions. WIS evaluation confirms learning effectiveness."
    ),
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "low",
    "is_negative_example": True,
    "encoding_rationale": {
        "double_sigmoid": (
            "R = 2*sigmoid(aPTT-60) - 2*sigmoid(aPTT-100) - 1. "
            "This creates a smooth bump: +1 in [60,100], -1 outside. "
            "Both positive and negative values — NOT all-negative. "
            "goodhart should pass clean."
        ),
        "life_or_death": (
            "aPTT < 60: thrombosis risk (blood clots too easily). "
            "aPTT > 100: bleeding risk (blood doesn't clot). "
            "7,000 deaths/year in the US from anticoagulant errors."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Eq. 2: R = 2*sigmoid(aPTT-60) - 2*sigmoid(aPTT-100) - 1
    # Therapeutic range: aPTT 60-100 seconds

    model = EnvironmentModel(
        name="Heparin Dosing (Lim et al. 2025)",
        max_steps=72,  # up to 72 hours of ICU treatment
        gamma=0.99,
        n_actions=6,  # dose categories 0-5
        action_type="discrete",
        death_probability=0.01,
    )

    # In-range reward: +1 when aPTT in [60, 100]
    model.add_reward_source(RewardSource(
        name="therapeutic_range_reward",
        reward_type=RewardType.PER_STEP,
        value=1.0,  # +1 when in range
        requires_action=True,
        intentional=True,
        state_dependent=True,
        value_range=(-1.0, 1.0),  # double sigmoid range
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
