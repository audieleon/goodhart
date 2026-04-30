"""ICU ventilator optimization EZ-Vent (Liu et al. 2024, JMIR).

Terminal survival/death reward + intermittent SpO2 and MBP
tracking. 18 discrete actions for PEEP, FiO2, tidal volume.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "ventilator_ezvent",
    "source_paper": (
        'Liu, Xu, Xu, Liu, Sun, Xie, Feng & See, "Reinforcement '
        'Learning to Optimize Ventilator Settings for Patients on '
        'Invasive Mechanical Ventilation: Retrospective Study," '
        'JMIR 26, e44494, 2024'
    ),
    "paper_url": "https://www.jmir.org/2024/1/e44494",
    "source_code_url": None,
    "reward_location": (
        "Page 5-6: Terminal: R = +a (survive), -a/2 (death). "
        "Intermittent: R_im = +b if 94<=SpO2<=98 else -b/2; "
        "+c if 70<=MBP<=80 else -c/2. a, b, c tuned. "
        "18 actions (3x3x2: PEEP, FiO2, tidal volume). "
        "40 features, BCQ, gamma=0.99. MIMIC-IV + eICU."
    ),
    "year": 2024,
    "domain": "healthcare",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "RL optimizes ICU ventilator settings (PEEP, FiO2, tidal volume) "
        "for mechanically ventilated patients. Terminal reward for "
        "survival, intermittent rewards for SpO2 and blood pressure "
        "in optimal ranges. 26,700 patients across two ICU databases."
    ),
    "documented_failure": (
        "None — well-designed reward. Estimated 12.1% mortality with "
        "AI policy vs 30.6% physician policy on MIMIC-IV. Higher "
        "proportion of optimal SpO2 (49% vs 40.5%) and MBP (41.2% "
        "vs 37.1%)."
    ),
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "medium",
    "is_negative_example": True,
    "encoding_rationale": {
        "three_component": (
            "Terminal survival (+a), SpO2 range (+b/-b/2), MBP range "
            "(+c/-c/2). Both positive and negative values — not all "
            "negative. The range-based rewards create clear targets."
        ),
        "asymmetric_penalty": (
            "Penalty for out-of-range (-b/2, -c/2) is half the "
            "reward magnitude, and death penalty (-a/2) is half the "
            "survival reward. This asymmetry encourages maintaining "
            "range rather than avoiding it."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Terminal survival + intermittent SpO2/MBP tracking
    # 18 actions, 40 features, BCQ, gamma=0.99

    model = EnvironmentModel(
        name="EZ-Vent ICU Ventilator (Liu et al. 2024)",
        max_steps=12,  # 48 hours in 4-hour steps
        gamma=0.99,
        n_actions=18,  # 3 PEEP × 3 FiO2 × 2 tidal volume levels
        action_type="discrete",
        death_probability=0.2,  # ICU mortality ~18-31%
    )

    # Terminal: survival reward
    model.add_reward_source(RewardSource(
        name="survival_reward",
        reward_type=RewardType.TERMINAL,
        value=15.0,  # +a for survival
        requires_action=True,
        intentional=True,
        discovery_probability=0.7,  # ~70% survive
    ))

    # Terminal: death penalty
    model.add_reward_source(RewardSource(
        name="death_penalty",
        reward_type=RewardType.TERMINAL,
        value=-7.5,  # -a/2 for death
        requires_action=True,
    ))

    # Intermittent: SpO2 in optimal range 94-98%
    model.add_reward_source(RewardSource(
        name="spo2_tracking",
        reward_type=RewardType.PER_STEP,
        value=1.0,  # +b in range, -b/2 out
        requires_action=True,
        state_dependent=True,
        value_range=(-0.5, 1.0),
    ))

    # Intermittent: MBP in optimal range 70-80 mmHg
    model.add_reward_source(RewardSource(
        name="mbp_tracking",
        reward_type=RewardType.PER_STEP,
        value=1.0,  # +c in range, -c/2 out
        requires_action=True,
        state_dependent=True,
        value_range=(-0.5, 1.0),
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
