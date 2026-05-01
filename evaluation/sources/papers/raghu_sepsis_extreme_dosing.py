"""Raghu et al. 2017 sepsis treatment — extreme dosing failure.

DQN/Dueling DDQN trained on MIMIC-III learns to recommend extreme
vasopressor doses (3x frequency of max dose vs clinicians). The
reward (SOFA score improvement + survival) doesn't penalize dose
magnitude, so the agent maximizes expected outcome by pushing
doses to extremes.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "raghu_sepsis_extreme",
    "source_paper": (
        'Raghu, Komorowski, Celi, Szolovits & Ghassemi, "Deep '
        'Reinforcement Learning for Sepsis Treatment," NeurIPS '
        'ML4H Workshop 2017. Extreme dosing documented in: '
        'Li, Wang & Liu, "Optimizing Sepsis Treatment Strategies '
        'via a RL Model," Biomed Eng Letters 2023'
    ),
    "paper_url": "https://arxiv.org/abs/1711.09602",
    "source_code_url": None,
    "reward_location": (
        "Raghu Section 3: r_t based on SOFA score changes. "
        "r_terminal = +15 (survival) or -15 (death). Intermediate "
        "r_t = SOFA(t) - SOFA(t+1) (improvement in organ failure "
        "score). 25 discrete actions (5 IV fluid × 5 vasopressor "
        "dose bins). Li et al. 2023: Dueling DDQN gives 3x frequency "
        "of max-dose actions vs clinician policy."
    ),
    "year": 2017,
    "domain": "healthcare",
    "encoding_basis": "primary_source",
    "verification_date": "2026-05-01",
    "brief_summary": (
        "Sepsis treatment RL agent trained on ICU data recommends "
        "extreme vasopressor doses — 3x the frequency of maximum "
        "doses compared to clinicians. The reward (SOFA improvement "
        "+ survival bonus) doesn't penalize dose magnitude or "
        "rapid dose changes."
    ),
    "documented_failure": (
        "Li et al. 2023: Dueling DDQN shows nearly 3-fold increase "
        "in frequency of actions corresponding to highest levels of "
        "fluids and pressors. Rapid dose changes cause acute "
        "hypotension, hypertension, or cardiac arrhythmias. The "
        "reward function has no constraint on dose magnitude."
    ),
    "failure_mechanism": "ontological_proxy",
    "detection_type": "dynamic",
    "discovery_stage": "post_training",
    "fix_known": (
        "Li et al. 2023: Safe-Dueling DDQN hybrid with clinician "
        "constraints. Also: add dose-change penalty, action "
        "smoothness constraint, or use constrained RL with max "
        "dose rate as safety constraint."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "missing_dose_constraint": (
            "Reward is SOFA improvement + survival. No penalty for "
            "extreme doses or rapid dose changes. The agent correctly "
            "maximizes expected survival but via dangerous actions. "
            "advisory_missing_constraint should fire."
        ),
        "sparse_terminal": (
            "+15/-15 terminal reward for survival/death is sparse "
            "and delayed. SOFA intermediate reward is small per-step. "
            "This creates credit assignment challenges."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Raghu et al. 2017: Sepsis treatment from MIMIC-III
    # 25 discrete actions (5×5 dose grid), ~48h episodes
    model = EnvironmentModel(
        name="Sepsis Treatment — Extreme Dosing (Raghu 2017)",
        max_steps=48,
        gamma=0.99,
        n_states=750,
        n_actions=25,
        action_type="discrete",
        death_probability=0.1,
    )

    # SOFA score improvement: SOFA(t) - SOFA(t+1)
    # Small per-step, ~0-2 point changes
    model.add_reward_source(RewardSource(
        name="sofa_improvement",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Terminal: +15 survival, -15 death
    model.add_reward_source(RewardSource(
        name="survival_bonus",
        reward_type=RewardType.TERMINAL,
        value=15.0,
        state_dependent=True,
        requires_action=False,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
