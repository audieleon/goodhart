"""Prasad et al. 2017 — ventilator weaning reward (clinician divergence).

RL agent learns mechanical ventilation weaning policy from ICU data.
Reward based on successful extubation vs reintubation. Agent
recommends earlier weaning than clinicians — optimizing for short
ventilation duration but risking premature extubation.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "prasad_ventilator_weaning",
    "source_paper": (
        'Prasad, Laszewski, Atanasov, "A Reinforcement Learning '
        'Approach to Weaning of Mechanical Ventilation in Intensive '
        'Care Units," UAI 2017. Also: Yu, Liu & Nemati, "RL in '
        'Healthcare: A Survey," ACM Computing Surveys 2021'
    ),
    "paper_url": "https://arxiv.org/abs/1704.06300",
    "source_code_url": None,
    "reward_location": (
        "Section 3: reward r = +1 for successful extubation "
        "(patient breathes independently), -1 for reintubation "
        "(patient needs ventilator again), -0.01 per step "
        "(encourages faster weaning). Actions: ventilator settings "
        "(FiO2, PEEP, tidal volume)."
    ),
    "year": 2017,
    "domain": "healthcare",
    "encoding_basis": "primary_source",
    "verification_date": "2026-05-01",
    "brief_summary": (
        "Ventilator weaning agent rewarded for successful extubation "
        "with per-step penalty encouraging speed. Agent learns to "
        "wean earlier than clinicians, risking premature extubation "
        "and reintubation — the time penalty biases toward speed "
        "over safety."
    ),
    "documented_failure": (
        "Yu et al. 2021 survey: RL ventilator policies tend to "
        "diverge significantly from clinician behavior, recommending "
        "earlier weaning. The -0.01/step penalty incentivizes speed "
        "at the cost of safety. Reintubation (requiring re-ventilation) "
        "has serious patient consequences beyond the -1 penalty."
    ),
    "failure_mechanism": "ontological_proxy",
    "detection_type": "structural",
    "discovery_stage": "post_training",
    "fix_known": (
        "Increase reintubation penalty (currently only -1, same "
        "magnitude as success). Add intermediate physiological "
        "monitoring rewards. Use offline safe RL (conservative "
        "Q-learning) to stay near clinician policy."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "time_vs_safety": (
            "Per-step penalty (-0.01) × 500 steps = -5.0. "
            "Reintubation penalty is only -1.0. After 100 steps, "
            "the accumulated time penalty exceeds the reintubation "
            "cost, making premature extubation preferable."
        ),
        "sparse_binary": (
            "+1/-1 terminal is too coarse. Reintubation should "
            "cost much more than failed-to-extubate since it "
            "has serious clinical consequences."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    model = EnvironmentModel(
        name="Ventilator Weaning — Speed vs Safety (Prasad 2017)",
        max_steps=500,
        gamma=0.99,
        n_states=5000,
        n_actions=10,
        action_type="discrete",
        death_probability=0.02,
    )

    # Per-step penalty (encourages fast weaning)
    model.add_reward_source(RewardSource(
        name="time_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.01,
        state_dependent=False,
        requires_action=False,
        intentional=True,
    ))

    # Successful extubation: +1
    model.add_reward_source(RewardSource(
        name="extubation_success",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Reintubation: -1
    model.add_reward_source(RewardSource(
        name="reintubation_failure",
        reward_type=RewardType.TERMINAL,
        value=-1.0,
        state_dependent=True,
        requires_action=False,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
