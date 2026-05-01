"""Lazic et al. 2022 — data center cooling reward delay (Google/DeepMind).

Optimizing data center PUE (Power Usage Effectiveness) with RL.
The reward (negative PUE) has significant delay — HVAC changes
take 10-15 minutes to affect temperatures, but reward is
computed every 5 minutes. Credit assignment across thermal lag.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "lazic_datacenter_delay",
    "source_paper": (
        'Lazic, Boutilier, Lu, Wong, Roy, Ryu & Imwalle, "Data '
        'Center Cooling Using Model-Predictive Control," NeurIPS '
        '2018. Also: Evans & Gao, "DeepMind AI Reduces Google '
        'Data Centre Cooling Bill by 40%," DeepMind Blog 2016'
    ),
    "paper_url": "https://arxiv.org/abs/1805.00541",
    "source_code_url": None,
    "reward_location": (
        "Section 3: reward r_t = -PUE_t (negative power usage "
        "effectiveness). PUE = total facility power / IT equipment "
        "power. Range [1.0, 2.0+]. Actions: 19 continuous setpoints "
        "(chiller temps, air handler configs). Key challenge: thermal "
        "lag means PUE changes take 10-15 minutes after action."
    ),
    "year": 2018,
    "domain": "energy",
    "encoding_basis": "primary_source",
    "verification_date": "2026-05-01",
    "brief_summary": (
        "RL optimizes data center cooling by minimizing PUE. The "
        "thermal lag (10-15 min) between actions and effects creates "
        "a credit assignment challenge — the agent must attribute "
        "PUE changes to actions taken several steps earlier."
    ),
    "documented_failure": (
        "Section 4: model-free RL struggles with the thermal delay. "
        "The authors switched to model-predictive control (MPC) "
        "because pure RL couldn't handle the credit assignment "
        "across the thermal lag. The reward delay is inherent to "
        "the physics of HVAC systems."
    ),
    "failure_mechanism": "credit_assignment",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Model-predictive control (MPC) with learned dynamics model. "
        "The model handles the thermal delay explicitly rather than "
        "relying on RL credit assignment."
    ),
    "compute_cost_class": "high",
    "is_negative_example": False,
    "encoding_rationale": {
        "advisory_credit_assignment": (
            "PUE changes lag actions by 2-3 steps (10-15 min vs "
            "5 min decision frequency). advisory_credit_assignment "
            "should fire."
        ),
        "negative_only": (
            "Reward is -PUE, always negative (PUE >= 1.0). "
            "negative_only_reward should fire."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    model = EnvironmentModel(
        name="Data Center Cooling — PUE Delay (Lazic 2018)",
        max_steps=288,
        gamma=0.99,
        n_states=100000,
        n_actions=19,
        action_type="continuous",
        death_probability=0.0,
    )

    # -PUE: always negative, lower is better
    # Typical PUE: 1.1-1.3 for efficient data centers
    model.add_reward_source(RewardSource(
        name="negative_pue",
        reward_type=RewardType.PER_STEP,
        value=-1.2,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
