"""Nuclear microreactor drum control (Tunkle et al. 2025, U. Michigan).

Reward r = 2 - |power - target|. Simple tracking with positive
anchor. 8 control drums, Holos-Quad 22 MWt reactor.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "nuclear_microreactor",
    "source_paper": (
        'Tunkle, Abdulraheem, Lin & Radaideh, "Nuclear Microreactor '
        'Control with Deep Reinforcement Learning," '
        'arXiv:2504.00156, 2025'
    ),
    "paper_url": "https://arxiv.org/abs/2504.00156",
    "source_code_url": None,
    "reward_location": (
        "Section 3.1, Figure 2: r_t = 2 - |p_{t+1} - p*_{t+1}|. "
        "p = reactor power, p* = target power. Range [0,2] when "
        "error < 2. Actions: dtheta_1...dtheta_8 (drum rotation). "
        "Holos-Quad 22 MWt, point kinetics + thermal + xenon feedback."
    ),
    "year": 2025,
    "domain": "energy",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "RL controls 8 drum angles in a nuclear microreactor for "
        "load-following. Reward = 2 minus power tracking error. "
        "Simple, well-designed: positive anchor avoids reward desert, "
        "matches or exceeds PID performance."
    ),
    "documented_failure": (
        "None — well-designed reward. RL matches PID on short "
        "transients, maintains <1% error on 300-min load-following "
        "despite training only on short scenarios. Robust to noise."
    ),
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "medium",
    "is_negative_example": True,
    "encoding_rationale": {
        "positive_anchor": (
            "r = 2 - |error|. The +2 constant provides a positive "
            "anchor — unlike the nuclear plant paper (all-negative "
            "tracking), this reward is positive when error < 2. "
            "This avoids the reward desert problem entirely."
        ),
        "contrast_with_plant": (
            "Compare with nuclear_power_plant.py which uses "
            "r = -||p-a||^2 (all-negative). This paper's reward is "
            "better designed — same tracking objective, but with "
            "a positive anchor that gives the agent a clear signal."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # r = 2 - |p - p*|
    # 8 drum control, Holos-Quad 22 MWt

    model = EnvironmentModel(
        name="Nuclear Microreactor (Tunkle et al. 2025)",
        max_steps=300,  # 5-min steps, 25-hour transient
        gamma=0.99,
        n_actions=8,  # 8 drum rotation speeds
        action_type="continuous",
        n_states=100000,
        death_probability=0.0,
    )

    # Positive anchor: +2 constant
    model.add_reward_source(RewardSource(
        name="power_tracking_bonus",
        reward_type=RewardType.PER_STEP,
        value=2.0,  # constant positive anchor
        requires_action=False,  # always received
        intentional=True,
    ))

    # Tracking error penalty: -|p - p*|
    model.add_reward_source(RewardSource(
        name="power_error_penalty",
        reward_type=RewardType.PER_STEP,
        value=-1.0,  # average |error| (normalized)
        requires_action=True,
        state_dependent=True,
        scales_with="distance",
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
