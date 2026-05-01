"""Tokamak plasma control via RL (Degrave et al. 2022, Nature).

Modular reward: individual quality measures for plasma current,
position, shape → combined via nonlinear combiners. Max 100/sec.
Deployed on TCV tokamak. DeepMind/EPFL collaboration.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "tokamak_degrave",
    "source_paper": (
        'Degrave, Felici, Buchli, Neunert et al., "Magnetic control of '
        'tokamak plasmas through deep reinforcement learning," '
        'Nature 602, 414-419, 2022'
    ),
    "paper_url": "https://www.nature.com/articles/s41586-021-04301-9",
    "source_code_url": None,
    "reward_location": (
        "Methods, Rewards and terminations section (page 8): "
        "individual objectives → non-linear quality [0,1] → combiner → "
        "scalar reward. Normalized to max 100 per 1s of control. "
        "Termination → large negative reward. Quality via softplus "
        "or sigmoid. Combiner via weighted smooth max or geometric mean. "
        "Full definitions in Extended Data Tables 3-5."
    ),
    "year": 2022,
    "domain": "fusion",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "RL controls 19 magnetic coils of TCV tokamak to achieve target "
        "plasma configurations. Modular reward combines quality measures "
        "for current, position, and shape. Zero-shot sim-to-real on "
        "real fusion reactor. One of the most complex real-world RL "
        "deployments ever."
    ),
    "documented_failure": (
        "None for the RL reward — the reward design is well-crafted. "
        "The paper that already exists in our dataset (tokamak_plasma.py) "
        "documents the MISSING CONSTRAINT failure: coil balance term "
        "absent, causing asymmetric electromagnetic forces. This entry "
        "encodes the full Degrave reward architecture as published."
    ),
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "post_deployment",
    "fix_known": None,
    "compute_cost_class": "extreme",
    "is_negative_example": True,
    "encoding_rationale": {
        "modular_architecture": (
            "Individual objectives (Ip tracking, position, shape) are "
            "each converted to quality measures via softplus/sigmoid, "
            "then combined through weighted nonlinear combiners. This "
            "modular design allows easy reconfiguration for new plasma "
            "shapes without redesigning the reward."
        ),
        "normalized_reward": (
            "Max cumulative reward = 100 per second. This normalization "
            "ensures rewards are comparable across different plasma "
            "configurations and episode lengths."
        ),
        "termination_penalty": (
            "Large negative reward on termination (hitting operational "
            "limits). Teaches the agent to avoid unsafe states. "
            "Termination conditions in Extended Data Table 5."
        ),
        "contrast_with_existing": (
            "Our existing tokamak_plasma.py encodes a simplified "
            "3-source version showing the MISSING CONSTRAINT pattern. "
            "This entry encodes the full Degrave architecture from "
            "the Nature paper as a well-designed negative example."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Modular reward architecture
    # 19 actuated coils, 10 kHz control, MPO algorithm
    # Quality measures combined via weighted nonlinear combiners

    model = EnvironmentModel(
        name="Tokamak TCV Plasma Control (Degrave et al. 2022)",
        max_steps=10000,  # ~1s at 10 kHz
        gamma=0.99,
        n_actions=19,  # 19 TCV control coils
        action_type="continuous",
        n_states=100000,  # 92 sensor inputs
        death_probability=0.01,  # termination on operational limits
    )

    # Plasma current tracking (Ip)
    # Quality = softplus/sigmoid(|Ip - Ip_target|)
    model.add_reward_source(RewardSource(
        name="current_tracking",
        reward_type=RewardType.PER_STEP,
        value=30.0,  # ~30% of max 100/s reward budget
        requires_action=True,
        intentional=True,
        state_dependent=True,
        scales_with="distance",
    ))

    # Plasma position (R, Z axis)
    model.add_reward_source(RewardSource(
        name="position_tracking",
        reward_type=RewardType.PER_STEP,
        value=30.0,  # ~30% of reward budget
        requires_action=True,
        intentional=True,
        state_dependent=True,
        scales_with="distance",
    ))

    # Plasma shape (boundary points, elongation, triangularity)
    model.add_reward_source(RewardSource(
        name="shape_tracking",
        reward_type=RewardType.PER_STEP,
        value=30.0,  # ~30% of reward budget
        requires_action=True,
        intentional=True,
        state_dependent=True,
        scales_with="distance",
    ))

    # Termination penalty (operational limits exceeded)
    model.add_reward_source(RewardSource(
        name="termination_penalty",
        reward_type=RewardType.TERMINAL,
        value=-100.0,  # large negative on termination
        requires_action=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
