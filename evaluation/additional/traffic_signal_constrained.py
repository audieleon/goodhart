"""Constrained traffic signal control (Satheesh & Powell 2025).

Reward = lambda_f*vehicles_moving + lambda_w*vehicles_waiting.
Three fairness constraints via Lagrange multipliers: GreenTime,
PhaseSkip, GreenSkip. Multi-agent, one agent per intersection.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "traffic_signal_constrained",
    "source_paper": (
        'Satheesh & Powell, "A Constrained Multi-Agent Reinforcement '
        'Learning Approach to Autonomous Traffic Signal Control," '
        'arXiv:2503.23626, 2025'
    ),
    "paper_url": "https://arxiv.org/abs/2503.23626",
    "source_code_url": "https://github.com/Asatheesh6561/MAPPO",
    "reward_location": (
        "Section 3.1 page 4: R = lambda_f*R_f + lambda_w*R_w. "
        "R_f = vehicles moving, R_w = vehicles waiting. "
        "Eq. 1-3: 3 constraints (GreenTime, PhaseSkip, GreenSkip). "
        "Eq. 4: constraint penalty = avg indicator across lights. "
        "8 phases per intersection, 12 lights, MAPPO-LCE."
    ),
    "year": 2025,
    "domain": "multi_agent",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Multi-intersection traffic signal control with fairness "
        "constraints. Reward maximizes vehicle throughput while "
        "Lagrangian constraints enforce max green time, prevent "
        "phase skipping, and ensure all lanes get served."
    ),
    "documented_failure": (
        "Unconstrained MAPPO violates fairness: some directions get "
        "perpetual green while others starve. Constrained version "
        "(MAPPO-LCE) improves by 12.60% over unconstrained MAPPO."
    ),
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Add Lagrangian constraints for GreenTime, PhaseSkip, GreenSkip. "
        "MAPPO-LCE outperforms MAPPO by 12.60%, IPPO by 10.29%."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": True,
    "encoding_rationale": {
        "throughput_reward": (
            "R_f (vehicles moving) is positive, R_w (vehicles waiting) "
            "is negative. Both are state-dependent and require action "
            "(signal phase selection determines who moves/waits)."
        ),
        "constraints_as_penalties": (
            "Three constraint costs enforced via Lagrange multipliers. "
            "Modeled as ON_EVENT penalties. Without constraints, agent "
            "gives perpetual green to high-traffic directions."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # R = lambda_f * R_f + lambda_w * R_w
    # 3 constraints via Lagrange multipliers

    model = EnvironmentModel(
        name="Traffic Signal MAPPO-LCE (Satheesh & Powell 2025)",
        max_steps=500,
        gamma=0.99,
        n_actions=8,  # 8 phases per intersection
        action_type="discrete",
        death_probability=0.0,
    )

    # Vehicles moving (positive reward)
    model.add_reward_source(RewardSource(
        name="vehicles_moving",
        reward_type=RewardType.PER_STEP,
        value=1.0,  # lambda_f * R_f
        requires_action=True,
        intentional=True,
        state_dependent=True,
    ))

    # Vehicles waiting (negative component)
    model.add_reward_source(RewardSource(
        name="vehicles_waiting",
        reward_type=RewardType.PER_STEP,
        value=-0.5,  # lambda_w * R_w
        requires_action=True,
        state_dependent=True,
    ))

    # GreenTime constraint (Eq. 1)
    model.add_reward_source(RewardSource(
        name="greentime_constraint",
        reward_type=RewardType.ON_EVENT,
        value=-0.1,  # Lagrange penalty per violation
        requires_action=True,
    ))

    # PhaseSkip constraint (Eq. 2)
    model.add_reward_source(RewardSource(
        name="phaseskip_constraint",
        reward_type=RewardType.ON_EVENT,
        value=-0.1,
        requires_action=True,
    ))

    # GreenSkip constraint (Eq. 3)
    model.add_reward_source(RewardSource(
        name="greenskip_constraint",
        reward_type=RewardType.ON_EVENT,
        value=-0.1,
        requires_action=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
