"""Satheesh & Powell 2025 — constrained MARL for traffic signals.

Reward: lambda_f*R_f + lambda_w*R_w (vehicles flowing + waiting).
Three constraints via Lagrange multipliers: GreenTime (max green
duration), PhaseSkip (no skipping phases), GreenSkip (no skipping
lights). Without constraints, agents learn unfair policies that
starve some directions.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "satheesh_traffic_constrained",
    "source_paper": (
        'Satheesh & Powell, "A Constrained Multi-Agent Reinforcement '
        'Learning Approach to Autonomous Traffic Signal Control," '
        'arXiv 2025 (U. Maryland)'
    ),
    "paper_url": "https://arxiv.org/abs/2503.23626",
    "source_code_url": "https://github.com/Asatheesh6561/MAPPO",
    "reward_location": (
        "Section 3.1, page 4: r = lambda_f * R_f + lambda_w * R_w. "
        "R_f = total vehicles moving, R_w = total vehicles waiting. "
        "Section 3.2: 3 constraints — GreenTime (Eq. 1): G_time(l) "
        "<= G_max_time. PhaseSkip (Eq. 2): P_skips(p) <= P_max_skips. "
        "GreenSkip (Eq. 3): G_skips(l) <= G_max_skips. Constraint "
        "cost (Eq. 4): averaged indicator across agents and lights. "
        "Constrained Markov Game formulation, page 4."
    ),
    "year": 2025,
    "domain": "multi_agent",
    "encoding_basis": "primary_source",
    "verification_date": "2026-05-01",
    "brief_summary": (
        "Multi-agent traffic signal control with reward for vehicle "
        "throughput. Without constraints, agents learn unfair "
        "policies (starving some directions). Three fairness "
        "constraints enforced via Lagrange multipliers."
    ),
    "documented_failure": (
        "Section 1-2: unconstrained MARL agents learn policies that "
        "violate real-world traffic rules — keeping lights green "
        "indefinitely, skipping phases, starving lanes. The reward "
        "(throughput) alone does not prevent these violations."
    ),
    "failure_mechanism": "ontological_proxy",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "MAPPO-LCE: Lagrange multiplier method with cost estimator. "
        "Improves on MAPPO by 12.6%, IPPO by 10.3%, QTRAN by 13.1% "
        "across three real-world traffic datasets."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "advisory_constrained_rl": (
            "The paper explicitly uses constrained RL (Lagrangian) "
            "because unconstrained reward optimization produces "
            "unfair/illegal traffic policies. We encode the "
            "unconstrained version to show what goodhart catches."
        ),
        "multi_agent_nonstationarity": (
            "Each agent controls one intersection. Other agents' "
            "changing policies make each agent's environment "
            "non-stationary."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Satheesh & Powell 2025: MAPPO-LCE for traffic
    # Multi-agent, each agent controls 1 intersection with 8 phases
    model = EnvironmentModel(
        name="MAPPO Traffic Signals — Unconstrained (Satheesh 2025)",
        max_steps=500,
        gamma=0.99,
        n_states=100000,
        n_actions=8,
        action_type="discrete",
        death_probability=0.0,
    )

    # R_f: vehicles flowing (positive, higher is better)
    model.add_reward_source(RewardSource(
        name="vehicles_flowing",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # R_w: vehicles waiting (negative, penalizes congestion)
    model.add_reward_source(RewardSource(
        name="vehicles_waiting",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
