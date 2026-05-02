"""Example: Constrained RL -- CPO/FOCOPS (LIMITATION).

Soft reward penalties are not equivalent to hard constraint optimization.
Source: Achiam et al. 2017 (CPO, ICML), Zhang et al. 2020 (FOCOPS), Ray et al. 2019 (Safety Gym)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "safety_constrained",
    "source_paper": "Achiam et al. 2017 (CPO, ICML), Zhang et al. 2020 (FOCOPS), Ray et al. 2019 (Safety Gym)",
    "paper_url": "https://arxiv.org/abs/1705.10528",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2017,
    "domain": "safety",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to maximize reward subject to safety constraints. Tool models costs as penalties, but this is not equivalent to constrained optimization.",
    "documented_failure": "goodhart models costs as reward penalties, but a soft penalty allows trading safety for reward while a hard constraint does not. CPO/FOCOPS enforce constraints the tool cannot represent.",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "CostSource model, constraint budgets, and Lagrangian dual analysis would be needed",
    "compute_cost_class": "medium",
    "is_negative_example": True,
    "encoding_rationale": {
        "limitation_example": "Penalty strength is arbitrary; real constrained RL uses learned Lagrange multiplier",
        "hard_vs_soft": "A budget of 25 violations is not the same as -0.5 per violation",
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Constrained RL (CPO/FOCOPS) — (TOOL LIMITATION)")
    print("=" * 70)
    print()
    print("Source: Achiam et al. 2017, Zhang et al. 2020, Ray et al. 2019")
    print()
    print("Constrained RL: maximize J(pi) subject to C(pi) <= d")
    print("  J(pi) = expected return (reward)")
    print("  C(pi) = expected cost (constraint violations)")
    print("  d = cost budget (e.g., 25 hazard entries per episode)")
    print()

    # Best approximation: cost as penalty
    model = EnvironmentModel(
        name="Safety Gym (constrained, approximated)",
        max_steps=1000,
        gamma=0.99,
        n_states=100000,
        n_actions=2,
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="goal_reward",
        reward_type=RewardType.ON_EVENT,
        value=1.0,
        respawn=RespawnBehavior.ON_EPISODE,
        requires_action=True,
        discovery_probability=0.1,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="hazard_cost_approx",
        reward_type=RewardType.ON_EVENT,
        value=-0.5,  # how much penalty?
        requires_action=False,
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.01,
        num_envs=16,
        n_actors=16,
        total_steps=10_000_000,
    )
    engine.print_report(model, config)

    print()
    print("=" * 70)
    print("LIMITATION: The penalty value (-0.5) is arbitrary. In real")
    print("constrained RL, the Lagrange multiplier is LEARNED during")
    print("training to satisfy the constraint. We pick a fixed number.")
    print()
    print("What this gets wrong:")
    print("  1. Penalty strength: -0.5 might over- or under-penalize")
    print("  2. Hard vs soft: a budget of 25 violations is NOT the same")
    print("     as -0.5 per violation (agent can exceed budget if reward")
    print("     is high enough under penalty, but not under constraint)")
    print("  3. Feasible set: CPO restricts to policies that satisfy")
    print("     the constraint; penalty methods allow any policy")
    print()
    print("What goodhart would need:")
    print("  - CostSource (separate from RewardSource)")
    print("  - Constraint budget (max cost per episode)")
    print("  - Analysis of Lagrangian dual (penalty at optimality)")
    print("  - Feasibility analysis (can ANY policy satisfy both")
    print("    reward and cost objectives?)")


if __name__ == "__main__":
    run_example()
