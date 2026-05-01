"""Power grid RL — voltage violations from cost-only reward.

Grid operators use RL to minimize dispatch cost. But cost-only
reward ignores voltage and frequency constraints. Agent learns
to dispatch cheapest generators regardless of grid stability,
causing voltage violations at the distribution level.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "grid_voltage_violation",
    "source_paper": (
        'Marchesini, Maguire, Mukherjee, Kandasamy & Hines, '
        '"RL2Grid: Benchmarking RL in Power Grid Operations," '
        'arXiv 2025 (U. Vermont)'
    ),
    "paper_url": "https://arxiv.org/abs/2503.23101",
    "source_code_url": "https://github.com/DEUCE1957/rl2grid",
    "reward_location": (
        "Section 3: CMDP formulation with cost = dispatch_cost and "
        "constraints = voltage limits + thermal limits + load "
        "shedding + islanding. Base reward: r = -dispatch_cost. "
        "Without constraints: agent minimizes cost, violating "
        "voltage bounds. With CMDP: constraint violations as costs "
        "with limit c."
    ),
    "year": 2025,
    "domain": "energy",
    "encoding_basis": "primary_source",
    "verification_date": "2026-05-01",
    "brief_summary": (
        "Power grid dispatch with cost-only reward causes voltage "
        "violations. The agent dispatches cheapest generators "
        "without regard to grid stability. RL2Grid benchmark "
        "explicitly adds CMDP constraints to prevent this."
    ),
    "documented_failure": (
        "RL2Grid Section 1: 'conventional RL relies on trial-and-error "
        "which often leads to exploring unsafe operating regions.' "
        "Section 3: voltage, thermal, load shedding, and islanding "
        "constraints are needed because cost-only reward ignores "
        "physical grid limits."
    ),
    "failure_mechanism": "ontological_proxy",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "CMDP formulation: add voltage/thermal/load constraints "
        "with Lagrange multiplier enforcement. RL2Grid provides "
        "this as a benchmark."
    ),
    "compute_cost_class": "high",
    "is_negative_example": False,
    "encoding_rationale": {
        "advisory_constrained_rl": (
            "Voltage and thermal limits are physical constraints "
            "that cannot be softened. They should be hard constraints "
            "in a CMDP, not reward penalties."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    model = EnvironmentModel(
        name="Power Grid Dispatch — Cost Only (RL2Grid 2025)",
        max_steps=288,
        gamma=0.99,
        n_states=100000,
        n_actions=10,
        action_type="continuous",
        death_probability=0.0,
    )

    # Dispatch cost (negative, minimize)
    model.add_reward_source(RewardSource(
        name="dispatch_cost",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
