"""Sortino ratio as RL reward — asymmetric aggregation trap.

The Sortino ratio (return/downside_deviation) penalizes only
negative volatility, unlike Sharpe. As a ratio, it has the same
aggregation trap as Sharpe: not trading (zero downside deviation)
makes the ratio undefined or infinite.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "finance_sortino_trap",
    "source_paper": (
        'Srivastava, Aryan & Singh, "Risk-Aware Reinforcement '
        'Learning for Algorithmic Trading," 2025. Sortino ratio '
        'trap also discussed in Moody & Saffell 1998 (NeurIPS) '
        'and Rodinos et al. 2023 (AIAI)'
    ),
    "paper_url": "https://arxiv.org/abs/2307.00557",
    "source_code_url": None,
    "reward_location": (
        "Sortino = E[R] / sqrt(E[min(R-target,0)^2]). Like Sharpe, "
        "this is a ratio where the denominator (downside deviation) "
        "approaches zero for idle agents, creating division-by-zero "
        "or infinite reward. Same aggregation trap as Sharpe but "
        "only penalizing negative returns."
    ),
    "year": 2025,
    "domain": "finance",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Sortino ratio reward has the same aggregation trap as "
        "Sharpe: not trading minimizes downside deviation, making "
        "the ratio infinite. The per-step encoding uses small "
        "positive values, triggering advisory_aggregation_trap."
    ),
    "documented_failure": (
        "The Sortino ratio = mean_return / downside_deviation. An "
        "idle agent has zero downside deviation (never loses money), "
        "making the ratio undefined (division by zero) or artificially "
        "high. This is the same idle exploit as Sharpe."
    ),
    "failure_mechanism": "aggregation_trap",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Use differential Sortino ratio (analogous to Moody's "
        "differential Sharpe). Or use per-step P&L with explicit "
        "drawdown penalty."
    ),
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "aggregation_trap": (
            "Sortino = ratio of mean to downside std. Not a sum. "
            "Per-step encoding uses the differential approximation "
            "which is small and positive, like differential Sharpe."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    model = EnvironmentModel(
        name="Sortino Ratio Trading — Aggregation Trap",
        max_steps=252,
        gamma=0.99,
        n_states=100000,
        n_actions=3,
        action_type="discrete",
        death_probability=0.0,
    )

    # Differential Sortino approximation
    # Small per-step, positive when profitable, like diff Sharpe
    model.add_reward_source(RewardSource(
        name="differential_sortino",
        reward_type=RewardType.PER_STEP,
        value=0.01,
        state_dependent=True,
        requires_action=True,
        intentional=True,
        explore_fraction=0.0,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
