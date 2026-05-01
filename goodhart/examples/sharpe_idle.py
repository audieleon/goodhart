"""Sharpe ratio as RL reward — aggregation trap (Moody & Saffell 1998).

Differential Sharpe ratio D_t is small per-step, positive when
profitable. Minimizing volatility (not trading) maximizes the ratio.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "sharpe_idle",
    "source_paper": (
        'Moody & Saffell, "Reinforcement Learning for Trading," '
        'NeurIPS 1998, pp. 917-923'
    ),
    "paper_url": "http://papers.neurips.cc/paper/1551-reinforcement-learning-for-trading.pdf",
    "source_code_url": None,
    "reward_location": (
        "Section 2.2, Eq. 3: S_T = Average(R_t)/StdDev(R_t) "
        "(episode-level Sharpe ratio). Eq. 4: D_t = differential "
        "Sharpe ratio (per-step approximation). Eq. 1: P_T = additive "
        "profit. F_t in {-1,0,1} (short/neutral/long). delta = "
        "transaction cost. 84 inputs, monthly S&P 500 1970-1994."
    ),
    "year": 1998,
    "domain": "finance",
    "encoding_basis": "primary_source",
    "verification_date": "2026-05-01",
    "brief_summary": (
        "RL trading agent optimizes Sharpe ratio via differential "
        "Sharpe ratio (Eq. 4). The per-step reward is small and "
        "positive when profitable. The aggregation trap: Sharpe = "
        "mean/std, so minimizing volatility (not trading) maximizes "
        "the ratio. advisory_aggregation_trap should fire."
    ),
    "documented_failure": (
        "The paper introduced the differential Sharpe ratio (Eq. 4) "
        "specifically because the batch Sharpe ratio (Eq. 3) is an "
        "episode-level aggregation incompatible with online RL. The "
        "idle exploit is implicit: StdDev(R_t) in the denominator "
        "means not trading (zero variance) dominates any trading "
        "strategy with nonzero variance, regardless of returns."
    ),
    "failure_mechanism": "aggregation_trap",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Differential Sharpe ratio (Eq. 4): D_t approximates the "
        "marginal contribution of R_t to S_T, enabling online "
        "learning without the aggregation trap."
    ),
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "aggregation_trap": (
            "The Sharpe ratio is a RATIO (mean/std), not a sum. "
            "goodhart analyzes per-step sums. The per-step encoding "
            "uses the differential Sharpe (Eq. 4) which is small "
            "and positive, triggering advisory_aggregation_trap."
        ),
        "per_step_encoding": (
            "D_t (Eq. 4) is the per-step reward. Typical values "
            "are ~0.001-0.01 per monthly step. Small, positive, "
            "no terminal goal — the aggregation trap pattern."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Moody & Saffell 1998: S&P 500 trading
    # F_t in {-1, 0, 1}, 84 inputs, monthly, 25 years
    # Per-step reward: differential Sharpe ratio D_t (Eq. 4)

    model = EnvironmentModel(
        name="Trading via Differential Sharpe (Moody & Saffell 1998)",
        max_steps=300,    # 25 years monthly
        gamma=0.99,
        n_states=100000,
        n_actions=3,      # short, neutral, long
        action_type="discrete",
        death_probability=0.0,
    )

    # Differential Sharpe ratio D_t (Eq. 4)
    # Small per-step value, positive when profitable
    model.add_reward_source(RewardSource(
        name="differential_sharpe",
        reward_type=RewardType.PER_STEP,
        value=0.01,       # typical D_t magnitude
        state_dependent=True,
        requires_action=True,
        intentional=True,
        explore_fraction=0.0,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
