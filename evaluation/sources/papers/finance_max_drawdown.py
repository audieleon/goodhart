"""Maximum drawdown constraint in portfolio RL — missing from reward.

Portfolio optimization agents maximize return but drawdown constraints
are typically enforced externally (position limits, risk systems).
When drawdown is not in the reward, the agent learns to ignore it
and takes concentrated positions that maximize expected return.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "finance_max_drawdown",
    "source_paper": (
        'Yang, Liu, Wang & Gao, "Deep Reinforcement Learning for '
        'Automated Stock Trading: An Ensemble Strategy," ACM '
        'ICAIF 2020. Also: Liu, Yang, Wang, Lu & Gao, "FinRL: '
        'Deep Reinforcement Learning Framework for Automated '
        'Stock Trading," ACM ICAIF 2021'
    ),
    "paper_url": "https://arxiv.org/abs/2011.09607",
    "source_code_url": "https://github.com/AI4Finance-Foundation/FinRL",
    "reward_location": (
        "Yang et al. Section 3: r_t = v(t) - v(t-1) (portfolio "
        "value change). The ensemble strategy selects among PPO, "
        "A2C, DDPG based on Sharpe ratio on validation period. "
        "Maximum drawdown is used as an EVALUATION metric (Table 1) "
        "but NOT in the reward function."
    ),
    "year": 2020,
    "domain": "finance",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Automated stock trading with P&L reward. Maximum drawdown "
        "is measured (Table 1: 18-34% drawdown) but never penalized "
        "in the reward. The agent accumulates concentrated positions "
        "because drawdown is invisible to the optimizer."
    ),
    "documented_failure": (
        "Table 1: PPO achieves highest return but 33.6% max drawdown. "
        "The reward function (r_t = v(t)-v(t-1)) cannot distinguish "
        "between low-risk and high-risk strategies that achieve the "
        "same expected return. advisory_constrained_rl should fire — "
        "drawdown should be a constraint, not ignored."
    ),
    "failure_mechanism": "ontological_proxy",
    "detection_type": "structural",
    "discovery_stage": "post_training",
    "fix_known": (
        "Add drawdown penalty to reward, or use constrained RL "
        "(CPO/Lagrangian) with max drawdown as constraint. Yang's "
        "ensemble selection on Sharpe partially mitigates but doesn't "
        "fix the fundamental reward gap."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "advisory_missing_constraint": (
            "Drawdown is measured but not in reward. The agent "
            "has no incentive to minimize drawdown. "
            "advisory_missing_constraint should fire."
        ),
        "pnl_only": (
            "P&L change reward gives no information about risk. "
            "High-variance strategies are treated the same as "
            "low-variance strategies with same expected return."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    model = EnvironmentModel(
        name="FinRL Stock Trading — Missing Drawdown (Yang 2020)",
        max_steps=252,
        gamma=0.99,
        n_states=100000,
        n_actions=30,
        action_type="continuous",
        death_probability=0.0,
    )

    # P&L change: v(t) - v(t-1)
    model.add_reward_source(RewardSource(
        name="portfolio_pnl",
        reward_type=RewardType.PER_STEP,
        value=0.04,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
