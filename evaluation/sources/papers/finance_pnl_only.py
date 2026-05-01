"""P&L-only trading reward — ignores risk (common failure pattern).

Most DRL trading systems use raw P&L (profit and loss) as reward.
This ignores risk entirely: an agent that makes $100 with 50%
drawdown risk is treated the same as one making $100 with 1%
drawdown. The agent learns to take maximum leverage.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "finance_pnl_only",
    "source_paper": (
        'Rodinos, Nousi, Passalis & Tefas, "A Sharpe Ratio Based '
        'Reward Scheme in Deep RL for Financial Trading," AIAI 2023. '
        'Also documented in Yang, Liu, Wang & Gao, "Deep '
        'Reinforcement Learning for Automated Stock Trading," '
        'ACM ICAIF 2020'
    ),
    "paper_url": "https://dl.acm.org/doi/10.1145/3383455.3422540",
    "source_code_url": None,
    "reward_location": (
        "Common baseline in financial RL: r_t = portfolio_value(t) "
        "- portfolio_value(t-1). Yang et al. Section 3: r_t = "
        "v(t) - v(t-1) (change in portfolio value). Rodinos Slide 5: "
        "P&L reward 'doesn\\'t take into account the risk associated "
        "with the returns.' No risk penalty, no drawdown constraint."
    ),
    "year": 2023,
    "domain": "finance",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Trading agent rewarded by raw P&L (change in portfolio "
        "value). Ignores risk, drawdown, and volatility. Agent "
        "learns to take maximum leverage since upside is rewarded "
        "but downside risk has no penalty."
    ),
    "documented_failure": (
        "Rodinos et al. (AIAI 2023) identified P&L-only as the "
        "standard baseline and showed it produces policies with "
        "high volatility and large drawdowns compared to "
        "Sharpe-ratio-based alternatives. The agent maximizes "
        "expected profit regardless of risk."
    ),
    "failure_mechanism": "ontological_proxy",
    "detection_type": "structural",
    "discovery_stage": "post_training",
    "fix_known": (
        "Use risk-adjusted reward: Sharpe ratio (Moody 1998), "
        "differential Sharpe (Moody 2001), Sortino ratio, or "
        "add explicit drawdown penalty."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "missing_risk_constraint": (
            "P&L reward has no risk penalty. The agent can achieve "
            "high expected P&L by taking extreme positions. "
            "advisory_missing_constraint should fire."
        ),
        "positive_only": (
            "P&L change can be positive (profit) or negative (loss) "
            "but is symmetric. There's no asymmetric downside penalty "
            "to discourage catastrophic losses."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # P&L-only trading
    # Daily trading, ~252 trading days/year, gamma=0.99
    model = EnvironmentModel(
        name="P&L-Only Trading — No Risk Penalty",
        max_steps=252,
        gamma=0.99,
        n_states=100000,
        n_actions=3,
        action_type="discrete",
        death_probability=0.0,
    )

    # P&L: portfolio_value(t) - portfolio_value(t-1)
    # Average daily return ~0.04% = $40 on $100k
    model.add_reward_source(RewardSource(
        name="pnl_reward",
        reward_type=RewardType.PER_STEP,
        value=0.04,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
