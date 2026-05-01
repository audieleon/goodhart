"""FinRL transaction cost omission — excessive trading (Liu et al. 2021).

FinRL framework defaults to zero transaction costs in training.
Agent learns to trade every step (maximum turnover) since there's
no cost to entering/exiting positions. When deployed with real
transaction costs, profitability disappears.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "finrl_transaction_cost",
    "source_paper": (
        'Liu, Yang, Wang, Lu & Gao, "FinRL: Deep Reinforcement '
        'Learning Framework for Automated Stock Trading," ACM ICAIF '
        '2021. Transaction cost issue documented in: Yang et al., '
        '"Deep RL for Automated Stock Trading," ACM ICAIF 2020'
    ),
    "paper_url": "https://arxiv.org/abs/2011.09607",
    "source_code_url": "https://github.com/AI4Finance-Foundation/FinRL",
    "reward_location": (
        "Yang et al. Section 3: r_t = v(t) - v(t-1). Transaction "
        "cost c is configurable but defaults to 0. When c > 0, "
        "r_t = v(t) - v(t-1) - c * |trade_volume|. At c=0, agent "
        "learns maximum-turnover policy."
    ),
    "year": 2021,
    "domain": "finance",
    "encoding_basis": "primary_source",
    "verification_date": "2026-05-01",
    "brief_summary": (
        "Trading agent trained without transaction costs learns to "
        "trade every step. When transaction costs are added at "
        "deployment, the high-turnover policy becomes unprofitable."
    ),
    "documented_failure": (
        "With c=0, the agent has no incentive to hold positions — "
        "every small price movement triggers a trade. At realistic "
        "transaction costs (0.1-0.3%), the high-frequency churning "
        "erases all profits. This is a training/deployment mismatch."
    ),
    "failure_mechanism": "ontological_proxy",
    "detection_type": "structural",
    "discovery_stage": "post_deployment",
    "fix_known": (
        "Include realistic transaction costs in training. Use "
        "action smoothness penalty. FinRL later added configurable "
        "transaction cost parameter."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "missing_cost": (
            "Without transaction costs, every trade is free. "
            "advisory_missing_constraint should fire — the agent "
            "has continuous actions but no penalty for churning."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    model = EnvironmentModel(
        name="FinRL No Transaction Costs (Liu 2021)",
        max_steps=252,
        gamma=0.99,
        n_states=100000,
        n_actions=30,
        action_type="continuous",
        death_probability=0.0,
    )

    # P&L without transaction costs
    model.add_reward_source(RewardSource(
        name="pnl_no_costs",
        reward_type=RewardType.PER_STEP,
        value=0.04,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
