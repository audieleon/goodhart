"""RELAVER latency-aware market making (Jiang et al. 2025).

R = PnL - inventory_penalty + exchange_compensation - execution_risk.
Documents failure of latency-unaware MM methods. 6D action space.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "latency_market_making",
    "source_paper": (
        'Jiang, Yang, Wang, Li, Huang & Li, "Resolving Latency and '
        'Inventory Risk in Market Making with Reinforcement Learning," '
        'arXiv:2505.12465, 2025'
    ),
    "paper_url": "https://arxiv.org/abs/2505.12465",
    "source_code_url": None,
    "reward_location": (
        "Section 2.3, page 3: R = PnL - IP + C - ER. "
        "PnL = spread earnings + mark-to-market. "
        "IP = quadratic inventory penalty. "
        "C = exchange fee rebates. "
        "ER = Σ(σ·V·(1+t/t_c)) stacking execution risk. "
        "6D action (Eq. 1). 500ms steps, 4h episodes. PPO."
    ),
    "year": 2025,
    "domain": "finance",
    "encoding_basis": "primary_source",
    "verification_date": "2026-05-01",
    "brief_summary": (
        "RL market maker with latency-aware reward. Exchange latency "
        "(30-80ms) causes order cancellations and unintended inventory "
        "accumulation. Standard RL MM ignores this. RELAVER adds "
        "stacking execution risk term to penalize orders vulnerable "
        "to latency-induced failure."
    ),
    "documented_failure": (
        "Existing RL MM methods overlook exchange latency, leading to "
        "unintended order cancellations and inventory accumulation. "
        "Without ER term, agent accumulates dangerous inventory "
        "positions during volatile markets. RELAVER's ER term reduces "
        "MAP (mean absolute position) by 67% vs DRL_os baseline."
    ),
    "failure_mechanism": "missing_constraint",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Add stacking execution risk ER = Σ(σ·V·(1+t/t_c)) to reward. "
        "Penalizes orders that accumulate in queue during volatile "
        "periods, accounting for latency-induced cancellation risk."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "four_component": (
            "PnL (positive, from spreads), IP (negative, quadratic), "
            "C (positive, exchange rebates), ER (negative, latency risk). "
            "Well-balanced — the documented failure is the ABSENCE of ER "
            "in prior work, not a flaw in this reward."
        ),
        "latency_novel": (
            "The ER term is the novel contribution. Without it, "
            "the agent doesn't account for the 30-80ms delay between "
            "decision and execution, leading to stale orders."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # R = PnL - IP + C - ER
    # 6D action, 500ms steps, 4h episodes, Chinese futures

    model = EnvironmentModel(
        name="RELAVER Market Making (Jiang et al. 2025)",
        max_steps=28800,  # 4 hours at 500ms
        gamma=0.99,
        n_actions=6,  # bid/ask deltas, volumes, holding times
        action_type="continuous",
        death_probability=0.0,
    )

    # PnL: spread earnings + mark-to-market
    model.add_reward_source(RewardSource(
        name="profit_and_loss",
        reward_type=RewardType.PER_STEP,
        value=0.5,  # average spread capture per step
        requires_action=True,
        intentional=True,
        state_dependent=True,
    ))

    # Inventory Penalty: -IP (quadratic)
    model.add_reward_source(RewardSource(
        name="inventory_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.2,  # quadratic cost of position deviation
        requires_action=True,
        state_dependent=True,
    ))

    # Exchange Compensation: +C (rebates)
    model.add_reward_source(RewardSource(
        name="exchange_rebate",
        reward_type=RewardType.PER_STEP,
        value=0.1,  # fee rebates for liquidity provision
        requires_action=True,
        state_dependent=True,
    ))

    # Stacking Execution Risk: -ER (latency)
    # ER = Σ(σ·V·(1+t/t_c)) — penalizes stale orders
    model.add_reward_source(RewardSource(
        name="execution_risk",
        reward_type=RewardType.PER_STEP,
        value=-0.1,  # latency-aware risk penalty
        requires_action=True,
        state_dependent=True,
        value_type="proportional",
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
