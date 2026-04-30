"""Crypto market making via DRL (Sadighian 2019, EPITA).

Two reward functions compared: positional PnL (Eq. 13)
and trade completion (Eq. 14). 17 discrete actions for
bid/ask LOB placement. BTC/ETH/LTC pairs.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "crypto_market_making",
    "source_paper": (
        'Sadighian, "Deep Reinforcement Learning in Cryptocurrency '
        'Market Making," arXiv:1911.08647, 2019'
    ),
    "paper_url": "https://arxiv.org/abs/1911.08647",
    "source_code_url": None,
    "reward_location": (
        "Section 4.1.4, Eq. 13: Positional PnL r = Δm*ic + PNL_realized. "
        "Eq. 14: Trade Completion r = clip(PNL_realized, ε*ϖ, -ϖ). "
        "ε=2, ϖ=0.2% (Fig. 2). 17 actions (Table 2). "
        "143 features × 100 lookback. A2C and PPO."
    ),
    "year": 2019,
    "domain": "finance",
    "encoding_basis": "primary_source",
    "verification_date": "2026-05-01",
    "brief_summary": (
        "RL agent acts as cryptocurrency market maker, placing bid/ask "
        "orders on the LOB. Two reward functions compared: positional "
        "PnL (continuous, inventory-exposed) and trade completion "
        "(clipped, encourages round-trip trades). Tested on BTC, ETH, LTC."
    ),
    "documented_failure": (
        "None — framework paper comparing two reward designs. Trade "
        "completion reward (Eq. 14) outperforms positional PnL on "
        "daily returns. Both are well-designed for their purpose."
    ),
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "medium",
    "is_negative_example": True,
    "encoding_rationale": {
        "two_rewards_compared": (
            "Paper compares two reward designs head-to-head. "
            "Encoding the trade completion reward (Eq. 14) as the "
            "primary entry — it's the more interesting design with "
            "clipped goal-based structure."
        ),
        "inventory_risk_implicit": (
            "Positional PnL exposes agent to inventory risk (Δm*ic). "
            "Trade completion avoids this by rewarding round-trips. "
            "The comparison is relevant for reward design guidance."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Encoding the Trade Completion reward (Eq. 14)
    # r = clip(PNL_realized): +1 if profit ≥ threshold,
    # -1 if loss ≥ threshold, else proportional
    # 17 actions, A2C/PPO

    model = EnvironmentModel(
        name="Crypto Market Making (Sadighian 2019)",
        max_steps=1440,  # 24 hours at 1-min steps
        gamma=0.99,
        n_actions=17,  # bid/ask level combos + no-op + market
        action_type="discrete",
        death_probability=0.0,
    )

    # Trade completion reward (Eq. 14)
    # Fires only when a position is closed (realized PnL)
    # +1 if PNL >= ε*ϖ (profitable trade), -1 if PNL <= -ϖ (loss)
    # Otherwise PNL_realized (proportional to trade profit)
    model.add_reward_source(RewardSource(
        name="trade_completion",
        reward_type=RewardType.ON_EVENT,
        value=0.5,  # average realized PnL per trade
        requires_action=True,
        intentional=True,
        state_dependent=True,
        value_range=(-1.0, 1.0),  # clipped [-1, +1]
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
