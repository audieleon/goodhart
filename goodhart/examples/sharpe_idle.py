"""Example: Financial Trading -- Sharpe ratio as reward.

Not trading gives 0/0 Sharpe (often defaulting to 0), and any trade introduces volatility.
Source: Various; Dang-Nhu 2025, "Risk-Aware RL for Financial Trading"
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "sharpe_idle",
    "source_paper": "Dang-Nhu 2025, 'Risk-Aware RL for Financial Trading'; various quant finance RL papers",
    "paper_url": None,
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2025,
    "domain": "industrial",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to trade profitably. Instead it learned to make tiny trades or not trade at all, maximizing the Sharpe ratio by minimizing volatility.",
    "documented_failure": "Sharpe ratio idle exploit: not trading gives 0/0 (often defaulted to 0 or small positive). Any trade introduces volatility in the denominator. Agent learns to never trade or make tiny trades to maximize ratio rather than return.",
    "failure_mechanism": "idle_exploit",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "medium",
    "is_negative_example": True,
    "encoding_rationale": {
        "aggregation_trap": "Exploit emerges from episode-level Sharpe ratio aggregation, not per-step signal",
        "limitation": "goodhart analyzes per-step reward structure, not episode-level aggregation functions",
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Financial Trading — Sharpe ratio idle exploit")
    print("=" * 70)
    print()
    print("Source: Dang-Nhu 2025, various quant finance RL papers")
    print("Reward: portfolio return per step (proxy for Sharpe)")
    print("Exploit: not trading = 0 volatility = infinite Sharpe")
    print()

    # Per-step reward version (what most implementations use)
    model = EnvironmentModel(
        name="Trading (Sharpe as reward)",
        max_steps=252,    # trading days in a year
        gamma=0.99,
        n_states=100000,
        n_actions=3,      # buy, hold, sell
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="portfolio_return",
        reward_type=RewardType.PER_STEP,
        value=0.001,      # ~0.1% daily return
        state_dependent=True,
        requires_action=True,
        intentional=True,
        explore_fraction=0.0,  # random trading doesn't earn returns
    ))

    config = TrainingConfig(
        lr=3e-4,
        entropy_coeff=0.01,
        num_envs=16,
        n_actors=16,
        total_steps=1_000_000,
    )
    result = engine.print_report(model, config)

    print()
    print("The tool gives a mostly-clean bill because the per-step")
    print("reward structure looks fine. But the REAL reward in")
    print("production is the Sharpe ratio = mean(return)/std(return).")
    print()
    print("The idle exploit emerges from the aggregation:")
    print("  - Trading: mean=0.001, std=0.02, Sharpe=0.05")
    print("  - Not trading: mean=0, std=0, Sharpe=undefined (→ 0)")
    print("  - Safe small trades: mean=0.0001, std=0.0001, Sharpe=1.0")
    print()
    print("The agent learns to make tiny trades that maximize the")
    print("ratio rather than the return. Or it learns to not trade")
    print("at all if the implementation handles 0/0 favorably.")
    print()
    print("LIMITATION: goodhart analyzes per-step reward structure,")
    print("not episode-level aggregation functions. The Sharpe ratio")
    print("trap exists in the AGGREGATION, not the per-step signal.")


if __name__ == "__main__":
    run_example()
