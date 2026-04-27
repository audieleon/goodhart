"""Example: Financial Trading — Sharpe ratio as reward.

When Sharpe ratio (return/volatility) is the reward, doing nothing
gives 0/0 — which many implementations default to 0 or a small
positive number. Any trade introduces volatility in the denominator,
making the ratio worse. The optimal policy is to never trade.

This is an idle exploit, but a subtle one: the reward function
itself (portfolio return per step) doesn't show it. It emerges
from the Sharpe ratio AGGREGATION over the episode.

Source: Various; Dang-Nhu 2025, "Risk-Aware RL for Financial Trading"
Tool result: PASS (misses the Sharpe aggregation trap — honest limitation)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


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
