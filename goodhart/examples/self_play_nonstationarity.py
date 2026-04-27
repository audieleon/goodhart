"""Example: Self-play reward non-stationarity (LIMITATION).

In self-play (AlphaGo, OpenAI Five, hide-and-seek), the opponent
changes during training, making the reward distribution non-stationary.
An agent that beats last week's opponent may lose to this week's.

LIMITATION: goodhart assumes stationary reward dynamics. We can model
the reward components of a self-play environment, but we can't capture
the fact that the effective reward distribution shifts as the opponent
improves. This means we can't detect:
  - Forgetting cycles (agent forgets how to beat old strategies)
  - Strategy collapse (both players converge to a dominated equilibrium)
  - Non-transitivity (A beats B, B beats C, C beats A)

Source: Silver et al. 2017 (AlphaGo Zero), Bansal et al. 2018
("Emergent Complexity via Multi-Agent Competition", ICLR)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Self-Play Non-Stationarity (TOOL LIMITATION)")
    print("=" * 70)
    print()
    print("Source: Silver et al. 2017, Bansal et al. 2018 (ICLR)")
    print()
    print("In competitive self-play, the opponent IS the environment.")
    print("As the opponent improves, the reward distribution shifts.")
    print()
    print("What goodhart says about the reward structure:")
    print()

    # Model competitive self-play (e.g., Sumo wrestling from Bansal 2018)
    model = EnvironmentModel(
        name="Competitive Self-Play (Sumo)",
        max_steps=500,
        gamma=0.99,
        n_states=100000,
        n_actions=8,
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="win",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        requires_action=True,
        discovery_probability=0.5,  # ~50% win rate at equilibrium
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="lose",
        reward_type=RewardType.TERMINAL,
        value=-1.0,
        requires_action=False,
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=1e-4,
        entropy_coeff=0.01,
        num_epochs=4,
        num_envs=32,
        n_actors=32,
        total_steps=500_000_000,
    )
    result = engine.print_report(model, config)

    print()
    print("=" * 70)
    print("LIMITATION: goodhart says this is clean — and the REWARD")
    print("structure IS clean (+1 win, -1 lose). But the real issues")
    print("are all about non-stationarity:")
    print()
    print("  1. Forgetting: agent stops beating strategy X after")
    print("     training against strategy Y for too long")
    print("  2. Collapse: both agents converge to a poor equilibrium")
    print("     (e.g., both stand still in Sumo)")
    print("  3. Non-transitivity: rock-paper-scissors dynamics mean")
    print("     'better' isn't a total order")
    print()
    print("What goodhart would need:")
    print("  - Opponent model (opponent as part of environment)")
    print("  - Non-stationarity analysis (how fast does reward shift?)")
    print("  - Equilibrium analysis (Nash, correlated equilibrium)")
    print("  - Population diversity metrics")
    print()
    print("Self-play is fundamentally about game THEORY, not reward")
    print("STRUCTURE. Different tool entirely.")


if __name__ == "__main__":
    run_example()
