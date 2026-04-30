"""Example: Self-play reward non-stationarity (LIMITATION).

Opponent changes during training, making reward distribution non-stationary.
Source: Silver et al. 2017 (AlphaGo Zero), Bansal et al. 2018 (ICLR)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "self_play_nonstationarity",
    "source_paper": "Silver et al. 2017 (AlphaGo Zero), Bansal et al. 2018 ('Emergent Complexity via Multi-Agent Competition', ICLR)",
    "paper_url": "https://arxiv.org/abs/1710.03748",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2018,
    "domain": "multi_agent",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to learn competitive play. Reward structure is clean (+1/-1 win/lose) but non-stationarity causes forgetting cycles, strategy collapse, and non-transitivity.",
    "documented_failure": "goodhart assumes stationary reward dynamics. Cannot capture forgetting cycles, strategy collapse, or non-transitivity in self-play environments where reward distribution shifts as opponent improves.",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Population-based training, league training, diversity metrics",
    "compute_cost_class": "high",
    "is_negative_example": True,
    "encoding_rationale": {
        "limitation_example": "Reward structure is correct; the problem is non-stationarity from changing opponents",
        "game_theory": "Self-play is about game theory, not reward structure",
    },
}


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
