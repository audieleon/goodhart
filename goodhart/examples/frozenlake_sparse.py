"""Example: FrozenLake-8x8 — simplest exploration threshold example.

Sparse +1 goal reward with ~1% random discovery rate on 8x8 grid
demonstrates budget sufficiency warnings for the simplest RL setup.

Source: Gymnasium documentation, standard RL textbook environment
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "frozenlake_sparse",
    "source_paper": "Gymnasium documentation, standard RL textbook environment",
    "paper_url": "https://gymnasium.farama.org/environments/toy_text/frozen_lake/",
    "source_code_url": None,
    "reward_location": "Reward structure from Gymnasium documentation",
    "year": 2013,
    "domain": "navigation",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to reach the goal. Sparse reward with 1% discovery rate makes learning slow without shaping.",
    "documented_failure": "FrozenLake-8x8 has +1 for goal and 0 for everything else (no hole penalty). With 1% random discovery and 2500 episodes, the agent expects only ~25 goal discoveries. A reward desert where all non-goal states are equal.",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Add hole penalty (-1) to provide gradient signal, though this changes the optimal policy",
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "sparse_reward": "Only +1 at goal with 0 everywhere else, creating a reward desert",
        "low_discovery": "1% random discovery rate on stochastic 8x8 grid",
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("FrozenLake-8x8 — simplest exploration threshold example")
    print("=" * 70)
    print()
    print("Source: Gymnasium, standard RL textbook")
    print("Reward: +1 for goal, 0 for holes, 0 per step")
    print("Challenge: 1% random discovery rate on 8x8 grid")
    print()

    model = EnvironmentModel(
        name="FrozenLake-8x8",
        max_steps=200,
        gamma=0.99,
        n_states=64,
        n_actions=4,
        death_probability=0.0,
        wall_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="goal",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.01,
    ))
    # Note: no penalty for holes! This is actually a design issue —
    # without negative signal for holes, the agent has no feedback
    # on BAD actions, only on the rare GOOD outcome.

    config = TrainingConfig(
        lr=3e-4,
        entropy_coeff=0.01,
        num_envs=16,
        n_actors=16,
        total_steps=500_000,
    )
    result = engine.print_report(model, config)

    print()
    print("The tool flags budget sufficiency — with 1% discovery rate")
    print("and 500K steps / 200 steps per episode = 2500 episodes,")
    print("the agent expects ~25 goal discoveries. Learnable, but slow.")
    print()
    print("The deeper insight: adding a hole penalty (-1) would help")
    print("learning but changes the optimal policy. Without penalty,")
    print("all non-goal states are equal — a reward desert. With penalty,")
    print("the agent learns to avoid holes, which provides gradient signal.")
    print()
    print("This is the simplest environment where the tool demonstrates")
    print("value: a beginner creates FrozenLake, wonders why learning")
    print("is slow, runs goodhart, and sees 'budget_sufficiency' warning.")
    print("That's the intended workflow.")


if __name__ == "__main__":
    run_example()
