"""Example: FrozenLake-8x8 — simplest exploration threshold example.

FrozenLake has +1 for reaching the goal, 0 for everything else
(including falling in holes). With stochastic transitions and an
8x8 grid, random walk discovery probability is ~1%.

This is the simplest possible environment where goodhart adds
value. Good for tutorials and onboarding.

Source: Gymnasium documentation, standard RL textbook environment
Tool result: WARNING — budget_sufficiency (correctly flags sparsity)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


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
