"""Example: Atari Breakout — tunnel strategy is NOT a hack (POSITIVE).

The DQN agent discovers the tunnel strategy: dig through one side of
the wall to bounce the ball behind it, hitting high-value bricks
repeatedly. This looks like an exploit but is genuinely optimal play
that human experts also discovered.

The tool gives a clean bill, which is CORRECT. The reward function
accurately captures the game's scoring: different bricks have
different values, and maximizing score IS the intended objective.

Source: Mnih et al. 2015 (Nature), "Human-level control through
deep reinforcement learning"
Tool result: PASS (correct — creative play, not a reward hack)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Breakout Tunnel Strategy — creative play, not hacking (POSITIVE)")
    print("=" * 70)
    print()
    print("Source: Mnih et al. 2015 (Nature)")
    print("Agent discovers: tunnel through wall → bounce ball behind it")
    print("This is genuinely optimal play, not a reward exploit.")
    print()

    model = EnvironmentModel(
        name="Breakout (tunnel strategy)",
        max_steps=18000,
        gamma=0.99,
        n_states=100000,
        n_actions=4,
        death_probability=0.05,
    )
    # Three tiers of bricks with different values
    model.add_reward_source(RewardSource(
        name="brick_low_tier",
        reward_type=RewardType.ON_EVENT,
        value=1.0,
        max_occurrences=60,
        requires_action=True,
        discovery_probability=0.5,
    ))
    model.add_reward_source(RewardSource(
        name="brick_mid_tier",
        reward_type=RewardType.ON_EVENT,
        value=4.0,
        max_occurrences=30,
        requires_action=True,
        discovery_probability=0.2,
    ))
    model.add_reward_source(RewardSource(
        name="brick_high_tier",
        reward_type=RewardType.ON_EVENT,
        value=7.0,
        max_occurrences=20,
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.05,
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=2.5e-4,
        entropy_coeff=0.01,
        num_epochs=4,
        clip_epsilon=0.1,
        num_envs=8,
        n_actors=8,
        total_steps=10_000_000,
    )
    result = engine.print_report(model, config)

    print()
    print("Clean bill. This is correct.")
    print()
    print("Why the tunnel strategy is NOT a hack:")
    print("  1. Bricks are non-respawning (max_occurrences is finite)")
    print("  2. Higher bricks are worth more (correct incentive)")
    print("  3. No looping — each brick can only be hit once")
    print("  4. The agent maximizes score, which IS the objective")
    print()
    print("The tunnel strategy is the same insight human experts have:")
    print("getting the ball behind the wall is the fastest way to clear")
    print("high-value bricks. The agent is being creative, not gaming.")
    print()
    print("LESSON: not every surprising behavior is a reward hack.")
    print("A tool that flags this would be producing false positives.")
    print("goodhart's silence here is the correct answer.")


if __name__ == "__main__":
    run_example()
