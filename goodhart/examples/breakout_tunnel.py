"""Example: Atari Breakout — tunnel strategy is NOT a hack (POSITIVE).

DQN discovers the tunnel strategy to reach high-value bricks. This is
genuinely optimal play, not a reward exploit, and the tool correctly passes.

Source: Mnih et al. 2015 (Nature)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "breakout_tunnel",
    "source_paper": "Mnih et al. 2015, 'Human-level control through deep reinforcement learning' (Nature)",
    "paper_url": "https://doi.org/10.1038/nature14236",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2015,
    "domain": "game_ai",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent discovers the tunnel strategy to clear high-value bricks. This is creative optimal play, not a reward hack.",
    "documented_failure": "N/A — the tunnel strategy is genuinely optimal play that human experts also discovered. The tool correctly gives a clean bill.",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "medium",
    "is_negative_example": True,
    "encoding_rationale": {
        "finite_bricks": "Each brick can only be hit once (max_occurrences is finite)",
        "no_looping": "Non-respawning rewards prevent cycling exploits",
    },
}


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
    engine.print_report(model, config)

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
