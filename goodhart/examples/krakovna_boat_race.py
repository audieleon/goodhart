"""Example: Krakovna's Boat Race — checkpoint cycling.

From the Krakovna et al. 2020 specification gaming examples list.
A boat race agent discovers it can go in circles hitting checkpoints
instead of completing the race. Unlike CoastRunners (turbo powerups),
the exploit here is in the checkpoint system itself.

Source: Krakovna et al. 2020 ("Specification Gaming: The Flip Side
of AI Ingenuity"), DeepMind Safety Research blog
Tool should catch: respawning_exploit (checkpoints reset on loop)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna Boat Race — checkpoint cycling exploit")
    print("=" * 70)
    print()
    print("Source: Krakovna et al. 2020 (Specification Gaming)")
    print("Agent goes in circles hitting 3 checkpoints repeatedly")
    print("instead of completing the 12-checkpoint course.")
    print()

    model = EnvironmentModel(
        name="Boat Race (checkpoint cycling)",
        max_steps=500,
        gamma=0.99,
        n_states=10000,
        n_actions=5,
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="checkpoint",
        reward_type=RewardType.ON_EVENT,
        value=1.0,
        respawn=RespawnBehavior.TIMED,
        respawn_time=10,
        can_loop=True,
        loop_period=30,  # 3 checkpoints * ~10 steps each
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="finish_race",
        reward_type=RewardType.TERMINAL,
        value=10.0,
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.05,
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.01,
        num_envs=16,
        n_actors=16,
        total_steps=1_000_000,
    )
    engine.print_report(model, config)

    print()
    print("The tool catches the respawning checkpoint exploit:")
    print("  Loop reward: 500/30 * 1.0 = ~16.7 per episode")
    print("  Race reward: 10.0 * 0.05 = 0.5 expected")
    print("Cycling dominates racing by 33x in expected value.")


if __name__ == "__main__":
    run_example()
