"""Example: CartPole reward variants — default vs Sutton-Barto.

CartPole-v1 default: +1 per step for staying alive (max 500 steps).
Sutton-Barto variant: -1 on termination only.

The default is well-designed: the agent IS supposed to maximize survival.
The Sutton-Barto variant creates a reward desert (no signal until failure).

Source: Barto et al. 1983, Sutton & Barto 2018 (Ch. 3.4), Gymnasium
Tool should:
  - Default: pass clean (alive IS the goal, intentional=True)
  - Sutton-Barto: flag as reward desert (no positive signal)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # --- CartPole Default ---
    print("=" * 70)
    print("CartPole-v1 DEFAULT — +1/step alive reward")
    print("=" * 70)
    print()
    print("Source: Barto et al. 1983, Gymnasium documentation")
    print("Reward: +1 every step the pole stays upright.")
    print("The agent IS supposed to maximize survival time.")
    print()

    model_default = EnvironmentModel(
        name="CartPole-v1 (default)",
        max_steps=500,
        gamma=0.99,
        n_states=500,
        n_actions=2,
        death_probability=0.05,
    )
    model_default.add_reward_source(RewardSource(
        name="alive_reward",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        respawn=RespawnBehavior.INFINITE,
        requires_action=False,
        intentional=True,  # survival IS the goal
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.01,
        num_envs=8,
        n_actors=8,
        total_steps=100_000,
    )
    engine.print_report(model_default, config)

    # --- CartPole Sutton-Barto ---
    print()
    print("=" * 70)
    print("CartPole (Sutton-Barto) — -1 on termination only")
    print("=" * 70)
    print()
    print("Source: Sutton & Barto 2018, Chapter 3.4")
    print("Reward: -1 only when the pole falls. No per-step signal.")
    print("This is a reward desert: all non-terminal actions are equal.")
    print()

    model_sb = EnvironmentModel(
        name="CartPole (Sutton-Barto variant)",
        max_steps=500,
        gamma=1.0,
        n_states=500,
        n_actions=2,
        death_probability=0.05,
    )
    model_sb.add_reward_source(RewardSource(
        name="termination_penalty",
        reward_type=RewardType.ON_EVENT,
        value=-1.0,
        requires_action=False,
    ))

    config_sb = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.01,
        num_envs=8,
        n_actors=8,
        total_steps=100_000,
    )
    engine.print_report(model_sb, config_sb)

    print()
    print("The default CartPole has well-designed rewards: survival IS")
    print("the objective. The Sutton-Barto variant is a textbook example")
    print("of sparse negative-only reward — all non-terminal strategies")
    print("are equivalent, giving no gradient for learning.")


if __name__ == "__main__":
    run_example()
