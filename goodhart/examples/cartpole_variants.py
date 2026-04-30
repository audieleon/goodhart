"""Example: CartPole reward variants — default vs Sutton-Barto.

Compares well-designed +1/step alive reward (passes clean) against
the -1-on-termination variant that creates a reward desert.

Source: Barto et al. 1983, Sutton & Barto 2018 (Ch. 3.4), Gymnasium
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "cartpole_variants",
    "source_paper": "Barto et al. 1983, Sutton & Barto 2018 (Ch. 3.4), Gymnasium",
    "paper_url": "http://www.cs.ualberta.ca/~sutton/papers/barto-sutton-anderson-83.pdf",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 1983,
    "domain": "control",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Default CartPole (+1/step) is well-designed. Sutton-Barto variant (-1 on termination only) creates a reward desert with no gradient signal.",
    "documented_failure": "Sutton-Barto variant gives -1 only on termination with no per-step signal, making all non-terminal strategies equivalent and providing no gradient for learning.",
    "failure_mechanism": "death_beats_survival",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Use +1 per step alive reward (Gymnasium default)",
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "reward_desert": "Sutton-Barto variant has no positive signal, all non-terminal actions equal",
        "well_designed_default": "Default variant correctly makes survival the intentional goal",
    },
}


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
