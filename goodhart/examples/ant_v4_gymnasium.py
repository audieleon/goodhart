"""Example: Gymnasium Ant-v4 — control cost trap.

Ant-v4 has forward velocity reward, alive bonus, and control cost.
With the default ctrl_cost_weight=0.5, the control penalty is high
enough that agents learn to move with minimal joint actuation,
resulting in sliding/twitching gaits instead of natural locomotion.

Source: Todorov et al. 2012 (MuJoCo), Gymnasium documentation
  (ctrl_cost_weight reduced from 0.5 to 0.05 in some setups)
Tool should catch: control penalty interaction with velocity reward
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Gymnasium Ant-v4 — control cost weight analysis")
    print("=" * 70)
    print()
    print("Source: Todorov et al. 2012, Gymnasium v0.26+")
    print("Default: ctrl_cost_weight=0.5, healthy_reward=1.0")
    print("Known issue: high control cost produces twitching gaits")
    print()

    # Default (problematic) config
    model = EnvironmentModel(
        name="Ant-v4 (default ctrl_cost=0.5)",
        max_steps=1000,
        gamma=0.99,
        n_states=100000,
        n_actions=8,
        death_probability=0.02,
    )
    model.add_reward_source(RewardSource(
        name="forward_velocity",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        value_range=(-1.0, 6.0),
        state_dependent=True,
        scales_with="velocity",
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="healthy_reward",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        respawn=RespawnBehavior.INFINITE,
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="ctrl_cost",
        reward_type=RewardType.PER_STEP,
        value=-0.5,
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="contact_cost",
        reward_type=RewardType.PER_STEP,
        value=-0.0005,
        requires_action=False,
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.0,
        num_epochs=10,
        num_envs=1,
        n_actors=1,
        total_steps=1_000_000,
        rollout_length=2048,
        minibatch_size=64,
    )
    engine.print_report(model, config)

    print()
    print("With ctrl_cost_weight=0.5, the agent pays 0.5 * ||action||^2")
    print("per step. A strong action (||a||=2) costs 2.0 — more than the")
    print("velocity reward. The agent learns to minimize actuation.")
    print()
    print("Fix: reduce ctrl_cost_weight to 0.05 (as recommended in")
    print("several Gymnasium discussions) or use use_contact_force=False.")


if __name__ == "__main__":
    run_example()
