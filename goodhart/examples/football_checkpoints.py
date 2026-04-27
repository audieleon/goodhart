"""Example: Google Research Football — checkpoint shaping.

GRF uses checkpoint rewards (+0.1 for crossing field positions) to
shape learning toward the sparse +1 goal. This creates respawning
shaping that could incentivize cycling if checkpoints reset. In GRF
they don't (one-time per episode), so can_loop=False.

Source: Kurach et al. 2020 (ICML), "Google Research Football"
Tool should: mostly pass clean (checkpoints are one-time, not loopable)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Google Research Football — checkpoint shaping")
    print("=" * 70)
    print()
    print("Source: Kurach et al. 2020 (ICML)")
    print("Sparse +1 for goal, +0.1 for each checkpoint (10 zones).")
    print("Checkpoints are one-time per episode, so no looping.")
    print()

    model = EnvironmentModel(
        name="Google Research Football (academy)",
        max_steps=3000,
        gamma=0.997,
        n_states=100000,
        n_actions=19,
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="goal_scored",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.1,
    ))
    # 10 checkpoint zones, each gives +0.1 once per episode
    model.add_reward_source(RewardSource(
        name="checkpoint_shaping",
        reward_type=RewardType.ON_EVENT,
        value=0.1,
        max_occurrences=10,
        respawn=RespawnBehavior.NONE,
        requires_action=True,
        can_loop=False,
    ))
    model.add_reward_source(RewardSource(
        name="goal_conceded",
        reward_type=RewardType.ON_EVENT,
        value=-1.0,
        requires_action=False,
    ))

    config = TrainingConfig(
        algorithm="APPO",
        lr=2.5e-4,
        entropy_coeff=0.003,
        num_epochs=2,
        clip_epsilon=0.115,
        num_envs=16,
        num_workers=8,
        n_actors=128,
        total_steps=500_000_000,
        rollout_length=512,
        minibatch_size=4096,
    )
    result = engine.print_report(model, config)

    print()
    print("Checkpoint shaping is well-designed here: one-time per episode,")
    print("total shaping value (10 * 0.1 = 1.0) equals the goal reward,")
    print("and they're consumed, not respawned. The main risk would be if")
    print("checkpoints were respawnable — then cycling would dominate.")


if __name__ == "__main__":
    run_example()
