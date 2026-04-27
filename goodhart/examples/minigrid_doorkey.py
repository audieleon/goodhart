"""Example: MiniGrid DoorKey — staged exploration with time pressure.

MiniGrid DoorKey requires the agent to find a key, open a door, then
reach the goal. Reward = 1 - 0.9*(step_count/max_steps), so early
completion is heavily rewarded. This creates interesting dynamics:
the exploration needed to find the key conflicts with the time pressure.

Source: Chevalier-Boisvert et al. 2023 (Minigrid & Miniworld)
Tool should catch: penalty_dominates_goal risk at high step counts,
  sparse exploration threshold
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("MiniGrid DoorKey — staged exploration + time pressure")
    print("=" * 70)
    print()
    print("Source: Chevalier-Boisvert et al. 2023")
    print("R = 1 - 0.9*(steps/max_steps). Must find key, open door,")
    print("reach goal. 3-stage exploration problem.")
    print()

    model = EnvironmentModel(
        name="MiniGrid DoorKey-8x8",
        max_steps=300,
        gamma=0.99,
        n_states=5000,
        n_actions=7,
        death_probability=0.0,
        wall_probability=0.3,
    )
    model.add_reward_source(RewardSource(
        name="goal_reached",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.02,
    ))
    # Time pressure modeled as step penalty
    model.add_reward_source(RewardSource(
        name="time_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.003,  # 0.9/300 = -0.003/step
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=7e-4,
        entropy_coeff=0.01,
        num_epochs=4,
        num_envs=16,
        n_actors=16,
        total_steps=5_000_000,
        rollout_length=128,
    )
    engine.print_report(model, config)

    print()
    print("With 2% discovery probability and 300 steps, the agent")
    print("needs ~50 episodes to see a single success. The time")
    print("penalty (-0.9 total) reduces the goal from 1.0 to 0.1")
    print("at the last step, but doesn't make failure attractive.")


if __name__ == "__main__":
    run_example()
