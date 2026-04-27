"""Example: Taxi-v3 — step penalty vs goal ratio (Gymnasium).

Taxi has -1/step penalty and +20 for successful dropoff, with max 200
steps. Total penalty (-200) is 10x the goal (+20). However, unlike
Mountain Car, the discovery probability is moderate (~10% random) and
the environment has well-defined intermediate structure.

Source: Dietterich 2000 (MAXQ), Gymnasium classic control
Tool should catch: penalty_dominates_goal (ratio 200:20 = 10:1)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Taxi-v3 — step penalty vs goal reward ratio")
    print("=" * 70)
    print()
    print("Source: Dietterich 2000 (MAXQ hierarchical RL)")
    print("Reward: -1/step, +20 dropoff, -10 illegal pickup/dropoff")
    print("Max steps: 200, so total penalty = -200 vs goal = +20")
    print()

    model = EnvironmentModel(
        name="Taxi-v3",
        max_steps=200,
        gamma=0.99,
        n_states=500,
        n_actions=6,
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="step_penalty",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
    ))
    model.add_reward_source(RewardSource(
        name="dropoff_success",
        reward_type=RewardType.TERMINAL,
        value=20.0,
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.1,
    ))
    model.add_reward_source(RewardSource(
        name="illegal_action",
        reward_type=RewardType.ON_EVENT,
        value=-10.0,
        requires_action=True,
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.01,
        num_envs=16,
        n_actors=16,
        total_steps=500_000,
    )
    engine.print_report(model, config)

    print()
    print("The penalty-to-goal ratio is 10:1. While this is less extreme")
    print("than Mountain Car (200:1), it still means an agent that takes")
    print("20+ steps to reach the goal earns net negative return.")
    print("In practice, tabular Q-learning solves Taxi easily because")
    print("the state space is small (500). The trap matters more for")
    print("function approximation methods.")


if __name__ == "__main__":
    run_example()
