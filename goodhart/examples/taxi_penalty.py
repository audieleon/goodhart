"""Example: Taxi-v3 -- step penalty vs goal ratio (Gymnasium).

Total penalty (-200) is 10x the goal (+20), but moderate discovery probability mitigates it.
Source: Dietterich 2000 (MAXQ), Gymnasium classic control
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "taxi_penalty",
    "source_paper": "Dietterich 2000 (MAXQ hierarchical RL), Gymnasium classic control",
    "paper_url": "https://arxiv.org/abs/cs/9905014",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2000,
    "domain": "navigation",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to pick up and drop off passengers. Penalty-to-goal ratio is 10:1 (-200 vs +20), but small state space makes tabular methods succeed anyway.",
    "documented_failure": "Step penalty -1/step with max 200 steps creates -200 total penalty vs +20 goal. Agent taking 20+ steps earns net negative return. Trap matters more for function approximation than tabular methods.",
    "failure_mechanism": "idle_exploit",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "penalty_ratio": "10:1 penalty-to-goal ratio, less extreme than Mountain Car but still problematic",
        "small_state_space": "500 states means tabular Q-learning succeeds despite the ratio",
    },
}


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
    model.add_reward_source(
        RewardSource(
            name="step_penalty",
            reward_type=RewardType.PER_STEP,
            value=-1.0,
        )
    )
    model.add_reward_source(
        RewardSource(
            name="dropoff_success",
            reward_type=RewardType.TERMINAL,
            value=20.0,
            requires_action=True,
            requires_exploration=True,
            discovery_probability=0.1,
        )
    )
    model.add_reward_source(
        RewardSource(
            name="illegal_action",
            reward_type=RewardType.ON_EVENT,
            value=-10.0,
            requires_action=True,
        )
    )

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
