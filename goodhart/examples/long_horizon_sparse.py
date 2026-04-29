"""Example: Long horizon with sparse goal — discount makes goal invisible.

gamma=0.99 with a sparse terminal goal at ~step 5000 in a 10000-step
episode. The discounted value of the goal at step 0 is
0.99^5000 × 10.0 ≈ 0.0000005. The agent literally cannot see it.

This is the exact failure mode RUDDER was designed to address:
TD and MC methods are exponentially slowed by reward delay.

Source: Arjona-Medina et al. 2019, "RUDDER: Return Decomposition
  for Delayed Rewards" (NeurIPS, arXiv:1806.07857)
Tool should catch: discount_horizon_mismatch (CRITICAL),
  reward_delay_horizon (CRITICAL)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Long Horizon Sparse Goal — discount hides the reward")
    print("=" * 70)
    print()
    print("Source: Arjona-Medina et al. 2019 (NeurIPS, RUDDER)")
    print("gamma=0.99, 10000 steps, sparse goal at ~step 5000.")
    print()

    model = EnvironmentModel(
        name="Long Horizon Sparse Goal",
        max_steps=10000,
        gamma=0.99,
        n_actions=4,
        action_type="discrete",
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="sparse_goal",
        reward_type=RewardType.TERMINAL,
        value=10.0,
        requires_action=True,
        intentional=True,
        discovery_probability=0.001,
    ))

    engine.print_report(model)

    print()
    print("At gamma=0.99, the effective horizon is 100 steps.")
    print("The episode is 10000 steps (100x the horizon).")
    print("A reward at step 5000 is discounted to 0.99^5000 ≈ 5e-22.")
    print("The agent cannot learn from a reward it cannot see.")


if __name__ == "__main__":
    run_example()
