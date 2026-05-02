"""Example: Long horizon with sparse goal — discount makes goal invisible.

At gamma=0.99 over 10,000 steps, a sparse goal at step 5000 is
discounted to ~5e-22 -- the agent literally cannot see the reward.

Source: Arjona-Medina et al. 2019, "RUDDER: Return Decomposition
  for Delayed Rewards" (NeurIPS, arXiv:1806.07857)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "long_horizon_sparse",
    "source_paper": "Arjona-Medina et al. 2019, RUDDER: Return Decomposition for Delayed Rewards (NeurIPS)",
    "paper_url": "https://arxiv.org/abs/1806.07857",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2019,
    "domain": "control",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to reach a goal at step 5000. Instead gamma=0.99 discounts the reward to ~5e-22, making it invisible.",
    "documented_failure": "gamma=0.99 with sparse terminal goal at ~step 5000 in a 10000-step episode; discounted value at step 0 is 0.99^5000 * 10.0 ~ 5e-22, exponentially below learning threshold",
    "failure_mechanism": "discount_horizon_mismatch",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "RUDDER return decomposition, or reward shaping to redistribute temporal credit",
    "compute_cost_class": "medium",
    "is_negative_example": True,
    "encoding_rationale": {
        "discount_mismatch": "Effective horizon (100 steps at gamma=0.99) is 100x shorter than episode length",
        "rudder_motivation": "This exact failure mode motivated the RUDDER algorithm",
    },
}


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
    model.add_reward_source(
        RewardSource(
            name="sparse_goal",
            reward_type=RewardType.TERMINAL,
            value=10.0,
            requires_action=True,
            intentional=True,
            discovery_probability=0.001,
        )
    )

    engine.print_report(model)

    print()
    print("At gamma=0.99, the effective horizon is 100 steps.")
    print("The episode is 10000 steps (100x the horizon).")
    print("A reward at step 5000 is discounted to 0.99^5000 ≈ 5e-22.")
    print("The agent cannot learn from a reward it cannot see.")


if __name__ == "__main__":
    run_example()
