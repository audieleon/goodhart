"""Example: Discount myopia — gamma too low for the task.

gamma=0.95 gives an effective horizon of 20 steps in a 500-step
episode, making rewards beyond step 20 invisible to the agent.

Source: Hu et al. 2022, "On the Role of Discount Factor in Offline RL" (ICML)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "discount_myopia",
    "source_paper": "Hu et al. 2022, 'On the Role of Discount Factor in Offline Reinforcement Learning' (ICML)",
    "paper_url": "https://arxiv.org/abs/2206.01626",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2022,
    "domain": "control",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to reach a distant goal. Instead it ignores rewards beyond 20 steps due to gamma=0.95 myopia.",
    "documented_failure": "gamma=0.95 with a 500-step episode gives an effective horizon of 20 steps. The terminal goal at step ~250 is discounted to 0.95^250 ~ 0.00003, making it effectively invisible.",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Increase gamma to 0.999 (horizon = 1000 steps)",
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "gamma_horizon_mismatch": "Effective horizon 1/(1-gamma) = 20 steps vs 500-step episode",
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Discount Myopia — gamma=0.95, 500-step episode")
    print("=" * 70)
    print()
    print("Source: Hu et al. 2022 (ICML)")
    print("Effective horizon = 20 steps. Episode = 500 steps.")
    print()

    model = EnvironmentModel(
        name="Discount Myopia",
        max_steps=500,
        gamma=0.95,
        n_actions=4,
        action_type="discrete",
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="task_goal",
        reward_type=RewardType.TERMINAL,
        value=10.0,
        requires_action=True,
        intentional=True,
        discovery_probability=0.02,
    ))

    engine.print_report(model)

    print()
    print("The agent can only plan 20 steps ahead. The goal at")
    print("step ~250 (median) is discounted to 0.95^250 ≈ 0.00003.")
    print("Fix: increase gamma to 0.999 (horizon = 1000 steps).")


if __name__ == "__main__":
    run_example()
