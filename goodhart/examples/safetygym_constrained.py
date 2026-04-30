"""Example: Safety Gym with explicit constraint budget (advisory demo).

Advisory fires when negative event rewards approximate safety constraints.
Source: Ray et al. 2019 (Safety Gym), Achiam et al. 2017 (CPO)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "safetygym_constrained",
    "source_paper": "Ray et al. 2019 (Safety Gym), Achiam et al. 2017 (CPO)",
    "paper_url": "https://arxiv.org/abs/1910.01708",
    "source_code_url": "https://github.com/openai/safety-gym",
    "reward_location": "Reward structure from paper description",
    "year": 2019,
    "domain": "safety",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to reach goal with bounded hazard entries. Soft penalty -0.1 allows 100 hazard entries for only -10.0 cost, making constraint violations always worthwhile.",
    "documented_failure": "With soft penalty -0.1, agent can enter 100 hazards for -10.0 total. With hard constraint (budget=25), the 26th entry is forbidden regardless of reward. Different optimization landscape entirely.",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Use constrained optimization (CPO/FOCOPS) with hard cost budget instead of soft penalty",
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "advisory_demo": "Shows how advisory_constrained_rl rule fires on negative event rewards",
        "soft_vs_hard": "Demonstrates the gap between penalty approximation and true constraint enforcement",
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Safety Gym PointGoal2 — constraint advisory demo")
    print("=" * 70)
    print()
    print("Source: Ray et al. 2019, Achiam et al. 2017")
    print("Real config: cost budget = 25 hazard entries per episode.")
    print("Our model: hazard penalty = -0.1 (soft approximation).")
    print()

    model = EnvironmentModel(
        name="Safety Gym PointGoal2",
        max_steps=1000,
        gamma=0.99,
        n_states=100000,
        n_actions=2,
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="goal_reward",
        reward_type=RewardType.ON_EVENT,
        value=1.0,
        respawn=RespawnBehavior.ON_EPISODE,
        requires_action=True,
        discovery_probability=0.1,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="hazard_entry",
        reward_type=RewardType.ON_EVENT,
        value=-0.1,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="vase_contact",
        reward_type=RewardType.ON_EVENT,
        value=-0.1,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="distance_shaping",
        reward_type=RewardType.SHAPING,
        value=0.01,
        state_dependent=True,
        scales_with="distance",
        requires_action=True,
        can_loop=False,
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.01,
        num_envs=16,
        n_actors=16,
        total_steps=10_000_000,
    )
    result = engine.print_report(model, config)

    print()
    print("The advisory fires on hazard_entry and vase_contact.")
    print("Key insight: with soft penalty -0.1, the agent can enter")
    print("100 hazards per episode for only -10.0 total cost. If the")
    print("goal reward is +1.0, that's always worth it.")
    print()
    print("With a HARD constraint (budget=25), the 26th entry is")
    print("forbidden regardless of reward. Different optimization")
    print("landscape entirely.")


if __name__ == "__main__":
    run_example()
