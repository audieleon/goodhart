"""Eureka Shadow Hand Lift Underarm — GPT-4 generated reward.

Binary grasping reward and sharp goal temperature create a functional
reward structure that passes clean.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "eureka_shadow_hand_lift_underarm",
    "source_paper": (
        'Ma et al. 2024, "Eureka: Human-Level Reward Design via Coding'
        ' Large Language Models," ICLR 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2310.12931",
    "source_code_url": (
        "https://eureka-research.github.io/assets/reward_functions/"
        "shadow_hand_lift_underarm.txt"
    ),
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "year": 2024,
    "domain": "manipulation",
    "brief_summary": (
        "GPT-4 generated Shadow Hand Lift Underarm reward."
        " Passes clean."
    ),
    "documented_failure": "None — reward is well-designed",
    "failure_mechanism": None,
    "discovery_stage": None,
    "fix_known": None,
    "compute_cost_class": "low",
    "is_negative_example": True,
    "encoding_rationale": {
        "object_goal_reward": (
            "Active (requires_action=True) and intentional."
            " exp(-15*dist) with sharp temperature."
        ),
        "grasping_reward": (
            "Binary threshold (0.1) for hand proximity to"
            " handles. Functional gating signal."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Lift Underarm — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Shadow Hand Lift Underarm.")
    print("Passive grasping reward and extreme goal temperature.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Lift Underarm",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="object_goal_reward",
        reward_type=RewardType.PER_STEP,
        value=0.5,  # exp(-15*dist), very sharp
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="grasping_reward",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # binary: both hands within 0.1 of handles
        value_range=(0.0, 0.3),  # idle: hands start near handles
        requires_action=False,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("The grasping_reward is binary (threshold 0.1) and passive:")
    print("hands start near pot handles, so both left_hand_grasping and")
    print("right_hand_grasping trigger at init. The object_goal_reward")
    print("uses temperature 15.0, making it extremely sensitive — reward")
    print("drops to near-zero very quickly with distance. The passive")
    print("grasping reward (0.3/step) dominates early training.")


if __name__ == "__main__":
    run_example()
