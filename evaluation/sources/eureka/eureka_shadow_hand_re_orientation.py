"""Eureka Shadow Hand Re-Orientation — GPT-4 generated reward.

Clean rotation-based reward with success bonuses; all components
are active and intentional.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "eureka_shadow_hand_re_orientation",
    "source_paper": (
        'Ma et al. 2024, "Eureka: Human-Level Reward Design via Coding'
        ' Large Language Models," ICLR 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2310.12931",
    "source_code_url": (
        "https://eureka-research.github.io/assets/reward_functions/"
        "shadow_hand_re_orientation.txt"
    ),
    "reward_location": "GPT-4 generated reward code at https://eureka-research.github.io/assets/reward_functions/shadow_hand_re_orientation.txt",
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "year": 2024,
    "domain": "manipulation",
    "brief_summary": (
        "GPT-4 generated Shadow Hand Re-Orientation reward."
        " Passes clean."
    ),
    "documented_failure": "None — reward is well-designed",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": None,
    "fix_known": None,
    "compute_cost_class": "low",
    "is_negative_example": True,
    "encoding_rationale": {
        "object_rotation_reward": (
            "Active (requires_action=True) and intentional."
            " exp(-8*rot_dist) directly measures the task."
        ),
        "object_another_rotation_reward": (
            "Active (requires_action=True) and intentional."
            " exp(-5*rot_dist) for second object."
        ),
        "object_orientation_success_bonus": (
            "Active (requires_action=True) and intentional."
            " 5x scale when rot_dist < 0.02."
        ),
        "object_another_orientation_success_bonus": (
            "Active (requires_action=True) and intentional."
            " 5x scale when rot_dist < 0.02."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Re-Orientation — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Shadow Hand Re-Orientation.")
    print("Clean rotation-based reward with success bonuses.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Re-Orientation",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="object_rotation_reward",
        reward_type=RewardType.PER_STEP,
        value=0.5,  # exp(-8*rot_distance)
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="object_another_rotation_reward",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # exp(-5*rot_distance)
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="object_orientation_success_bonus",
        reward_type=RewardType.PER_STEP,
        value=1.0,  # 5*(1-rot_dist) when rot_dist < 0.02
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="object_another_orientation_success_bonus",
        reward_type=RewardType.PER_STEP,
        value=1.0,  # 5*(1-rot_dist) when rot_dist < 0.02
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("Well-structured reward: all components directly measure the")
    print("task objective (object orientation matching goal). The exp")
    print("transforms with temperatures 8.0 and 5.0 create appropriate")
    print("gradients. Success bonuses (5x scale) fire when rotation")
    print("distance < 0.02, providing strong near-goal incentive. No")
    print("passive components — rotation always requires active control.")


if __name__ == "__main__":
    run_example()
