"""Eureka Franka Cabinet — GPT-4 generated cabinet opening.

Clean reward: opening and gripper bonus are active; velocity penalty is small.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "eureka_franka_cabinet",
    "source_paper": (
        'Ma et al. 2024, "Eureka: Human-Level Reward Design via Coding'
        ' Large Language Models," ICLR 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2310.12931",
    "source_code_url": (
        "https://eureka-research.github.io/assets/reward_functions/"
        "franka_cabinet.txt"
    ),
    "reward_location": "GPT-4 generated reward code at https://eureka-research.github.io/assets/reward_functions/franka_cabinet.txt",
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "year": 2024,
    "domain": "manipulation",
    "brief_summary": (
        "GPT-4 generated Franka Cabinet opening reward."
        " Passes clean."
    ),
    "documented_failure": "None — reward is well-designed",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "post_training",
    "fix_known": None,
    "compute_cost_class": "low",
    "is_negative_example": True,
    "encoding_rationale": {
        "cabinet_open": (
            "Active (requires_action=True) and intentional."
            " Directly measures task objective. Random movement"
            " partially opens cabinet (explore_fraction=0.2)."
        ),
        "gripper_bonus": (
            "Active (requires_action=True) and intentional."
            " Measures handle gripping quality."
        ),
        "velocity_penalty": (
            "Active (requires_action=True) and small (-0.05 vs"
            " 1.3 positive). Regularizes without idle exploit."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Franka Cabinet — GPT-4 generated cabinet opening")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Franka Cabinet task.")
    print("Velocity penalty correctly structured with requires_action.")
    print()

    model = EnvironmentModel(
        name="Eureka Franka Cabinet",
        max_steps=500,
        gamma=0.99,
        n_actions=9,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="cabinet_open",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=True,
        intentional=True,
        explore_fraction=0.2,  # random arm movement partially opens cabinet
    ))
    model.add_reward_source(RewardSource(
        name="gripper_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.3,
        requires_action=True,
        intentional=True,
        explore_fraction=0.2,  # random gripper actuation earns partial credit
    ))
    model.add_reward_source(RewardSource(
        name="velocity_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.05,
        requires_action=True,
    ))

    engine.print_report(model)

    print()
    print("Well-designed reward: both positive components directly measure")
    print("the task objective (opening the cabinet, gripping the handle).")
    print("The velocity penalty (-0.05) is small relative to the task")
    print("rewards (1.0 + 0.3) and requires action, so it cannot create")
    print("an idle exploit. This is one of Eureka's better outputs.")


if __name__ == "__main__":
    run_example()
