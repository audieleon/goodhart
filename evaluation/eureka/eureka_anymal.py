"""Eureka ANYmal — GPT-4 generated quadruped locomotion.

Clean reward: velocity tracking with small energy penalty, both active.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "eureka_anymal",
    "source_paper": (
        'Ma et al. 2024, "Eureka: Human-Level Reward Design via Coding'
        ' Large Language Models," ICLR 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2310.12931",
    "source_code_url": (
        "https://eureka-research.github.io/assets/reward_functions/"
        "anymal.txt"
    ),
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "domain": "locomotion",
    "brief_summary": (
        "GPT-4 generated ANYmal quadruped locomotion reward."
        " Passes clean."
    ),
    "documented_failure": "None — reward is well-designed",
    "failure_mechanism": None,
    "discovery_stage": "post_training",
    "fix_known": None,
    "compute_cost_class": "low",
    "is_negative_example": True,
    "encoding_rationale": {
        "velocity_tracking": (
            "Active (requires_action=True) and intentional."
            " Directly measures locomotion objective. Random"
            " movement produces some velocity (explore_fraction=0.3)"
            " but not enough to dominate."
        ),
        "energy_penalty": (
            "Active (requires_action=True) and tiny (-0.01 vs 1.0)."
            " Regularizes without creating idle exploit."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka ANYmal — GPT-4 generated quadruped locomotion")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym ANYmal task.")
    print("Velocity tracking with small energy penalty.")
    print()

    model = EnvironmentModel(
        name="Eureka ANYmal",
        max_steps=1000,
        gamma=0.99,
        n_actions=12,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="velocity_tracking",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=True,
        intentional=True,
        explore_fraction=0.3,  # random movement produces some velocity
    ))
    model.add_reward_source(RewardSource(
        name="energy_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.01,
        requires_action=True,
    ))

    engine.print_report(model)

    print()
    print("Well-structured reward: velocity tracking directly measures")
    print("the locomotion objective and the energy penalty (-0.01) is")
    print("tiny relative to the tracking reward (1.0), so it regularizes")
    print("without creating an idle exploit. Both terms require action.")


if __name__ == "__main__":
    run_example()
