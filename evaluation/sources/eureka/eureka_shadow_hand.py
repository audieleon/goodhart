"""Eureka Shadow Hand — GPT-4 generated dexterous manipulation.

Passive distance reward creates idle floor (EV 87) exceeding active EV (43).
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "eureka_shadow_hand",
    "source_paper": (
        'Ma et al. 2024, "Eureka: Human-Level Reward Design via Coding'
        ' Large Language Models," ICLR 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2310.12931",
    "source_code_url": (
        "https://eureka-research.github.io/assets/reward_functions/"
        "shadow_hand.txt"
    ),
    "reward_location": "GPT-4 generated reward code at https://eureka-research.github.io/assets/reward_functions/shadow_hand.txt",
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "year": 2024,
    "domain": "manipulation",
    "brief_summary": (
        "GPT-4 generated Shadow Hand manipulation reward."
        " Passive distance reward creates idle floor that"
        " exceeds active expected value."
    ),
    "documented_failure": (
        "distance_reward is passive: fingertips start near the"
        " object, so idle policy earns 1.0/step (EV ~100)."
        " Active manipulation moves fingertips away, reducing"
        " distance_reward to ~0.2/step. Idle EV (87) exceeds"
        " active EV (43), making inaction dominant."
    ),
    "failure_mechanism": "idle_exploit",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Make distance_reward active (requires_action=True) by"
        " measuring progress toward a manipulation target rather"
        " than proximity to the object's initial position."
    ),
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "distance_reward": (
            "Passive (requires_action=False). Fingertips start"
            " near the object, saturating this term at rest."
            " Any movement reduces it."
        ),
        "rotation_reward": (
            "Active (requires_action=True) and intentional."
            " Measures actual manipulation but cannot overcome"
            " the idle floor from distance_reward."
        ),
        "fingertip_bonus": (
            "Active (requires_action=True) but unintentional."
            " Proxy for contact quality."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand — GPT-4 generated manipulation reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Shadow Hand task.")
    print("Passive distance reward creates idle exploit.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand",
        max_steps=1000,
        gamma=0.99,
        n_actions=20,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="distance_reward",
        reward_type=RewardType.PER_STEP,
        value=0.2,  # active: fingertips move away from object
        value_range=(0.0, 1.0),  # idle: fingertips stay near object (1.0)
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="rotation_reward",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="fingertip_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.2,
        requires_action=True,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("The distance_reward is passive (requires_action=False) because")
    print("fingertips start near the object. Idle: value_range max (1.0/step)")
    print("gives EV ~100. Active manipulation moves fingertips away, reducing")
    print("distance_reward to ~0.2/step while gaining 0.5/step rotation at")
    print("explore_fraction=0.0. Explore EV ~40. The idle floor exceeds the")
    print("active ceiling, making inaction the dominant strategy.")


if __name__ == "__main__":
    run_example()
