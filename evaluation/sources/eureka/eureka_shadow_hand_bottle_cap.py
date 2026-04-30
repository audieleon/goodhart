"""Eureka Shadow Hand Bottle Cap — GPT-4 generated reward.

Passive hand-distance reward creates idle floor; raw negative cap-goal
distance is unscaled and hard to optimize.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "eureka_shadow_hand_bottle_cap",
    "source_paper": (
        'Ma et al. 2024, "Eureka: Human-Level Reward Design via Coding'
        ' Large Language Models," ICLR 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2310.12931",
    "source_code_url": (
        "https://eureka-research.github.io/assets/reward_functions/"
        "shadow_hand_bottle_cap.txt"
    ),
    "reward_location": "GPT-4 generated reward code at https://eureka-research.github.io/assets/reward_functions/shadow_hand_bottle_cap.txt",
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "year": 2024,
    "domain": "manipulation",
    "brief_summary": (
        "GPT-4 generated Shadow Hand Bottle Cap reward."
        " Passive hand-distance component saturates at idle;"
        " raw negative cap-goal distance lacks exp scaling."
    ),
    "documented_failure": (
        "hand_distance_transformed_reward is passive: hands start"
        " near the bottle/cap, so exp(-dist/50) is ~1.0 at idle."
        " cap_goal_distance_reward is raw negative distance with"
        " no exponential transform, making it hard to balance."
        " Idle strategy dominates."
    ),
    "failure_mechanism": "idle_exploit",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Make hand_distance_transformed_reward active by measuring"
        " grasp progress, and apply exp transform to cap-goal"
        " distance for balanced gradient."
    ),
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "hand_distance_transformed_reward": (
            "Passive (requires_action=False). Hands start near"
            " bottle/cap, saturating exp(-dist/50) at rest."
        ),
        "cap_goal_distance_reward": (
            "Active (requires_action=True) and intentional."
            " Raw negative distance without exp scaling makes"
            " gradient hard to follow."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Bottle Cap — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Shadow Hand Bottle Cap.")
    print("Passive hand distance reward creates idle exploit.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Bottle Cap",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="hand_distance_transformed_reward",
        reward_type=RewardType.PER_STEP,
        value=0.5,  # active: hands move, exp decays
        value_range=(0.0, 1.0),  # idle: hands start near objects, ~1.0
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="cap_goal_distance_reward",
        reward_type=RewardType.PER_STEP,
        value=-0.5,  # negative raw distance, no exp transform
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("The hand_distance_transformed_reward is passive: hands start")
    print("near the bottle/cap, so exp(-dist/50) is near 1.0 at idle.")
    print("The cap_goal_distance_reward is raw negative distance with no")
    print("exponential scaling, making it hard to balance against the")
    print("passive component. Idle strategy dominates.")


if __name__ == "__main__":
    run_example()
