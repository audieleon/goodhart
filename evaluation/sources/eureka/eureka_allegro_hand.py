"""Eureka Allegro Hand — GPT-4 generated dexterous manipulation.

Clean reward: orientation and fingertip bonus both require action.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "eureka_allegro_hand",
    "source_paper": (
        'Ma et al. 2024, "Eureka: Human-Level Reward Design via Coding'
        ' Large Language Models," ICLR 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2310.12931",
    "source_code_url": (
        "https://eureka-research.github.io/assets/reward_functions/"
        "allegro_hand.txt"
    ),
    "reward_location": "GPT-4 generated reward code at https://eureka-research.github.io/assets/reward_functions/allegro_hand.txt",
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "year": 2024,
    "domain": "manipulation",
    "brief_summary": (
        "GPT-4 generated Allegro Hand manipulation reward."
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
        "orientation_reward": (
            "Requires action (must rotate object) and intentional"
            " (directly measures task objective). No passive component."
        ),
        "fingertip_bonus": (
            "Requires action (must maintain contact) and intentional"
            " (directly measures manipulation quality)."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Allegro Hand — GPT-4 generated manipulation reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Allegro Hand task.")
    print("Both components require action and are intentional.")
    print()

    model = EnvironmentModel(
        name="Eureka Allegro Hand",
        max_steps=1000,
        gamma=0.99,
        n_actions=16,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="orientation_reward",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="fingertip_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.3,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("Another clean Eureka output. Both reward components directly")
    print("measure the manipulation objective: orienting the object and")
    print("maintaining fingertip contact. No passive terms to exploit.")


if __name__ == "__main__":
    run_example()
