"""Eureka Shadow Hand Door Close Outward — GPT-4 generated reward.

Approach penalties and alignment rewards create a functional reward
structure that passes clean.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "eureka_shadow_hand_door_close_outward",
    "source_paper": (
        'Ma et al. 2024, "Eureka: Human-Level Reward Design via Coding'
        ' Large Language Models," ICLR 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2310.12931",
    "source_code_url": (
        "https://eureka-research.github.io/assets/reward_functions/"
        "shadow_hand_door_close_outward.txt"
    ),
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "year": 2024,
    "domain": "manipulation",
    "brief_summary": (
        "GPT-4 generated Shadow Hand Door Close Outward reward."
        " Passes clean."
    ),
    "documented_failure": "None — reward is well-designed",
    "failure_mechanism": None,
    "discovery_stage": None,
    "fix_known": None,
    "compute_cost_class": "low",
    "is_negative_example": True,
    "encoding_rationale": {
        "left_hand_approach_reward": (
            "Active (requires_action=True). Negative exp penalty"
            " grows with distance, encouraging approach."
        ),
        "right_hand_approach_reward": (
            "Active (requires_action=True). Negative exp penalty"
            " grows with distance, encouraging approach."
        ),
        "alignment_reward_left": (
            "Intentional. Measures quaternion dot product with"
            " goal rotation for door handle alignment."
        ),
        "alignment_reward_right": (
            "Intentional. Measures quaternion dot product with"
            " goal rotation for door handle alignment."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Door Close Outward — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Shadow Hand Door Close Outward.")
    print("Inverted exp penalties and passive alignment rewards.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Door Close Outward",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="left_hand_approach_reward",
        reward_type=RewardType.PER_STEP,
        value=-0.3,  # -exp(dist/10), negative and grows with distance
        requires_action=True,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="right_hand_approach_reward",
        reward_type=RewardType.PER_STEP,
        value=-0.3,  # -exp(dist/10), negative and grows with distance
        requires_action=True,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="alignment_reward_left",
        reward_type=RewardType.PER_STEP,
        value=0.5,  # dot product * 5, passive if aligned at start
        value_range=(0.0, 1.0),
        requires_action=False,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="alignment_reward_right",
        reward_type=RewardType.PER_STEP,
        value=0.5,  # dot product * 5, passive if aligned at start
        value_range=(0.0, 1.0),
        requires_action=False,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("The hand approach rewards use -exp(dist/10), an inverted")
    print("exponential that penalizes distance but grows unbounded.")
    print("The alignment rewards measure quaternion dot product with")
    print("goal rotation, which may be high at start if the door is")
    print("already partially aligned. The approach penalties encourage")
    print("reaching the handle but the alignment rewards can be earned")
    print("passively if the door starts near the closed position.")


if __name__ == "__main__":
    run_example()
