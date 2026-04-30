"""Eureka Shadow Hand Door Open Inward — GPT-4 generated reward.

All four components are passive or partially passive; no terminal goal
anchors the reward structure.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "eureka_shadow_hand_door_open_inward",
    "source_paper": (
        'Ma et al. 2024, "Eureka: Human-Level Reward Design via Coding'
        ' Large Language Models," ICLR 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2310.12931",
    "source_code_url": (
        "https://eureka-research.github.io/assets/reward_functions/"
        "shadow_hand_door_open_inward.txt"
    ),
    "reward_location": "GPT-4 generated reward code at https://eureka-research.github.io/assets/reward_functions/shadow_hand_door_open_inward.txt",
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "year": 2024,
    "domain": "manipulation",
    "brief_summary": (
        "GPT-4 generated Shadow Hand Door Open Inward reward."
        " All four components are passive or partially passive;"
        " no terminal goal anchors the reward."
    ),
    "documented_failure": (
        "Hand-handle distance rewards are passive: small initial"
        " distances yield high exp reward at idle. Orientation"
        " alignment rewards may also start high. Agent earns"
        " substantial reward by doing nothing."
    ),
    "failure_mechanism": "idle_exploit",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Gate distance rewards on active contact. Add terminal"
        " success reward for door opening."
    ),
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "reward_hand_handle_dist_left": (
            "Passive (requires_action=False). Hand starts near"
            " handle, saturating exp reward at rest."
        ),
        "reward_hand_handle_dist_right": (
            "Passive (requires_action=False). Hand starts near"
            " handle, saturating exp reward at rest."
        ),
        "reward_door_handle_orientation_left": (
            "Passive (requires_action=False). Orientation may"
            " start aligned, yielding idle reward."
        ),
        "reward_door_handle_orientation_right": (
            "Passive (requires_action=False). Orientation may"
            " start aligned, yielding idle reward."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Door Open Inward — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Shadow Hand Door Open Inward.")
    print("Passive hand-handle distance and orientation rewards.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Door Open Inward",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="reward_hand_handle_dist_left",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # active: hand moves away
        value_range=(0.0, 0.9),  # idle: hand starts near handle
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="reward_hand_handle_dist_right",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # active: hand moves away
        value_range=(0.0, 0.9),  # idle: hand starts near handle
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="reward_door_handle_orientation_left",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        value_range=(0.0, 1.0),  # idle: may start aligned
        requires_action=False,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="reward_door_handle_orientation_right",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        value_range=(0.0, 1.0),  # idle: may start aligned
        requires_action=False,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("All four components are passive or partially passive: hand-")
    print("handle distances start small (high exp reward) and orientation")
    print("alignment may be high at initialization. The agent can earn")
    print("substantial reward by doing nothing. No terminal goal anchors")
    print("the overall reward structure.")


if __name__ == "__main__":
    run_example()
