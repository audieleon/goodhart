"""Eureka Shadow Hand Door Close Inward — GPT-4 generated reward.

Passive hand-distance rewards create idle floor; no terminal goal
anchors the shaping signals.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "eureka_shadow_hand_door_close_inward",
    "source_paper": (
        'Ma et al. 2024, "Eureka: Human-Level Reward Design via Coding'
        ' Large Language Models," ICLR 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2310.12931",
    "source_code_url": (
        "https://eureka-research.github.io/assets/reward_functions/"
        "shadow_hand_door_close_inward.txt"
    ),
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "year": 2024,
    "domain": "manipulation",
    "brief_summary": (
        "GPT-4 generated Shadow Hand Door Close Inward reward."
        " Passive hand-to-handle distance rewards create idle"
        " floor with no terminal goal to anchor shaping."
    ),
    "documented_failure": (
        "Hand distance rewards are passive: exp(-dist) is high"
        " when hands start near handles. Two passive components"
        " each yield ~1.0/step at idle, creating an idle floor"
        " (2.0/step) that competes with goal handle rewards."
        " No terminal goal anchors the shaping signals."
    ),
    "failure_mechanism": "idle_exploit",
    "discovery_stage": "during_training",
    "fix_known": (
        "Gate hand-distance rewards on active grasping."
        " Add terminal success reward for door closure."
    ),
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "left_hand_distance_reward": (
            "Passive (requires_action=False). Hand starts near"
            " handle, saturating exp(-dist) at rest."
        ),
        "right_hand_distance_reward": (
            "Passive (requires_action=False). Hand starts near"
            " handle, saturating exp(-dist) at rest."
        ),
        "goal_left_handle_distance_reward": (
            "Active (requires_action=True) and intentional."
            " Measures handle-to-goal distance."
        ),
        "goal_right_handle_distance_reward": (
            "Active (requires_action=True) and intentional."
            " Measures handle-to-goal distance."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Door Close Inward — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Shadow Hand Door Close Inward.")
    print("Passive hand-to-handle distance rewards create idle exploit.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Door Close Inward",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="left_hand_distance_reward",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # active: hand moves
        value_range=(0.0, 1.0),  # idle: hand starts near handle
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="right_hand_distance_reward",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # active: hand moves
        value_range=(0.0, 1.0),  # idle: hand starts near handle
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="goal_left_handle_distance_reward",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="goal_right_handle_distance_reward",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("The hand distance rewards are passive: exp(-dist) is high when")
    print("hands start near handles. With two passive components each")
    print("yielding ~1.0/step at idle, the idle floor (2.0/step) competes")
    print("with the goal handle rewards. No terminal goal anchors the")
    print("shaping signals.")


if __name__ == "__main__":
    run_example()
