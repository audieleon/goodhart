"""Eureka Shadow Hand Door Open Outward — GPT-4 generated reward.

Door position and orientation rewards measure distance from closed,
rewarding the initial state and opposing the task objective.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "eureka_shadow_hand_door_open_outward",
    "source_paper": (
        'Ma et al. 2024, "Eureka: Human-Level Reward Design via Coding'
        ' Large Language Models," ICLR 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2310.12931",
    "source_code_url": (
        "https://eureka-research.github.io/assets/reward_functions/"
        "shadow_hand_door_open_outward.txt"
    ),
    "reward_location": "GPT-4 generated reward code at https://eureka-research.github.io/assets/reward_functions/shadow_hand_door_open_outward.txt",
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "year": 2024,
    "domain": "manipulation",
    "brief_summary": (
        "GPT-4 generated Shadow Hand Door Open Outward reward."
        " Door rewards measure distance from closed position,"
        " rewarding the initial state and opposing the task."
    ),
    "documented_failure": (
        "handle_reward is passive: hand starts near handle."
        " door_position_reward and door_orientation_reward both"
        " measure distance from closed position using quaternion"
        " dot product and angle diff. Since the task is to OPEN"
        " the door, these reward keeping it closed (the initial"
        " state), directly opposing the objective."
    ),
    "failure_mechanism": "reward_inversion",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Invert door rewards to measure distance from open"
        " position instead of closed position."
    ),
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "handle_reward": (
            "Passive (requires_action=False). Hand starts near"
            " handle, saturating exp(-0.5*dist) at rest."
        ),
        "door_position_reward": (
            "Passive (requires_action=False). Measures distance"
            " from closed position; door starts closed = max"
            " reward. Inverted relative to task goal."
        ),
        "door_orientation_reward": (
            "Passive (requires_action=False). Measures angle"
            " diff from closed; door starts closed = max"
            " reward. Redundant with door_position_reward."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Door Open Outward — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Shadow Hand Door Open Outward.")
    print("Passive handle reward and redundant door rotation metrics.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Door Open Outward",
        max_steps=1000,
        gamma=0.99,
        n_actions=20,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="handle_reward",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # active: hand moves away
        value_range=(0.0, 1.0),  # idle: hand starts near handle
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="door_position_reward",
        reward_type=RewardType.PER_STEP,
        value=0.5,  # exp(-5*(1 - quat_dot)), high when door closed
        value_range=(0.0, 1.0),  # idle: door starts closed = max reward
        requires_action=False,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="door_orientation_reward",
        reward_type=RewardType.PER_STEP,
        value=0.5,  # exp(-0.5*angle_diff), high when door closed
        value_range=(0.0, 1.0),  # idle: door starts closed = max reward
        requires_action=False,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("The handle_reward is passive: exp(-0.5*dist) is high when the")
    print("hand starts near the handle. Critically, both door_position_")
    print("reward and door_orientation_reward measure distance from the")
    print("closed position using quaternion dot product and angle diff.")
    print("Since the task is to OPEN the door, these rewards are inverted:")
    print("they reward the door being CLOSED (the initial state). The")
    print("agent is rewarded for keeping the door closed, which directly")
    print("opposes the task objective.")


if __name__ == "__main__":
    run_example()
