"""Eureka Shadow Hand Swing Cup — GPT-4 generated reward.

Passive grasp reward, always-negative distance reward, and velocity
penalty actively discourage task completion.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "eureka_shadow_hand_swing_cup",
    "source_paper": (
        'Ma et al. 2024, "Eureka: Human-Level Reward Design via Coding'
        ' Large Language Models," ICLR 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2310.12931",
    "source_code_url": (
        "https://eureka-research.github.io/assets/reward_functions/"
        "shadow_hand_swing_cup.txt"
    ),
    "reward_location": "GPT-4 generated reward code at https://eureka-research.github.io/assets/reward_functions/shadow_hand_swing_cup.txt",
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "year": 2024,
    "domain": "manipulation",
    "brief_summary": (
        "GPT-4 generated Shadow Hand Swing Cup reward."
        " Passive grasp, always-negative distance, and velocity"
        " penalty actively discourage task completion."
    ),
    "documented_failure": (
        "grasp_reward is passive: hands start near cup handles."
        " object_goal_distance_reward is always negative"
        " (-exp(-0.1*dist)*2), so approaching the goal reduces"
        " total reward. cup_linvel_penalty is zero at rest,"
        " rewarding stillness. touch_reward duplicates grasp."
        " The reward structure discourages task completion."
    ),
    "failure_mechanism": "idle_exploit",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Invert distance reward sign. Remove velocity penalty"
        " or gate it on successful grasp."
    ),
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "grasp_reward": (
            "Passive (requires_action=False). Hands start near"
            " cup handles, saturating exp at rest."
        ),
        "object_goal_distance_reward": (
            "Active (requires_action=True) and intentional but"
            " always negative: moving toward goal reduces"
            " total reward (inverted signal)."
        ),
        "cup_orientation_reward": (
            "Active (requires_action=True) and intentional."
            " Measures rotation alignment with goal."
        ),
        "cup_linvel_penalty": (
            "Passive (requires_action=False). Zero at rest,"
            " penalizes motion, rewarding stillness."
        ),
        "touch_reward": (
            "Passive (requires_action=False). Redundant with"
            " grasp_reward; measures same hand-handle distance."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Swing Cup — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Shadow Hand Swing Cup.")
    print("Passive grasp, negative distance, velocity penalty.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Swing Cup",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="grasp_reward",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # exp(-0.25*dist) for each handle, sum - 1.0
        value_range=(-1.0, 1.0),  # idle: hands near handles = ~1.0
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="object_goal_distance_reward",
        reward_type=RewardType.PER_STEP,
        value=-0.5,  # -exp(-0.1*dist)*2, always negative
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="cup_orientation_reward",
        reward_type=RewardType.PER_STEP,
        value=-0.1,  # -(1 - dot(rot,goal)^2)*0.25
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="cup_linvel_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.05,  # -||linvel||*0.1, zero at rest
        value_range=(-0.5, 0.0),  # idle: zero velocity = zero penalty
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="touch_reward",
        reward_type=RewardType.PER_STEP,
        value=0.1,  # exp(-0.25*dist) for handles, redundant with grasp
        value_range=(0.0, 0.125),  # idle: hands near handles
        requires_action=False,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("The grasp_reward is passive: hands start near cup handles,")
    print("giving high exp(-0.25*dist) at idle. The distance reward is")
    print("always negative (-exp(-0.1*dist)*2), so moving the cup toward")
    print("the goal reduces total reward. The velocity penalty is zero")
    print("at rest, rewarding stillness. touch_reward duplicates grasp.")
    print("The reward structure actively discourages task completion.")


if __name__ == "__main__":
    run_example()
