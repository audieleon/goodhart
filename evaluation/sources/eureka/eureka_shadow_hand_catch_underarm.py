"""Eureka Shadow Hand Catch Underarm — GPT-4 generated reward.

Passive fingertip reward and velocity penalty reward stillness,
creating an idle-dominant strategy.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "eureka_shadow_hand_catch_underarm",
    "source_paper": (
        'Ma et al. 2024, "Eureka: Human-Level Reward Design via Coding'
        ' Large Language Models," ICLR 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2310.12931",
    "source_code_url": (
        "https://eureka-research.github.io/assets/reward_functions/"
        "shadow_hand_catch_underarm.txt"
    ),
    "reward_location": "GPT-4 generated reward code at https://eureka-research.github.io/assets/reward_functions/shadow_hand_catch_underarm.txt",
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "year": 2024,
    "domain": "manipulation",
    "brief_summary": (
        "GPT-4 generated Shadow Hand Catch Underarm reward."
        " Passive fingertip proximity and velocity penalty"
        " reward stillness, dominating active components."
    ),
    "documented_failure": (
        "fingertip_reward is passive: fingertips start near the"
        " object, giving high exp(-dist/0.005) at idle."
        " vel_reward explicitly rewards zero velocity. Together"
        " these passive components (0.50/step at idle) compete"
        " with active pos_reward and rot_reward."
    ),
    "failure_mechanism": "idle_exploit",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Remove vel_reward or gate it on successful catch."
        " Make fingertip_reward require contact force."
    ),
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "pos_reward": (
            "Active (requires_action=True) and intentional."
            " Measures position distance to goal."
        ),
        "rot_reward": (
            "Active (requires_action=True) and intentional."
            " Measures rotation distance to goal."
        ),
        "fingertip_reward": (
            "Passive (requires_action=False). Fingertips start"
            " near object, saturating at rest."
        ),
        "vel_reward": (
            "Passive (requires_action=False). exp(-(vel)/0.1)"
            " is maximal at zero velocity, rewarding stillness."
        ),
        "catch_bonus": (
            "Active (requires_action=True) and intentional."
            " 2x multiplier when pos_dist < 0.05."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Catch Underarm — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Shadow Hand Catch Underarm.")
    print("Passive fingertip reward and velocity penalty create idle exploit.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Catch Underarm",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="pos_reward",
        reward_type=RewardType.PER_STEP,
        value=0.25,  # exp(-pos_dist/0.1)
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="rot_reward",
        reward_type=RewardType.PER_STEP,
        value=0.25,  # exp(-rot_dist/0.005)
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="fingertip_reward",
        reward_type=RewardType.PER_STEP,
        value=0.1,  # active: fingertips move away
        value_range=(0.0, 0.25),  # idle: fingertips start near object
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="vel_reward",
        reward_type=RewardType.PER_STEP,
        value=0.25,  # exp(-(linvel+angvel)/0.1), max at zero velocity
        value_range=(0.0, 0.25),  # idle: zero velocity = max reward
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="catch_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.5,  # 2x multiplier when pos_dist < 0.05
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("The fingertip_reward is passive: fingertips start near the")
    print("object, giving high exp(-dist/0.005) at idle. The vel_reward")
    print("explicitly rewards zero velocity (stillness). Together these")
    print("two passive components (0.25 + 0.25 = 0.50/step at idle)")
    print("compete with the active pos_reward and rot_reward. The catch")
    print("bonus provides 2x multiplier but requires getting close first.")


if __name__ == "__main__":
    run_example()
