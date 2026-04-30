"""Eureka Shadow Hand Catch Abreast — GPT-4 generated reward.

Passive grasp and contact rewards create idle floor that dominates
the sparse goal bonus.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "eureka_shadow_hand_catch_abreast",
    "source_paper": (
        'Ma et al. 2024, "Eureka: Human-Level Reward Design via Coding'
        ' Large Language Models," ICLR 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2310.12931",
    "source_code_url": (
        "https://eureka-research.github.io/assets/reward_functions/"
        "shadow_hand_catch_abreast.txt"
    ),
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "year": 2024,
    "domain": "manipulation",
    "brief_summary": (
        "GPT-4 generated Shadow Hand Catch Abreast reward."
        " Passive grasp and contact rewards dominate the"
        " sparse goal bonus, creating an idle floor."
    ),
    "documented_failure": (
        "grasp_reward is passive: exp(-dist/0.08) is high when"
        " hands start near the object. smooth_contact_reward is"
        " also passive: symmetric hand positions at idle yield"
        " high reward. Together these create an idle floor that"
        " dominates the goal_reward signal."
    ),
    "failure_mechanism": "idle_exploit",
    "discovery_stage": "during_training",
    "fix_known": (
        "Make grasp_reward conditional on active grasping"
        " (contact force threshold) rather than proximity."
    ),
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "grasp_reward": (
            "Passive (requires_action=False). Hands start near"
            " object, saturating exp(-dist/0.08) at rest."
        ),
        "goal_reward": (
            "Active (requires_action=True) and intentional."
            " Measures object-goal distance."
        ),
        "smooth_contact_reward": (
            "Passive (requires_action=False). Symmetric hand"
            " positions at idle yield high reward."
        ),
        "goal_bonus": (
            "Active (requires_action=True) and intentional."
            " Sparse signal only when dist < 0.03."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Catch Abreast — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Shadow Hand Catch Abreast.")
    print("Passive grasp reward creates idle exploit.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Catch Abreast",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="grasp_reward",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # active: object moves away
        value_range=(0.0, 1.0),  # idle: hands start near object
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="goal_reward",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="smooth_contact_reward",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # exp(-|left_dist - right_dist|/0.12)
        value_range=(0.0, 1.0),  # idle: symmetric hand distances
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="goal_bonus",
        reward_type=RewardType.PER_STEP,
        value=1.0,  # sparse: only when dist < 0.03
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("The grasp_reward is passive: exp(-dist/0.08) is high when")
    print("hands start near the object. smooth_contact_reward is also")
    print("passive: symmetric hand positions at idle yield high reward.")
    print("Together these create an idle floor that may dominate the")
    print("goal_reward signal. The 25x goal_bonus helps but is sparse.")


if __name__ == "__main__":
    run_example()
