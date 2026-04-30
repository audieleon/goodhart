"""Eureka Shadow Hand Grasp and Place — GPT-4 generated reward.

Passive hand-distance and grab-success rewards create idle floor;
inverted distance/rotation logic rewards being far from goal.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "eureka_shadow_hand_grasp_and_place",
    "source_paper": (
        'Ma et al. 2024, "Eureka: Human-Level Reward Design via Coding'
        ' Large Language Models," ICLR 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2310.12931",
    "source_code_url": (
        "https://eureka-research.github.io/assets/reward_functions/"
        "shadow_hand_grasp_and_place.txt"
    ),
    "reward_location": "GPT-4 generated reward code at https://eureka-research.github.io/assets/reward_functions/shadow_hand_grasp_and_place.txt",
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "year": 2024,
    "domain": "manipulation",
    "brief_summary": (
        "GPT-4 generated Shadow Hand Grasp and Place reward."
        " Passive hand-distance and grab-success create idle"
        " floor; inverted |exp(-x)-1| rewards distance from goal."
    ),
    "documented_failure": (
        "hand_distance_reward and grab_success are both passive:"
        " hands start near the object. instability_penalty is"
        " -exp(-linvel), penalizing stillness. Distance and"
        " rotation rewards use |exp(-x)-1| which approaches 0"
        " at success, creating inverted signal."
    ),
    "failure_mechanism": "idle_exploit",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Use direct distance reward instead of |exp(-x)-1|."
        " Gate grab_success on actual contact force."
    ),
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "hand_distance_reward": (
            "Passive (requires_action=False). Hands start near"
            " object, saturating exp(-0.1*dist) at rest."
        ),
        "grab_success": (
            "Passive (requires_action=False). Sigmoid of hand"
            " proximity; high at idle when hands start close."
        ),
        "distance_reward": (
            "Active (requires_action=True) and intentional."
            " Uses |exp(-0.05*dist)-1| which is 0 at goal,"
            " inverting the expected gradient."
        ),
        "rot_reward": (
            "Active (requires_action=True) and intentional."
            " Uses |exp(-2*rot)-1| which is 0 at goal,"
            " inverting the expected gradient."
        ),
        "instability_penalty": (
            "Passive (requires_action=False). -exp(-linvel)"
            " penalizes zero velocity (stillness)."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Grasp and Place — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Shadow Hand Grasp and Place.")
    print("Instability penalty rewards stillness, inverted reward logic.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Grasp and Place",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="hand_distance_reward",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # exp(-0.1*hand_dist)
        value_range=(0.0, 1.0),  # idle: hands start near object
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="grab_success",
        reward_type=RewardType.PER_STEP,
        value=0.5,  # sigmoid(5*(1-hand_dist))
        value_range=(0.0, 1.0),  # idle: hands near object = high
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="distance_reward",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # |exp(-0.05*dist)-1| * grab_success
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="rot_reward",
        reward_type=RewardType.PER_STEP,
        value=0.2,  # |exp(-2*rot_dist)-1| * grab_success
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="instability_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.3,  # -exp(-linvel), rewards zero velocity
        value_range=(-1.0, 0.0),  # idle: zero velocity = max penalty
        requires_action=False,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("The instability_penalty is -exp(-linvel), which is most")
    print("negative (penalty is largest) when velocity is zero. This")
    print("actually penalizes stillness. But hand_distance_reward and")
    print("grab_success are both passive, creating an idle floor.")
    print("The distance and rotation rewards use |exp(-x)-1|, which")
    print("approaches 0 as distance/rotation approach 0 (success),")
    print("creating an inverted signal that rewards being far from goal.")


if __name__ == "__main__":
    run_example()
