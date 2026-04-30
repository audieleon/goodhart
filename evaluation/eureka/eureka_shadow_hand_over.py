"""Eureka Shadow Hand Over — GPT-4 generated hand-over reward.

Toss-catch structure with active components; passes clean despite
unusual catch formulation.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "eureka_shadow_hand_over",
    "source_paper": (
        'Ma et al. 2024, "Eureka: Human-Level Reward Design via Coding'
        ' Large Language Models," ICLR 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2310.12931",
    "source_code_url": (
        "https://eureka-research.github.io/assets/reward_functions/"
        "shadow_hand_over.txt"
    ),
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "year": 2024,
    "domain": "manipulation",
    "brief_summary": (
        "GPT-4 generated Shadow Hand Over reward."
        " Passes clean."
    ),
    "documented_failure": "None — reward is well-designed",
    "failure_mechanism": None,
    "discovery_stage": None,
    "fix_known": None,
    "compute_cost_class": "low",
    "is_negative_example": True,
    "encoding_rationale": {
        "toss_reward": (
            "Active (requires_action=True) and intentional."
            " Measures position distance to goal."
        ),
        "catch_reward": (
            "Active (requires_action=True). Unusual formulation"
            " comparing goal position to velocity, but requires"
            " action to generate velocity signal."
        ),
        "penalty": (
            "Active (requires_action=True). Proportional to"
            " catch_reward scaled by fingertip contact."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Over — GPT-4 generated hand-over reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Shadow Hand Over.")
    print("Semantically wrong catch_reward and unusual penalty structure.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Over",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="toss_reward",
        reward_type=RewardType.PER_STEP,
        value=0.5,  # exp(-10*pos_dist/1.5)
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="catch_reward",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # exp(-30*||goal_pos - linvel||/0.5), wrong semantics
        requires_action=True,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.2,  # sigmoid(fingertip_count * 10) * 0.5 * catch_reward
        requires_action=True,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("The catch_reward is exp(-30*||goal_pos - object_linvel||/0.5),")
    print("which compares a position vector to a velocity vector. These")
    print("have different units and semantics — this is a GPT-4 error.")
    print("The penalty is proportional to catch_reward * sigmoid of")
    print("fingertip contact count, creating a coupled structure that")
    print("penalizes catching when fingertips touch the object.")


if __name__ == "__main__":
    run_example()
