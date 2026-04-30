"""Eureka Shadow Hand Pen — GPT-4 generated pen cap removal.

Passive hands-to-handles reward creates idle floor; sharp pen-cap
sigmoid is nearly binary.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "eureka_shadow_hand_pen",
    "source_paper": (
        'Ma et al. 2024, "Eureka: Human-Level Reward Design via Coding'
        ' Large Language Models," ICLR 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2310.12931",
    "source_code_url": (
        "https://eureka-research.github.io/assets/reward_functions/"
        "shadow_hand_pen.txt"
    ),
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "year": 2024,
    "domain": "manipulation",
    "brief_summary": (
        "GPT-4 generated Shadow Hand Pen reward."
        " Passive hands-to-handles reward creates idle floor;"
        " sharp pen-cap sigmoid is nearly binary."
    ),
    "documented_failure": (
        "hands_to_handles_reward is passive: sigmoid(-5*(dist-0.1))"
        " is high when hands start near pen handles. Idle reward"
        " from proximity competes with the active pen_cap and"
        " pen_handles_pull components."
    ),
    "failure_mechanism": "idle_exploit",
    "discovery_stage": "during_training",
    "fix_known": (
        "Gate hands_to_handles_reward on actual grasp contact"
        " rather than proximity."
    ),
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "hands_to_handles_reward": (
            "Passive (requires_action=False). Hands start near"
            " handles, saturating sigmoid at rest."
        ),
        "pen_cap_reward": (
            "Active (requires_action=True) and intentional."
            " Very sharp sigmoid (temp 50), nearly binary."
        ),
        "pen_handles_pull_reward": (
            "Active (requires_action=True) and intentional."
            " Requires pulling handles apart."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Pen — GPT-4 generated pen cap reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Shadow Hand Pen.")
    print("Passive hands-to-handles reward creates idle exploit.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Pen",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="hands_to_handles_reward",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # active: hands move away
        value_range=(0.0, 0.5),  # idle: hands start near handles
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="pen_cap_reward",
        reward_type=RewardType.PER_STEP,
        value=0.5,  # sigmoid(-50*(dist-0.1)), very sharp
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="pen_handles_pull_reward",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # sigmoid(5*(handle_dist-0.1))
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("The hands_to_handles_reward is passive: sigmoid(-5*(dist-0.1))")
    print("is 0.5 when dist=0.1 and higher when closer. Hands typically")
    print("start near pen handles, giving idle reward. The pen_cap_reward")
    print("uses temperature 50 (very sharp sigmoid), meaning it is nearly")
    print("binary: full reward only when cap is within 0.1 of goal.")
    print("pen_handles_pull_reward correctly requires pulling apart.")


if __name__ == "__main__":
    run_example()
