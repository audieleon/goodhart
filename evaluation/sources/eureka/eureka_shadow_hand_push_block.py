"""Eureka Shadow Hand Push Block — GPT-4 generated reward.

Passive hand-to-block proximity rewards at equal weight with active
block-to-goal rewards create 50% idle floor.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "eureka_shadow_hand_push_block",
    "source_paper": (
        'Ma et al. 2024, "Eureka: Human-Level Reward Design via Coding'
        ' Large Language Models," ICLR 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2310.12931",
    "source_code_url": (
        "https://eureka-research.github.io/assets/reward_functions/"
        "shadow_hand_push_block.txt"
    ),
    "reward_location": "GPT-4 generated reward code at https://eureka-research.github.io/assets/reward_functions/shadow_hand_push_block.txt",
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "year": 2024,
    "domain": "manipulation",
    "brief_summary": (
        "GPT-4 generated Shadow Hand Push Block reward."
        " Passive hand-proximity rewards at equal weight with"
        " active block-goal rewards create 50% idle floor."
    ),
    "documented_failure": (
        "Hand proximity rewards are passive: exp(-0.1*dist) is"
        " high when hands start near the block. With slow decay"
        " (temp 0.1), even moderate distances yield significant"
        " reward. Equal weighting (0.25 each) gives passive"
        " components 50% of total reward."
    ),
    "failure_mechanism": "idle_exploit",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Gate hand rewards on contact or reduce their weight"
        " relative to block-goal rewards."
    ),
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "left_proximity_reward": (
            "Active (requires_action=True) and intentional."
            " Measures block-to-left-goal distance."
        ),
        "right_proximity_reward": (
            "Active (requires_action=True) and intentional."
            " Measures block-to-right-goal distance."
        ),
        "left_hand_reward": (
            "Passive (requires_action=False). Hand starts near"
            " block, saturating exp(-0.1*dist) at rest."
        ),
        "right_hand_reward": (
            "Passive (requires_action=False). Hand starts near"
            " block, saturating exp(-0.1*dist) at rest."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Push Block — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Shadow Hand Push Block.")
    print("Passive hand-to-block proximity rewards create idle exploit.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Push Block",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="left_proximity_reward",
        reward_type=RewardType.PER_STEP,
        value=0.25,  # exp(-0.5*dist_block_to_left_goal)
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="right_proximity_reward",
        reward_type=RewardType.PER_STEP,
        value=0.25,  # exp(-0.5*dist_block_to_right_goal)
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="left_hand_reward",
        reward_type=RewardType.PER_STEP,
        value=0.1,  # active: hand moves away
        value_range=(0.0, 0.25),  # idle: hand starts near block
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="right_hand_reward",
        reward_type=RewardType.PER_STEP,
        value=0.1,  # active: hand moves away
        value_range=(0.0, 0.25),  # idle: hand starts near block
        requires_action=False,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("The hand proximity rewards are passive: exp(-0.1*dist) is")
    print("high when hands start near the block. With temperature 0.1")
    print("the decay is very slow, so even moderate distances yield")
    print("significant reward. The block-to-goal rewards require pushing")
    print("the block but compete with the passive hand rewards. Equal")
    print("weighting (0.25 each) gives passive components 50% of total.")


if __name__ == "__main__":
    run_example()
