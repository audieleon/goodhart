"""EME Exploration — held-out evaluation (negative example).

Wang et al. 2024 well-calibrated intrinsic motivation where bonus
is 10% of task reward. Should pass clean.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "heldout_eme_exploration",
    "source_paper": (
        'Wang, Zhao, Liu & Hou U, "Rethinking Exploration in '
        'Reinforcement Learning with Effective Metric-Based '
        'Exploration Bonus," NeurIPS 2024 (Spotlight)'
    ),
    "paper_url": "https://openreview.net/forum?id=QpKWFLtZKi",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2024,
    "domain": "navigation",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Well-calibrated intrinsic motivation setup. Intrinsic"
        " bonus (0.1) is 10% of task reward (1.0), within safe"
        " bounds."
    ),
    "documented_failure": "None — reward is well-designed.",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "unknown",
    "is_negative_example": True,
    "encoding_rationale": {
        "well_calibrated": (
            "Intrinsic bonus at 10% of task reward is within"
            " safe bounds; no structural problems."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("EME Exploration — held-out evaluation")
    print("=" * 70)
    print()
    print("Source: EME Exploration, NeurIPS 2024")
    print("Intrinsic bonus dominance test: 0.1 vs 1.0 task reward.")
    print("Well-calibrated — tool should note at INFO level at most.")
    print()

    model = EnvironmentModel(
        name="EME Exploration (Held-Out)",
        max_steps=1000,
        gamma=0.99,
        n_actions=18,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="task_reward",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="intrinsic_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.1,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("The intrinsic bonus (0.1) is 10% of the task reward (1.0),")
    print("which is well within safe bounds. The tool should pass clean")
    print("or note the intrinsic bonus at INFO level without flagging")
    print("any structural problems.")


if __name__ == "__main__":
    run_example()
