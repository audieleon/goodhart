"""Eureka Ant — GPT-4 generated locomotion reward.

Shaping with no terminal goal lets the agent cycle the signal forever.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "eureka_ant",
    "source_paper": (
        'Ma et al. 2024, "Eureka: Human-Level Reward Design via Coding'
        ' Large Language Models," ICLR 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2310.12931",
    "source_code_url": (
        "https://eureka-research.github.io/assets/reward_functions/"
        "ant.txt"
    ),
    "reward_location": "GPT-4 generated reward code at https://eureka-research.github.io/assets/reward_functions/ant.txt",
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "year": 2024,
    "domain": "locomotion",
    "brief_summary": (
        "GPT-4 generated Ant locomotion reward. Forward-progress"
        " shaping has no terminal goal, so the agent cycles"
        " the shaping signal indefinitely."
    ),
    "documented_failure": (
        "forward_progress shaping has no terminal goal to anchor"
        " it. The agent earns reward for moving forward but has no"
        " incentive to reach any destination — it cycles the"
        " shaping signal forever."
    ),
    "failure_mechanism": "shaping_loop",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Add a terminal goal reward that the shaping leads toward,"
        " so shaping is consumed on arrival rather than cycled."
    ),
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "forward_progress": (
            "Shaping term with no terminal anchor. Active"
            " (requires_action=True) but can be cycled because"
            " nothing absorbs the potential on arrival."
        ),
        "height_bonus": (
            "Passive (requires_action=False) and unintentional."
            " Standing upright earns reward without locomotion."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Ant — GPT-4 generated locomotion reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Ant task.")
    print()

    model = EnvironmentModel(
        name="Eureka Ant",
        max_steps=1000,
        gamma=0.99,
        n_actions=8,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="forward_progress",
        reward_type=RewardType.SHAPING,
        value=1.0,
        requires_action=True,
        can_loop=False,
    ))
    model.add_reward_source(RewardSource(
        name="forward_velocity",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        value_type="exponential",
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="height_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        requires_action=False,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("The shaping term forward_progress has no terminal goal to")
    print("anchor it. The agent earns reward for moving forward but")
    print("has no incentive to reach any destination — it will cycle")
    print("the shaping signal forever. This is the encoding shown in")
    print("the Eureka paper appendix (Section: Encoding Workflow).")


if __name__ == "__main__":
    run_example()
