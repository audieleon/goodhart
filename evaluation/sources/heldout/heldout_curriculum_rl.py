"""CurriculumRL — held-out evaluation.

Freitag et al. 2025 sparse goal with constraint penalties making
random discovery extremely unlikely. Exploration threshold warning.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "heldout_curriculum_rl",
    "source_paper": (
        'Freitag, Ceder, Laezza, Akesson & Haghir Chehreghani, '
        '"Curriculum Reinforcement Learning for Complex Reward '
        'Functions," EWRL 2025'
    ),
    "paper_url": "https://openreview.net/forum?id=DHOxjoy1sP",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2025,
    "domain": "navigation",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Agent was supposed to reach a sparse goal via curriculum."
        " Instead, constraint penalties dominate exploration,"
        " making random goal discovery nearly impossible."
    ),
    "documented_failure": (
        "Sparse goal with only 0.5% random discovery probability"
        " combined with per-step constraint penalty (-0.05) means"
        " random episodes accumulate large negative returns. Agent"
        " must explore past heavy penalties to find the rare goal,"
        " requiring ~99% of training budget just to see enough"
        " goal examples. Constraint exploitation possible."
    ),
    "failure_mechanism": "exploration_threshold",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Curriculum provides partial credit to bridge the"
        " exploration gap."
    ),
    "compute_cost_class": "unknown",
    "is_negative_example": False,
    "encoding_rationale": {
        "sparse_goal": (
            "0.5% discovery probability with heavy penalties"
            " makes random exploration impractical."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("CurriculumRL — held-out evaluation")
    print("=" * 70)
    print()
    print("Source: CurriculumRL, EWRL 2025")
    print("Sparse goal (0.5% discovery) + constraint penalties.")
    print("Constraint exploitation possible; exploration threshold warning.")
    print()

    model = EnvironmentModel(
        name="CurriculumRL (Held-Out)",
        max_steps=500,
        gamma=0.99,
        n_actions=6,
        action_type="continuous",
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="task_goal",
        reward_type=RewardType.TERMINAL,
        value=10.0,
        requires_action=True,
        intentional=True,
        discovery_probability=0.005,
    ))
    model.add_reward_source(RewardSource(
        name="progress_reward",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        requires_action=True,
        intentional=True,
        explore_fraction=0.5,  # curriculum provides partial credit
    ))
    model.add_reward_source(RewardSource(
        name="constraint_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.05,
        requires_action=True,  # penalty for constraint violation during action
    ))
    model.add_reward_source(RewardSource(
        name="shaping_bonus",
        reward_type=RewardType.SHAPING,
        value=0.3,
        requires_action=True,
        can_loop=False,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("The tool should warn about exploration threshold: the sparse")
    print("goal has only 0.5% random discovery probability, and the")
    print("per-step constraint penalty (-1.0) means random episodes")
    print("accumulate large negative returns. The agent must explore")
    print("past heavy penalties to find the rare goal — requiring ~99%")
    print("of training budget just to see enough goal examples.")


if __name__ == "__main__":
    run_example()
