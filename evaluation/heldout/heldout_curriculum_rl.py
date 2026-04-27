"""Example: CurriculumRL — held-out evaluation.

Held-out paper not used during tool development. Tests whether
the tool warns about exploration threshold issues when a sparse
goal has high constraint penalties making random discovery
extremely unlikely.

Source: CurriculumRL (EWRL 2025) — curriculum learning for RL

Expected result: WARN. The sparse goal with high constraint
  penalties means random exploration is very unlikely to reach
  the goal. Tool should warn about exploration threshold.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


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
