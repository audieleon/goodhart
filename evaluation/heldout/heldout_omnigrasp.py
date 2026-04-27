"""Example: Omnigrasp — held-out evaluation.

Held-out paper not used during tool development. Tests whether
the tool catches shaping dominance in a grasping environment
where dense shaping is 30x the sparse goal reward.

Source: Luo et al. 2024, "Omnigrasp: Grasping Diverse Objects
  with Simulated Humanoids" (NeurIPS 2024)

Tool should catch: shaping dominates goal (30x), making goal
  discovery nearly impossible. Paper confirms they needed
  pre-grasp guidance as a workaround.

Expected result: FAIL (CRITICAL — shaping dominates goal)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Omnigrasp — held-out evaluation")
    print("=" * 70)
    print()
    print("Source: Luo et al. 2024, NeurIPS 2024")
    print("Sparse goal + dense shaping 30x goal reward.")
    print("Paper uses pre-grasp guidance as a workaround.")
    print()

    model = EnvironmentModel(
        name="Omnigrasp (Held-Out)",
        max_steps=500,
        gamma=0.99,
        n_actions=52,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="grasp_success",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        requires_action=True,
        intentional=True,
        discovery_probability=0.02,
    ))
    model.add_reward_source(RewardSource(
        name="pre_grasp_guidance",
        reward_type=RewardType.SHAPING,
        value=30.0,
        requires_action=True,
        can_loop=True,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="approach_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        requires_action=True,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("The tool should flag that pre_grasp_guidance (30.0) is 30x")
    print("the grasp_success goal (1.0). Dense shaping at this ratio")
    print("makes goal discovery nearly impossible — the agent optimizes")
    print("the shaping signal instead of reaching the actual goal.")
    print("The paper confirms this by needing pre-grasp guidance as a")
    print("workaround for the sparse goal.")


if __name__ == "__main__":
    run_example()
