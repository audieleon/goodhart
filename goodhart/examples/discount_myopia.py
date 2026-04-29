"""Example: Discount myopia — gamma too low for the task.

gamma=0.95 with a 500-step episode. The effective horizon is
1/(1-0.95) = 20 steps. The agent plans 20 steps ahead in a
500-step task. Rewards beyond step 20 are invisible.

Hu et al. (ICML 2022) showed that the wrong gamma makes agents
myopic to distant rewards even when the agent can physically
reach them. Lower gamma can paradoxically outperform higher
gamma due to regularization effects, but when the task
genuinely requires long-horizon planning, myopia is fatal.

Source: Hu et al. 2022, "On the Role of Discount Factor in
  Offline Reinforcement Learning" (ICML)
Tool should catch: discount_horizon_mismatch (CRITICAL)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Discount Myopia — gamma=0.95, 500-step episode")
    print("=" * 70)
    print()
    print("Source: Hu et al. 2022 (ICML)")
    print("Effective horizon = 20 steps. Episode = 500 steps.")
    print()

    model = EnvironmentModel(
        name="Discount Myopia",
        max_steps=500,
        gamma=0.95,
        n_actions=4,
        action_type="discrete",
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="task_goal",
        reward_type=RewardType.TERMINAL,
        value=10.0,
        requires_action=True,
        intentional=True,
        discovery_probability=0.02,
    ))

    engine.print_report(model)

    print()
    print("The agent can only plan 20 steps ahead. The goal at")
    print("step ~250 (median) is discounted to 0.95^250 ≈ 0.00003.")
    print("Fix: increase gamma to 0.999 (horizon = 1000 steps).")


if __name__ == "__main__":
    run_example()
