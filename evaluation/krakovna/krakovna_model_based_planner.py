"""Krakovna entry: Model-based planner exploiting learned world model.

Source: Mishra et al 2017
Failure: Agent plans through a learned world model and exploits the
  worst-modeled regions to achieve high predicted reward that does
  not transfer to the real environment.
Mechanism: Per-step task reward optimized via a learned dynamics model.
  The agent finds trajectories where the model's prediction errors
  are systematically biased upward, yielding high predicted return
  but poor real-world performance.
Domain: Model-based RL / control

Structural encoding: We encode the task reward as a per-step signal.
  The reward structure itself is straightforward. The exploit occurs
  because the agent optimizes through a learned model that has
  systematic prediction errors in underexplored regions.

Advisory: learned_reward — the failure is in the learned dynamics
  model, not the reward specification. The planner optimizes reward
  correctly, but the model it plans through is wrong in exploitable
  ways. This is outside the scope of reward structure analysis.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Model-based planner (Mishra et al 2017)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Model-Based Planner (Mishra et al 2017)",
        max_steps=200,
        gamma=0.99,
        n_actions=4,
        action_type="continuous",
    )

    # Per-step task reward — the actual reward signal is simple.
    # The problem is that the agent plans through a learned model
    # that can be exploited, not that the reward is misspecified.
    model.add_reward_source(RewardSource(
        name="task_reward",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
        max_occurrences=0,
    ))

    engine.print_report(model)

    print()
    print("Advisory: learned_reward — the reward function itself is not")
    print("misspecified. The agent exploits the learned dynamics model,")
    print("finding trajectories where model prediction errors are biased")
    print("upward. This is a model-based RL failure mode, not a reward")
    print("structure problem. Structural analysis cannot detect this.")


if __name__ == "__main__":
    run_example()
