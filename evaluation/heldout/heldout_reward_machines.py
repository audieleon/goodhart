"""Example: Reward Machines — held-out evaluation.

Held-out paper not used during tool development. Tests whether
the tool correctly passes an environment whose issues are
non-Markovian noise — outside the scope of structural analysis.

Source: Reward Machines (NeurIPS 2024) — non-Markovian reward
  specification

Expected result: PASS (warnings only, no criticals). The tool
  flags exploration threshold as a warning since it cannot model
  the automaton's guided structure. The paper's actual issues
  are about non-Markovian noise — outside static analysis scope.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Reward Machines — held-out evaluation")
    print("=" * 70)
    print()
    print("Source: Reward Machines, NeurIPS 2024")
    print("Non-Markovian reward specification.")
    print("Issues are outside structural scope — tool should pass.")
    print()

    model = EnvironmentModel(
        name="Reward Machines (Held-Out)",
        max_steps=200,
        gamma=0.99,
        n_actions=4,
        action_type="discrete",
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="subtask_completion",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        requires_action=True,
        intentional=True,
        discovery_probability=0.5,
    ))
    model.add_reward_source(RewardSource(
        name="step_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.001,
        requires_action=False,
    ))

    engine.print_report(model)

    print()
    print("The tool passes (no criticals) but flags an exploration")
    print("threshold warning — it cannot model the reward machine's")
    print("automaton structure that guides the agent through subtasks.")
    print("This is a reasonable false positive: static analysis sees a")
    print("sparse goal but cannot know about the non-Markovian guidance.")
    print("The paper's actual issues involve non-Markovian reward")
    print("dependencies, which are outside what static structural")
    print("analysis can detect.")


if __name__ == "__main__":
    run_example()
