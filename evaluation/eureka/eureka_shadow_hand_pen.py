"""Example: Eureka Shadow Hand Pen — GPT-4 generated pen cap removal.

Eureka uses GPT-4 to write reward functions for Isaac Gym tasks.
The Pen reward has passive hands_to_handles_reward: hands start
near pen handles, so sigmoid(-5*(dist-0.1)) is high at idle. The
pen_handles_pull_reward rewards separation of handles, which also
requires action. The pen_cap_reward correctly measures cap-to-goal.

Source: Ma et al. 2024 (ICLR), Eureka project — GPT-4 generated reward
Tool should catch: passive hands_to_handles_reward (critical)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


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
