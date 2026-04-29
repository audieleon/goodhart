"""Krakovna entry: Go -- agent passes forever to maximize average score.

Source: Chew 2019
Failure: Agent learns to pass on every turn, never placing a stone.
Mechanism: Reward is average score across games. Passing forever
  results in a draw (0 points lost), which is better than the
  expected negative outcome of playing aggressively and sometimes
  losing. The agent maximizes the proxy (average score) by
  never risking a loss.
Domain: Board games
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Go pass forever (Chew 2019)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Go pass (Chew 2019)",
        max_steps=500,
        gamma=1.0,
        n_actions=362,  # 19x19 board + pass
        action_type="discrete",
        death_probability=0.0,
    )

    # Win reward
    model.add_reward_source(RewardSource(
        name="win_game",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        discovery_probability=0.3,
        intentional=True,
    ))

    # Lose penalty
    model.add_reward_source(RewardSource(
        name="lose_game",
        reward_type=RewardType.TERMINAL,
        value=-1.0,
        discovery_probability=0.3,
    ))

    # Draw by passing -- the trap
    # Passing every turn yields 0.0 terminal reward, which dominates
    # the EV of risky play when the agent is weak.
    model.add_reward_source(RewardSource(
        name="draw_by_passing",
        reward_type=RewardType.TERMINAL,
        value=0.0,
        discovery_probability=1.0,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
