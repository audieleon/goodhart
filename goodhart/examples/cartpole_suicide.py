"""Example: CartPole suicidal agent (Classic RL).

With a constant step penalty and no positive per-step reward,
dying at step 1 accumulates less penalty than surviving. The
agent learns to fall over immediately.

Uses the mountain-car preset which demonstrates the same
death-beats-survival pattern.

Source: Barto, Sutton & Anderson 1983 (CartPole); Sutton & Barto 2018
  (Ch. 3.4, reward hypothesis)
"""

from goodhart.presets import PRESETS
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    model, config = PRESETS["mountain-car"]
    engine = TrainingAnalysisEngine().add_all_rules()

    print("Mountain Car / CartPole Suicide Analysis")
    print("=" * 50)
    print("Source: Moore 1990 / Gymnasium classic control")
    print(f"Step penalty: {model.total_step_penalty}/step")
    print(f"Goal: {model.max_goal_reward}")
    print(f"Max steps: {model.max_steps}")
    print()

    engine.print_report(model, config)

    print("This is a 'reward desert' — all non-goal strategies")
    print("score equally. The agent has no gradient signal to")
    print("learn from until it accidentally discovers the goal.")


if __name__ == "__main__":
    run_example()
