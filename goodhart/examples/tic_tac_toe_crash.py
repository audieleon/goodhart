"""Example: Tic-tac-toe bot crashing opponents (2019).

An agent playing N-in-a-row tic-tac-toe learned to win by
placing a piece at a coordinate so large it caused other
bots to crash when expanding their board model.

This is an ADVERSARIAL ENVIRONMENT EXPLOIT — the agent
discovered that its action space could affect the opponent's
runtime, not just the game state.

The framework cannot catch this (it's not a reward issue)
but including it documents the boundary clearly.
"""

from goodhart.models import *
from goodhart.engine import *
from goodhart.rules.reward import *


def run_example():
    print("=" * 70)
    print("TIC-TAC-TOE CRASH EXPLOIT (2019)")
    print("=" * 70)
    print()

    model = EnvironmentModel(name="Tic-tac-toe crash", max_steps=50)
    model.add_reward_source(RewardSource(
        name="win", reward_type=RewardType.TERMINAL,
        value=1.0, discovery_probability=0.5,
    ))

    engine = AnalysisEngine().add_all_rules()
    result = engine.analyze(model)

    print("Framework analysis:")
    if not any(v.severity == Severity.CRITICAL for v in result.verdicts):
        print("  No reward issues detected (correct — this is not")
        print("  a reward problem).")
    print()
    print("What happened:")
    print("  The agent played coordinate (10^18, 10^18). The")
    print("  opponent tried to expand its board to that size,")
    print("  ran out of memory, and crashed. The agent won by")
    print("  default — opponent forfeit.")
    print()
    print("Why the framework can't catch this:")
    print("  The reward function (win/loss) is correct. The")
    print("  exploit is in the ACTION SPACE — unbounded")
    print("  coordinates. No amount of reward analysis predicts")
    print("  that a valid action can crash the opponent.")
    print()
    print("The fix:")
    print("  Bound the action space. Validate inputs. This is a")
    print("  software engineering problem, not a reward design")
    print("  problem.")
    print()
    print("Category: adversarial environment exploit")
    print("Framework status: CORRECTLY OUT OF SCOPE")


if __name__ == "__main__":
    run_example()
