"""Example: CoinRun — goal misgeneralization (LIMITATION).

CoinRun's reward structure is clean: +10 for collecting the coin.
But the agent learns "go right" instead of "collect coin" because
the coin is always on the right side of training levels. At test
time on new levels, the agent ignores the coin and walks right.

LIMITATION: goodhart cannot detect goal misgeneralization. The
reward structure IS correct — the problem is in the training
distribution, not the reward function. Our model has no concept
of "what the agent actually learns to represent" vs "what the
reward incentivizes."

This is a fundamental boundary: static reward analysis catches
structural traps but not distributional ones.

Source: Langosco et al. 2022 ("Goal Misgeneralization in Deep RL",
ICML), Cobbe et al. 2019 (CoinRun)
"""

from goodhart.presets import PRESETS
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    model, config = PRESETS["coinrun"]
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("CoinRun — goal misgeneralization (TOOL LIMITATION)")
    print("=" * 70)
    print()
    print("Source: Langosco et al. 2022 (ICML), Cobbe et al. 2019")
    print()
    print("The agent learns 'go right' instead of 'collect coin'")
    print("because training levels always place the coin on the right.")
    print("At test time, it ignores coins placed elsewhere.")
    print()
    print("What goodhart says:")
    print()

    result = engine.print_report(model, config)

    print()
    print("=" * 70)
    print("LIMITATION: goodhart gives a mostly-clean result because")
    print("the reward structure IS correct. The coin reward (+10)")
    print("perfectly captures the designer's intent.")
    print()
    print("The real problem is distributional:")
    print("  - Training: coin always at rightmost position")
    print("  - Test: coin at random positions")
    print("  - Agent learns a spurious correlation, not the reward")
    print()
    print("What would be needed to catch this:")
    print("  - State distribution analysis (not just reward structure)")
    print("  - Feature attribution on the learned policy")
    print("  - Train/test distribution comparison")
    print()
    print("This is OUTSIDE goodhart's scope by design. We analyze")
    print("the reward function, not the learning dynamics.")


if __name__ == "__main__":
    run_example()
