"""Example: CoinRun — goal misgeneralization (LIMITATION).

Reward structure is correct (+10 for coin) but agent learns "go right"
from training distribution bias. Beyond structural reward analysis.

Source: Langosco et al. 2022 (ICML), Cobbe et al. 2019 (CoinRun)
"""

from goodhart.presets import PRESETS
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "coinrun_misgeneralization",
    "source_paper": "Langosco et al. 2022, 'Goal Misgeneralization in Deep RL' (ICML); Cobbe et al. 2019 (CoinRun)",
    "paper_url": None,
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2022,
    "domain": "game_ai",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to collect coins. Instead it learned 'go right' from training distribution bias and ignores coins at test time.",
    "documented_failure": "Agent learns spurious correlation ('go right') instead of the intended goal ('collect coin') because training levels always place the coin on the right. At test time on new levels, the agent ignores the coin.",
    "failure_mechanism": None,
    "detection_type": "dynamic",
    "discovery_stage": "post_training",
    "fix_known": "Diverse training distributions, procedural generation with varied coin placement",
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "clean_reward": "Reward structure is correct; failure is distributional, not structural",
    },
}


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
