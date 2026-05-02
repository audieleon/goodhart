"""Example: Tic-tac-toe bot crashing opponents (2019).

Agent won by placing at coordinates so large it crashed opponents via memory exhaustion.
Source: Lehman et al. 2020, "The Surprising Creativity of Digital Evolution"
"""

from goodhart.models import *
from goodhart.engine import *
from goodhart.rules.reward import *

METADATA = {
    "id": "tic_tac_toe_crash",
    "source_paper": "Lehman et al. 2020, 'The Surprising Creativity of Digital Evolution' (Section 3.2)",
    "paper_url": "https://arxiv.org/abs/1803.03453",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2020,
    "domain": "game_ai",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to win tic-tac-toe. Instead it placed at coordinate (10^18, 10^18), crashing the opponent via memory exhaustion to win by forfeit.",
    "documented_failure": "Agent discovered that unbounded action space coordinates could crash opponents. Placed at (10^18, 10^18) causing opponent memory bomb. Won by default forfeit.",
    "failure_mechanism": None,
    "detection_type": "dynamic",
    "discovery_stage": "post_training",
    "fix_known": "Bound the action space and validate inputs — a software engineering problem, not reward design",
    "compute_cost_class": "low",
    "is_negative_example": True,
    "encoding_rationale": {
        "out_of_scope": "Reward function (win/lose) is correct; exploit is in unbounded action space",
        "adversarial_environment": "Agent discovered its actions could affect opponent runtime, not just game state",
    },
}


def run_example():
    print("=" * 70)
    print("TIC-TAC-TOE CRASH EXPLOIT (2019)")
    print("=" * 70)
    print()

    model = EnvironmentModel(name="Tic-tac-toe crash", max_steps=50)
    model.add_reward_source(
        RewardSource(
            name="win",
            reward_type=RewardType.TERMINAL,
            value=1.0,
            discovery_probability=0.5,
        )
    )

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
