"""Example: MultiRoom-N4-Monster reward traps.

Three degenerate equilibria (stand still, die fast at -0.01, die fast
at -0.001) all mathematically predictable before training.

Source: Original experiments by the author on MiniHack
  MultiRoom-N4-Monster (Samvelyan et al. 2021). Three configurations
  tested; all failures predicted by goodhart before training.
"""

from goodhart.models import *
from goodhart.engine import *
from goodhart.rules.reward import *
from goodhart.rules.training import *

METADATA = {
    "id": "multiroom_traps",
    "source_paper": "Original experiments by the author on MiniHack MultiRoom-N4-Monster (Samvelyan et al. 2021)",
    "paper_url": None,
    "source_code_url": None,
    "reward_location": "Reward structure from original experiments",
    "year": 2021,
    "domain": "navigation",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to navigate 4 rooms. Instead three different penalty configs all produced degenerate equilibria (stand still, die fast).",
    "documented_failure": "Three configurations all trapped: (1) default penalty -- agent stands still; (2) always -0.01/step -- dying at step 1 costs -0.01 vs exploring 120 steps costs -1.20; (3) always -0.001/step -- still trapped because p(goal) ~1.1% < required 11.4%",
    "failure_mechanism": "penalty_dominates_goal",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "penalty <= 0.0001, or RND coeff >= 0.05, or no step penalty at all",
    "compute_cost_class": "low",
    "is_negative_example": True,
    "encoding_rationale": {
        "triple_failure": "Three configurations all produce mathematically predictable degenerate equilibria",
        "original_experiments": "Failures predicted by goodhart before training and confirmed empirically",
    },
}


def run_example():
    print("=== TRAP 1: Standing still (default penalty) ===")
    print("Observed: standing still gave 0.0 reward because")
    print("wait/search actions don't incur the step penalty.")
    print()

    model1 = EnvironmentModel(
        name="MultiRoom (default penalty — stand still)",
        max_steps=120,
        n_states=2000,
        n_actions=8,
        death_probability=0.01,
    )
    model1.add_reward_source(RewardSource(
        name="goal", reward_type=RewardType.TERMINAL, value=1.0,
        discovery_probability=0.01,
    ))
    model1.add_reward_source(RewardSource(
        name="wasted step penalty", reward_type=RewardType.PER_STEP,
        value=-0.01, requires_action=True,
    ))
    TrainingAnalysisEngine().add_all_rules().print_report(model1)

    print()
    print("=== TRAP 2: Dying fast (-0.01/step always) ===")
    print("We fixed stand-still by penalizing ALL steps.")
    print("But now dying at step 1 costs -0.01, while exploring")
    print("120 steps costs -1.20. Dying is 120x cheaper.")
    print()

    model2 = EnvironmentModel(
        name="MultiRoom (always -0.01 — die fast)",
        max_steps=120,
        n_states=2000,
        n_actions=8,
        death_probability=0.01,
    )
    model2.add_reward_source(RewardSource(
        name="goal", reward_type=RewardType.TERMINAL, value=1.0,
        discovery_probability=0.01,
    ))
    model2.add_reward_source(RewardSource(
        name="step penalty", reward_type=RewardType.PER_STEP,
        value=-0.01,
    ))
    TrainingAnalysisEngine().add_all_rules().print_report(model2)

    print()
    print("=== TRAP 3: Still dying fast (-0.001/step) ===")
    print("We reduced penalty by 10x. Math says we need")
    print("p(goal) > 11.4% but random walk gets ~1.1%.")
    print("Still trapped — just less aggressively.")
    print()

    model3 = EnvironmentModel(
        name="MultiRoom (always -0.001 — still trapped)",
        max_steps=120,
        n_states=2000,
        n_actions=8,
        death_probability=0.01,
    )
    model3.add_reward_source(RewardSource(
        name="goal", reward_type=RewardType.TERMINAL, value=1.0,
        discovery_probability=0.01,
    ))
    model3.add_reward_source(RewardSource(
        name="step penalty", reward_type=RewardType.PER_STEP,
        value=-0.001,
    ))
    TrainingAnalysisEngine().add_all_rules().print_report(model3)

    print()
    print("=== WHAT WOULD HAVE WORKED ===")
    print("The framework tells us we need either:")
    print("  - penalty ≤ 0.0001 (so exploration cost < 1% of goal)")
    print("  - RND coeff ≥ 0.05 (intrinsic reward > step penalty)")
    print("  - No step penalty at all (survival environments)")
    print()
    print("We burned ~8 hours of compute learning this. The")
    print("framework computes it in milliseconds.")


if __name__ == "__main__":
    run_example()
