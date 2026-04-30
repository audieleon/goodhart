"""MultiRoom-N4-Monster: the experiments that created goodhart.

Three reward configurations, three degenerate equilibria, all
mathematically predictable. These failures drove the creation
of the goodhart tool and paper.
"""

from goodhart.models import *
from goodhart.engine import *
from goodhart.rules.reward import *
from goodhart.rules.training import *

METADATA = {
    "id": "multiroom_traps",
    "source_paper": (
        "Original experiments by the author (Sheridan 2026) on MiniHack "
        "MultiRoom-N4-Monster (Samvelyan et al. 2021). These reward "
        "failures were the direct motivation for building goodhart: "
        "the author wanted to explain the failures mathematically "
        "and prevent them before spending compute."
    ),
    "paper_url": "https://github.com/audieleon/CognitiveAI",
    "source_code_url": "https://github.com/audieleon/CognitiveAI/tree/main/minihack_sf",
    "reward_location": (
        "CognitiveAI repo: minihack_sf/train.py (reward config), "
        "research_log/ entries 048-052 (lab notes documenting each "
        "failure and the mathematical analysis that followed)"
    ),
    "year": 2026,
    "domain": "navigation",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Agent was supposed to navigate 4 rooms to reach an exit. "
        "Three different penalty configurations all produced degenerate "
        "equilibria (stand still or die fast). These failures drove "
        "the creation of the goodhart tool."
    ),
    "documented_failure": (
        "Three configurations all trapped: (1) default penalty: agent "
        "stands still because idle EV > explore EV; (2) -0.01/step: "
        "dying at step 1 costs -0.01 vs exploring 120 steps costs "
        "-1.20, so dying is 120x cheaper; (3) -0.001/step: still "
        "trapped because p(goal) ~1.1% < required 11.4% for positive "
        "EV exploration. All three were mathematically predictable "
        "from the reward structure alone."
    ),
    "failure_mechanism": "penalty_dominates_goal",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Removed step penalty entirely, added PBRS door-distance "
        "shaping, room transition bonus, and alive bonus. Agent "
        "progressed from 0.000 score to 1.37 (first room transition). "
        "The fix was designed using goodhart's own rules."
    ),
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "triple_failure": (
            "Three configurations all produce mathematically predictable "
            "degenerate equilibria. Each failure maps to a specific "
            "goodhart rule: idle_exploit, death_beats_survival, "
            "penalty_dominates_goal."
        ),
        "genesis": (
            "These experiments are the origin of the goodhart project. "
            "The author experienced these failures firsthand, analyzed "
            "them mathematically, and built the tool to catch them "
            "before training."
        ),
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
