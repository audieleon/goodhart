"""Example: Robosuite staged rewards (Zhu et al. 2020, CoRL).

Pick-and-place with 4 prerequisite-gated stages creating learning plateaus.
Source: Zhu et al. 2020, "robosuite: A Modular Simulation Framework and Benchmark for Robot Learning" (CoRL)
"""

from goodhart.presets import PRESETS
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "robosuite_staged",
    "source_paper": "Zhu et al. 2020, 'robosuite: A Modular Simulation Framework and Benchmark for Robot Learning' (CoRL)",
    "paper_url": "https://arxiv.org/abs/2009.12293",
    "source_code_url": "https://github.com/ARISE-Initiative/robosuite",
    "reward_location": "Reward structure from paper description",
    "year": 2020,
    "domain": "manipulation",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to pick and place objects. Prerequisite-gated stages create compounding learning plateaus with zero gradient signal.",
    "documented_failure": "4-stage chain (grasp->lift->hover->place) where each stage only activates after the previous succeeds, creating compounding plateaus with zero signal for later stages.",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Add distance-based shaping between stages, or use curriculum learning to unlock stages gradually",
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "staged_prerequisites": "Each reward stage gates on the previous one, creating compounding discovery bottlenecks",
    },
}


def run_example():
    model, config = PRESETS["robosuite-pick-place"]
    engine = TrainingAnalysisEngine().add_all_rules()

    print("Robosuite Pick-and-Place — Staged Rewards")
    print("=" * 50)
    print("Source: Zhu et al. 2020 (CoRL), robosuite library")
    print()
    print("Reward chain:")
    for s in model.reward_sources:
        prereq = f" (requires: {s.prerequisite})" if s.prerequisite else ""
        print(f"  {s.name:10s} {s.value:+.2f}  p(discover)={s.discovery_probability:.2f}{prereq}")
    print()

    engine.print_report(model, config)

    print()
    print("The staged structure means:")
    print("  - Agent gets ZERO signal for 'lift' until 'grasp' succeeds")
    print("  - Agent gets ZERO signal for 'hover' until 'lift' succeeds")
    print("  - A 4-stage chain with low per-stage discovery creates")
    print("    compounding plateaus")
    print()
    print("Fix: Add distance-based shaping between stages,")
    print("or use curriculum learning to unlock stages gradually.")


if __name__ == "__main__":
    run_example()
