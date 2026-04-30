"""Example: Autonomous driving reward tradeoffs.

Crash penalties too small relative to speed reward cause agents to
weave aggressively and accept periodic collisions as optimal.

Source: Li et al. 2022 (MetaDrive, NeurIPS); Leurent 2018 (highway-env)
"""

from goodhart.presets import PRESETS
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "driving_safety",
    "source_paper": "Li et al. 2022 (MetaDrive, NeurIPS); Leurent 2018 (highway-env)",
    "paper_url": "https://arxiv.org/abs/2109.12674",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2022,
    "domain": "driving",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to drive safely. Instead it weaves aggressively because 2.5 steps of speed reward offsets a crash penalty.",
    "documented_failure": "In highway-env, -1.0 crash penalty vs +0.4/step speed means 2.5 steps of driving offsets a crash. Agents learn aggressive weaving and accept periodic collisions as optimal.",
    "failure_mechanism": "penalty_dominance",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Increase crash penalty relative to accumulated progress reward, not just per-step reward",
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "crash_penalty_ratio": "Crash penalty -1.0 is offset by just 2.5 steps of +0.4 speed reward",
    },
}


def run_example():
    print("Driving Reward Safety Analysis")
    print("=" * 50)

    for name in ["highway-env", "metadrive"]:
        model, config = PRESETS[name]
        engine = TrainingAnalysisEngine().add_all_rules()

        print(f"\n--- {model.name} ---")
        print(f"Source: {'Leurent 2018' if 'highway' in name else 'Li et al. 2021'}")
        for s in model.reward_sources:
            extra = ""
            if s.modifies:
                extra = f" [modifies: {s.modifies}]"
            if s.state_dependent:
                extra += " [state-dep]"
            print(f"  {s.name:20s} {s.value:+.1f} ({s.reward_type.value}){extra}")
        print()
        result = engine.analyze(model, config)
        for v in result.verdicts:
            icon = {"critical": "X", "warning": "!", "info": "i"}[v.severity.value]
            print(f"  [{icon}] {v.rule_name}: {v.message[:70]}")

    print()
    print("Key insight: crash penalties must be large relative to")
    print("accumulated progress reward, not just per-step reward.")
    print("A -5.0 crash is large per-step but small per-episode")
    print("when speed reward accumulates +0.4 * 1000 steps = 400.")


if __name__ == "__main__":
    run_example()
