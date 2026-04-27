"""Example: Autonomous driving reward tradeoffs.

MetaDrive and highway-env both face the same tension:
progress/speed rewards incentivize fast driving, while
crash penalties are often too small to prevent aggressive
behavior.

In highway-env, -1.0 crash penalty vs +0.4/step speed
means 2.5 steps of driving offsets a crash. Agents learn
to weave aggressively and accept periodic collisions.

MetaDrive adds a lateral factor modifier that scales the
driving reward by lane position — demonstrating the
reward modifier system.
"""

from goodhart.presets import PRESETS
from goodhart.engine import TrainingAnalysisEngine


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
