"""Example: CoastRunners reward loop (Clark & Amodei 2016, OpenAI).

The agent learned to go in circles collecting respawning turbo
powerups instead of finishing the race. It scored ~20% higher
than human players who actually completed the course.

This example uses the coast-runners preset (sourced from the
OpenAI blog post) and shows how the respawning_exploit rule
catches the loop before training.

Source: Clark & Amodei 2016, "Faulty Reward Functions in the Wild"
  (OpenAI Blog)
"""

from goodhart.presets import PRESETS
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    model, config = PRESETS["coast-runners"]
    engine = TrainingAnalysisEngine().add_all_rules()

    print("CoastRunners Reward Analysis")
    print("=" * 50)
    print("Source: Clark & Amodei 2016 (OpenAI blog)")
    print("Reward sources:")
    for s in model.reward_sources:
        print(f"  {s.name}: {s.value:+.1f} ({s.reward_type.value})")
    print()

    engine.print_report(model, config)

    print("The framework catches this because looping the")
    print("turbo powerup earns far more than finishing the race.")
    print()
    print("Fix: Cap turbo powerups at a few occurrences,")
    print("or restructure scoring to weight race completion higher.")


if __name__ == "__main__":
    run_example()
