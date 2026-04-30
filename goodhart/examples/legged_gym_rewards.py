"""Example: Legged Gym 19-component reward (Rudin et al. 2022, RSS).

Massively parallel PPO for ANYmal quadruped locomotion. The reward
has 19 terms including exponential velocity tracking, alive bonus,
and 7 penalty terms spanning 6 orders of magnitude.

Demonstrates three new analysis capabilities:
1. exponential_saturation: exp(-error/sigma) tracking plateaus
2. reward_dominance_imbalance: 800,000x magnitude spread
3. idle_exploit: feet_air_time pays for standing still

Source: Rudin et al. 2022, "Learning to Walk in Minutes Using Massively
  Parallel Deep RL" (RSS); reward table from Section III-B
"""

from goodhart.presets import PRESETS
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    model, config = PRESETS["legged-gym"]
    engine = TrainingAnalysisEngine().add_all_rules()

    print("Legged Gym ANYmal — 19 Reward Components")
    print("=" * 50)
    print("Source: Rudin et al. 2022 (RSS), leggedrobotics/legged_gym")
    print()
    print("Reward components:")
    for s in model.reward_sources:
        extra = ""
        if s.value_type != "constant":
            extra = f" [{s.value_type}]"
        if s.state_dependent:
            extra += " [state-dep]"
        print(f"  {s.name:25s} {s.value:+12.7f}{extra}")
    print()

    engine.print_report(model, config)

    print()
    print("Key findings:")
    print("  1. feet_air_time (+1.0/step, passive) creates an idle exploit")
    print("     The real code mitigates with only_positive_rewards=True clipping")
    print()
    print("  2. Penalty magnitudes span 800,000x:")
    print("     lin_vel_z at -2.0 vs torques at -0.00001")
    print("     Tiny penalties are invisible to the optimizer")
    print()
    print("  3. Exponential tracking (sigma=0.25) saturates at 95%")
    print("     when error < 0.75, creating a 'close enough' plateau")
    print()
    print("The Rudin et al. paper addresses these through careful")
    print("reward coefficient tuning and positive-reward clipping.")


if __name__ == "__main__":
    run_example()
