"""Example: Legged Gym 19-component reward (Rudin et al. 2022, RSS).

19 reward terms spanning 800,000x magnitude range; feet_air_time creates
an idle exploit and exponential velocity tracking saturates at 95%.

Source: Rudin et al. 2022, "Learning to Walk in Minutes Using Massively
  Parallel Deep RL" (RSS); reward table from Section III-B
"""

from goodhart.presets import PRESETS
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "legged_gym_rewards",
    "source_paper": "Rudin et al. 2022, Learning to Walk in Minutes Using Massively Parallel Deep RL (RSS)",
    "paper_url": "https://arxiv.org/abs/2109.11978",
    "source_code_url": "https://github.com/leggedrobotics/legged_gym",
    "reward_location": "Section III-B, reward table",
    "year": 2022,
    "domain": "locomotion",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to walk with 19-term reward. feet_air_time pays for standing still and penalty magnitudes span 800,000x.",
    "documented_failure": "feet_air_time (+1.0/step, passive) creates idle exploit; 7 penalty terms span 6 orders of magnitude (lin_vel_z at -2.0 vs torques at -0.00001); exponential tracking saturates at 95% when error < 0.75",
    "failure_mechanism": "idle_exploit",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "only_positive_rewards=True clipping and careful coefficient tuning in the real code",
    "compute_cost_class": "high",
    "is_negative_example": True,
    "encoding_rationale": {
        "idle_exploit": "feet_air_time rewards standing still passively",
        "magnitude_imbalance": "800,000x spread makes tiny penalties invisible to optimizer",
        "exponential_saturation": "exp(-error/sigma) tracking creates 'close enough' plateau",
    },
}


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
