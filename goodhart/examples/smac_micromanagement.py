"""Example: SMAC StarCraft Micromanagement (Samvelyan et al. 2019).

Asymmetric reward scaling (+10 enemy kill vs -5 ally kill) incentivizes hyper-aggressive strategies.
Source: Samvelyan et al. 2019, "The StarCraft Multi-Agent Challenge" (AAMAS)
"""

from goodhart.presets import PRESETS
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "smac_micromanagement",
    "source_paper": "Samvelyan et al. 2019, 'The StarCraft Multi-Agent Challenge' (AAMAS)",
    "paper_url": "https://arxiv.org/abs/1902.04043",
    "source_code_url": "https://github.com/oxwhirl/smac",
    "reward_location": "Section 3.2",
    "year": 2019,
    "domain": "multi_agent",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to manage units in combat. Asymmetric scaling (enemy kill=+10 vs ally kill=-5) incentivizes sacrificing allies for kills.",
    "documented_failure": "Killing enemies valued 2x over protecting allies incentivizes hyper-aggressive strategies. Dead allies stop accumulating damage penalty, creating a death-beats-survival incentive.",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "asymmetric_scaling": "Enemy kill reward is 2x the ally death penalty, biasing toward aggression",
        "death_stops_penalty": "Dead allies stop accumulating damage_received penalty",
    },
}


def run_example():
    model, config = PRESETS["smac"]
    engine = TrainingAnalysisEngine().add_all_rules()

    print("SMAC StarCraft Micromanagement")
    print("=" * 50)
    print("Source: Samvelyan et al. 2019 (NeurIPS), oxwhirl/smac")
    print()
    print("Reward components:")
    for s in model.reward_sources:
        print(f"  {s.name:20s} {s.value:+6.1f} ({s.reward_type.value})")
    print()
    print("Note: ally_killed at -5.0 vs enemy_killed at +10.0")
    print("means killing is 2x more rewarding than protecting.")
    print()

    engine.print_report(model, config)


if __name__ == "__main__":
    run_example()
