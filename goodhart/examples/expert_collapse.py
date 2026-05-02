"""Example: Expert collapse in multi-specialist models.

Without load balancing or routing floors, multi-specialist models
collapse to using one expert. Three documented cases, same fix.

Source: Shazeer et al. 2017 (ICLR)
"""

from goodhart.models import *
from goodhart.engine import *
from goodhart.rules.reward import *
from goodhart.rules.training import *

METADATA = {
    "id": "expert_collapse",
    "source_paper": "Shazeer et al. 2017, 'Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer' (ICLR)",
    "paper_url": "https://arxiv.org/abs/1701.06538",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2017,
    "domain": "multi_agent",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Specialists were supposed to share load. Instead most experts receive zero traffic without load balancing constraints.",
    "documented_failure": "Without load balancing, MoE models collapse to using a single expert. Three cases: Shazeer (most experts zero traffic), survival task (third specialist <5%), MiniHack (95% MLP, 0% Transformer).",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Add routing_floor and balance_coef (load balancing loss)",
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "routing_floor_zero": "No floor constraint allows routing to collapse to single expert",
        "balance_coef_zero": "No load balancing loss means no gradient to distribute traffic",
    },
}


def run_example():
    model = EnvironmentModel(
        name="Multi-specialist model (any task)",
        max_steps=500,
    )
    model.add_reward_source(
        RewardSource(
            name="task reward",
            reward_type=RewardType.TERMINAL,
            value=1.0,
            discovery_probability=0.5,
        )
    )

    # Config WITHOUT floor — will collapse
    config_bad = TrainingConfig(
        num_specialists=3,
        routing_floor=0.0,
        balance_coef=0.0,
    )

    # Config WITH floor — safe
    config_good = TrainingConfig(
        num_specialists=3,
        routing_floor=0.10,
        balance_coef=0.01,
    )

    engine = TrainingAnalysisEngine().add_all_rules()

    print("=== WITHOUT floor constraint ===")
    engine.print_report(model, config_bad)

    print("=== WITH floor constraint ===")
    engine.print_report(model, config_good)

    print("Three documented cases of this failure:")
    print()
    print("  1. Shazeer et al. 2017 (MoE paper)")
    print("     Setting: Sparse gating, no load balancing")
    print("     Outcome: Most experts receive zero traffic")
    print("     Fix: Introduced load balancing loss (now standard)")
    print()
    print("  2. Multi-specialist survival task")
    print("     Setting: 3 specialists, softmax routing, no floor")
    print("     Outcome: Third specialist <5% weight consistently")
    print("     Fix: Added balance_coef=0.01")
    print()
    print("  3. MiniHack multi-specialist")
    print("     Setting: 3 specialists, gate MLP, no floor")
    print("     Outcome: 95% MLP, 5% CNN, 0% Transformer")
    print("     Fix: Added routing_floor=0.10")
    print()
    print("All three are the SAME failure mode. The framework")
    print("catches it from the config alone: num_specialists > 1")
    print("and routing_floor == 0 → CRITICAL warning.")


if __name__ == "__main__":
    run_example()
