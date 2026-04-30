"""Example: Google Data Center Cooling — simple reward + hard constraints (POSITIVE).

Reward is just -energy_consumption with safety as action masks, not
penalties. Achieves 9-13% savings and the tool correctly passes clean.

Source: Lazic et al. 2022, "Controlling Commercial Cooling Systems Using RL"
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "datacenter_cooling",
    "source_paper": "Lazic et al. 2022, 'Controlling Commercial Cooling Systems Using Reinforcement Learning'",
    "paper_url": None,
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2022,
    "domain": "industrial",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Simple -energy_consumption reward with safety as hard constraints. Well-designed, deployed in production with 9-13% savings.",
    "documented_failure": "N/A — well-designed reward with external safety constraints. Tool correctly gives a clean bill.",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "high",
    "is_negative_example": True,
    "encoding_rationale": {
        "simple_reward": "Single reward term (-energy) with no competing objectives",
        "hard_constraints": "Safety enforced as action masks, not soft penalties",
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Data Center Cooling — simple reward works (POSITIVE)")
    print("=" * 70)
    print()
    print("Source: Lazic et al. 2022 (Google)")
    print("Reward: -energy_consumption (that's it)")
    print("Safety: temperature/humidity/equipment constraints as action masks")
    print("Result: 9-13% energy savings, deployed in production")
    print()

    model = EnvironmentModel(
        name="Data Center Cooling",
        max_steps=8760,    # hours in a year
        gamma=0.99,
        n_states=100000,
        n_actions=20,
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="energy_cost",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))
    # No other reward terms. Safety constraints are action masks,
    # not reward penalties. This is intentional and crucial.

    config = TrainingConfig(
        lr=1e-4,
        entropy_coeff=0.01,
        num_envs=8,
        n_actors=8,
        total_steps=50_000_000,
    )
    result = engine.print_report(model, config)

    print()
    print("Clean bill. This is correct — the reward is as simple as")
    print("possible and there are no structural traps.")
    print()
    print("DESIGN PATTERN: 'Simple reward + hard constraints'")
    print()
    print("Most RL practitioners try to put everything in the reward:")
    print("  r = energy_savings - temp_violation - humidity_violation")
    print("  → Agent learns the optimal NUMBER of violations to commit")
    print()
    print("Google's approach separates them:")
    print("  r = -energy_consumption (optimize)")
    print("  constraints: T ∈ [T_min, T_max] per zone (enforce)")
    print("  → Agent cannot violate constraints at any reward level")
    print()
    print("This is why the advisory_constrained_rl rule exists: when")
    print("you model constraints as penalties, you're using the wrong")
    print("tool. Constraints should be constraints, not soft penalties.")


if __name__ == "__main__":
    run_example()
