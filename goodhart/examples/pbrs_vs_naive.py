"""Example: PBRS vs Naive Shaping — side-by-side comparison.

PBRS (Ng 1999) gets 2 criticals; naive additive shaping gets 4. The tool
catches shaping_loop_exploit and respawning_exploit that PBRS eliminates.

Source: Eschmann et al. 2023 (RSS), "Benchmarking Potential Based
  Rewards for Learning Humanoid Locomotion"; Ng et al. 1999
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "pbrs_vs_naive",
    "source_paper": "Eschmann et al. 2023 (RSS), Benchmarking Potential Based Rewards for Learning Humanoid Locomotion; Ng et al. 1999",
    "paper_url": "https://arxiv.org/abs/2307.04217",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2023,
    "domain": "locomotion",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Side-by-side comparison: PBRS gets 2 criticals, naive shaping gets 4. PBRS eliminates loop and respawning exploits by construction.",
    "documented_failure": "Naive additive shaping (action-dependent, loopable) triggers shaping_loop_exploit, respawning_exploit, and shaping_not_potential_based; PBRS (state-only, can_loop=False) eliminates these by Ng 1999 theorem",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Use F(s,s') = gamma*Phi(s') - Phi(s) form (the only form that preserves optimal policy)",
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "comparison_example": "Shows before/after to demonstrate PBRS eliminates failure modes by construction",
        "ng_1999_theorem": "F(s,s') = gamma*Phi(s') - Phi(s) is the ONLY form preserving optimal policy",
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # --- PBRS (correct) ---
    print("=" * 70)
    print("Humanoid + Potential-Based Reward Shaping (CORRECT)")
    print("=" * 70)
    print()
    print("Source: Eschmann et al. 2023 (RSS), Ng et al. 1999")
    print("F(s,s') = gamma*Phi(s') - Phi(s), where Phi = -distance")
    print()

    model_pbrs = EnvironmentModel(
        name="Humanoid + PBRS",
        max_steps=1000,
        gamma=0.99,
        n_states=100000,
        n_actions=17,
        death_probability=0.03,
    )
    model_pbrs.add_reward_source(
        RewardSource(
            name="velocity",
            reward_type=RewardType.PER_STEP,
            value=1.25,
            state_dependent=True,
            requires_action=True,
            intentional=True,
        )
    )
    model_pbrs.add_reward_source(
        RewardSource(
            name="pbrs_distance_shaping",
            reward_type=RewardType.SHAPING,
            value=1.0,
            requires_action=False,  # Potential-based: state-only
            can_loop=False,  # Cycles net zero by Ng 1999
        )
    )
    model_pbrs.add_reward_source(
        RewardSource(
            name="ctrl_cost",
            reward_type=RewardType.PER_STEP,
            value=-0.1,
            requires_action=True,
        )
    )

    config = TrainingConfig(
        lr=3e-4,
        num_envs=1,
        n_actors=1,
        total_steps=1_000_000,
        num_epochs=10,
        rollout_length=2048,
        minibatch_size=64,
    )
    result_pbrs = engine.print_report(model_pbrs, config)

    # --- Naive shaping (exploitable) ---
    print()
    print("=" * 70)
    print("Humanoid + Naive Additive Shaping (EXPLOITABLE)")
    print("=" * 70)
    print()
    print("Same environment, but shaping is action-dependent and loopable")
    print()

    model_naive = EnvironmentModel(
        name="Humanoid + naive shaping",
        max_steps=1000,
        gamma=0.99,
        n_states=100000,
        n_actions=17,
        death_probability=0.03,
    )
    model_naive.add_reward_source(
        RewardSource(
            name="velocity",
            reward_type=RewardType.PER_STEP,
            value=1.25,
            state_dependent=True,
            requires_action=True,
            intentional=True,
        )
    )
    model_naive.add_reward_source(
        RewardSource(
            name="naive_distance_shaping",
            reward_type=RewardType.SHAPING,
            value=1.0,
            requires_action=True,  # Action-dependent: NOT potential-based
            can_loop=True,  # Cycles earn reward
            loop_period=10,
        )
    )
    model_naive.add_reward_source(
        RewardSource(
            name="ctrl_cost",
            reward_type=RewardType.PER_STEP,
            value=-0.1,
            requires_action=True,
        )
    )

    result_naive = engine.print_report(model_naive, config)

    # --- Comparison ---
    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print()
    pbrs_rules = {
        v.rule_name for v in result_pbrs.verdicts if v.severity == Severity.CRITICAL
    }
    naive_rules = {
        v.rule_name for v in result_naive.verdicts if v.severity == Severity.CRITICAL
    }
    extra = naive_rules - pbrs_rules

    print(f"PBRS criticals:  {sorted(pbrs_rules)}")
    print(f"Naive criticals: {sorted(naive_rules)}")
    print(f"Extra in naive:  {sorted(extra)}")
    print()
    print("PBRS eliminates these failure modes BY CONSTRUCTION:")
    for rule in sorted(extra):
        print(f"  - {rule}")
    print()
    print("This is the single most important result in reward design")
    print("theory: F(s,s') = gamma*Phi(s') - Phi(s) is the ONLY form")
    print("that preserves optimal policy (Ng 1999, both directions).")
    print("The tool proves this empirically — PBRS shaping doesn't")
    print("trigger the rules that naive shaping does.")


if __name__ == "__main__":
    run_example()
