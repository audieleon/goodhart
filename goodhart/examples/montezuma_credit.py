"""Example: Montezuma's Revenge — credit assignment (LIMITATION).

Reward structure is fine (+100/key, +300/room) but ~100 correct actions
are needed before any reward. The problem is credit assignment, not design.

Source: Bellemare et al. 2013 (ALE), Burda et al. 2019 (RND, ICLR)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "montezuma_credit",
    "source_paper": "Bellemare et al. 2013 (ALE); Burda et al. 2019 (RND, ICLR)",
    "paper_url": "https://arxiv.org/abs/1810.12894",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2013,
    "domain": "game_ai",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Reward structure is fine (+100/key, +300/room) but ~100 correct actions needed before any reward makes credit assignment the bottleneck.",
    "documented_failure": "Extreme temporal credit assignment: agent must execute ~100 correct actions in sequence before seeing any reward; goodhart catches low discovery probability but cannot distinguish 'sparse but learnable' from 'hierarchically sparse'",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Curiosity/RND for exploration (algorithmic fix, not structural)",
    "compute_cost_class": "high",
    "is_negative_example": True,
    "encoding_rationale": {
        "framework_limitation": "Tool sees symptom (low discovery_probability) but cannot diagnose root cause (hierarchical structure)",
        "credit_assignment": "Problem is temporal depth, not reward design",
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Montezuma's Revenge — credit assignment (TOOL LIMITATION)")
    print("=" * 70)
    print()
    print("Source: Bellemare et al. 2013, Burda et al. 2019 (RND)")
    print()
    print("The reward structure is fine: +100 per key, +300 per room.")
    print("The problem: ~100 precise actions needed before first reward.")
    print()
    print("What goodhart says:")
    print()

    model = EnvironmentModel(
        name="Montezuma's Revenge",
        max_steps=18000,  # 108K frames / 6 frame skip
        gamma=0.99,
        n_states=1000000,
        n_actions=18,
        death_probability=0.1,
    )
    model.add_reward_source(RewardSource(
        name="item_pickup",
        reward_type=RewardType.ON_EVENT,
        value=100.0,
        max_occurrences=10,
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.001,  # extremely rare by random play
    ))
    model.add_reward_source(RewardSource(
        name="room_complete",
        reward_type=RewardType.ON_EVENT,
        value=300.0,
        max_occurrences=24,
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.0001,
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=2.5e-4,
        entropy_coeff=0.01,
        num_epochs=4,
        clip_epsilon=0.1,
        num_envs=128,
        n_actors=128,
        total_steps=10_000_000_000,
        rollout_length=128,
    )
    engine.print_report(model, config)

    print()
    print("=" * 70)
    print("goodhart correctly flags the exploration problem, but its")
    print("recommendation ('add intrinsic motivation') is generic.")
    print()
    print("What goodhart CAN'T capture:")
    print("  - Temporal depth of credit assignment (~100 steps)")
    print("  - Hierarchical task structure (rooms → keys → doors)")
    print("  - The difference between 'sparse' and 'hierarchically")
    print("    sparse' (Montezuma needs a sequence of discoveries,")
    print("    not just a single rare event)")
    print()
    print("What would be needed:")
    print("  - Prerequisite chain depth analysis (we have prerequisites")
    print("    but only for explicit staged rewards, not implicit ones)")
    print("  - Temporal credit assignment difficulty estimation")
    print("  - Distinction between 'add shaping' vs 'add curiosity'")
    print()
    print("The tool sees the symptom (low discovery_probability) but")
    print("can't diagnose the root cause (hierarchical structure).")


if __name__ == "__main__":
    run_example()
