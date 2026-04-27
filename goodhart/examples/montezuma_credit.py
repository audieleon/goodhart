"""Example: Montezuma's Revenge — credit assignment (LIMITATION).

Montezuma's Revenge has a perfectly fine reward structure: +100 for
each key/item, +300 for completing rooms. The problem is extreme
temporal credit assignment — the agent must execute ~100 correct
actions in sequence before seeing any reward.

LIMITATION: goodhart's exploration_threshold rule catches the low
discovery probability, but can't distinguish "sparse but learnable
with better exploration" from "fundamentally too sparse." The fix
(curiosity/RND) is algorithmic, not structural.

Source: Bellemare et al. 2013 (ALE), Burda et al. 2019 (RND, ICLR)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


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
    result = engine.print_report(model, config)

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
