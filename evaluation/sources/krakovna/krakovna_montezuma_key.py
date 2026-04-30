"""Krakovna entry: Montezuma's Revenge key glitch.

Emulator bug causes the key to respawn after collection; agent
harvests score repeatedly from the same item.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "krakovna_montezuma_key",
    "source_paper": (
        'Salimans & Chen 2018, "Learning Montezuma\'s Revenge'
        ' from a Single Demonstration"'
    ),
    "paper_url": "https://arxiv.org/abs/1812.03381",
    "year": 2018,
    "domain": "game_ai",
    "encoding_basis": "primary_source",
    "brief_summary": (
        "Standard Atari score reward. The exploit is an"
        " emulator bug causing the key to respawn, not a"
        " reward specification problem."
    ),
    "documented_failure": (
        "Agent exploits emulator flaw to make the key"
        " re-appear after collection."
    ),
    "failure_mechanism": "environment_bug",
    "detection_type": "dynamic",
    "is_negative_example": True,
    "compute_cost_class": "low",
    "paper_url": "https://arxiv.org/abs/1812.03381",
    "reward_location": "Salimans & Chen 2018: ALE game score. Key respawns via emulator bug.",
    "verification_date": "2026-04-30",
    "discovery_stage": "during_training",
    "source_code_url": None,
    "fix_known": None,
    "encoding_rationale": {
        "basis": "Encoded from Krakovna specification gaming catalog with paper verification where available.",
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Montezuma's Revenge key (Salimans & Chen 2018)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Montezuma's Revenge Key (Salimans & Chen 2018)",
        max_steps=4500,
        gamma=0.99,
        n_actions=18,
        action_type="discrete",
        death_probability=0.05,
    )

    # Sparse score per item collection.
    # Intended: collect once, item gone. Emulator bug: key re-appears.
    model.add_reward_source(RewardSource(
        name="key_collection",
        reward_type=RewardType.ON_EVENT,
        value=100.0,
        respawn=RespawnBehavior.NONE,
        max_occurrences=1,
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.1,
        intentional=True,
    ))

    # Other score events (doors, enemies, etc.)
    model.add_reward_source(RewardSource(
        name="other_score",
        reward_type=RewardType.ON_EVENT,
        value=50.0,
        respawn=RespawnBehavior.NONE,
        max_occurrences=10,
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.05,
    ))

    # Death penalty
    model.add_reward_source(RewardSource(
        name="death_penalty",
        reward_type=RewardType.TERMINAL,
        value=-100.0,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("Advisory: The exploit is an emulator bug — the key respawns")
    print("when it should not. The reward structure (sparse score per")
    print("collection) is correct for the intended game. If the emulator")
    print("worked properly, this encoding would produce no warnings about")
    print("respawning exploits.")


if __name__ == "__main__":
    run_example()
