"""Krakovna entry: Montezuma's Revenge treasure room exploit.

Treasure room items respawn infinitely; agent farms them for
unlimited points instead of exploring.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "krakovna_montezuma_room",
    "source_paper": 'Ecoffet et al 2019, "Go-Explore"',
    "paper_url": "https://arxiv.org/abs/1901.10995",
    "year": 2019,
    "domain": "game_ai",
    "encoding_basis": "primary_source",
    "brief_summary": (
        "Treasure room items respawn infinitely (emulator bug)."
        " Encoded as respawning reward to capture the effective"
        " dynamics the agent exploits."
    ),
    "documented_failure": (
        "Agent stays in treasure room farming infinitely"
        " respawning items for unlimited score."
    ),
    "failure_mechanism": "respawning_loop",
    "detection_type": "structural",
    "is_negative_example": False,
    "compute_cost_class": "low",
    "paper_url": "https://arxiv.org/abs/1901.10995",
    "reward_location": "Ecoffet et al. 2019: ALE score. Gems 1000 pts in treasure room. 51/57 runs exploited the bug.",
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
    print("Krakovna: Montezuma's Revenge room (Ecoffet et al 2019)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Montezuma's Revenge Room (Ecoffet et al 2019)",
        max_steps=4500,
        gamma=0.99,
        n_actions=18,
        action_type="discrete",
        death_probability=0.05,
    )

    # Treasure room items — respawn infinitely due to bug.
    # Encoding the effective dynamics the agent actually exploits.
    model.add_reward_source(RewardSource(
        name="treasure_items",
        reward_type=RewardType.ON_EVENT,
        value=100.0,
        respawn=RespawnBehavior.INFINITE,
        max_occurrences=0,
        requires_action=True,
        can_loop=True,
        loop_period=10,
    ))

    # Score from normal gameplay (reaching rooms, enemies, etc.)
    model.add_reward_source(RewardSource(
        name="exploration_score",
        reward_type=RewardType.ON_EVENT,
        value=50.0,
        respawn=RespawnBehavior.NONE,
        max_occurrences=20,
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.05,
        intentional=True,
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
    print("This IS a respawning exploit that structural analysis can detect.")
    print("The treasure_items source has respawn=INFINITE and can_loop=True,")
    print("so the tool should flag it as an infinite farming opportunity.")
    print("Root cause is an emulator bug, but the effective reward dynamics")
    print("are structurally a respawning-reward loop.")


if __name__ == "__main__":
    run_example()
