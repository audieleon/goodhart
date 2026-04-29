"""Krakovna entry: Montezuma's Revenge key glitch.

Source: Salimans & Chen 2018
Failure: Agent exploits an Atari emulator flaw to make the key
  re-appear after collection, harvesting score repeatedly.
Mechanism: Sparse score reward on item collection, plus death
  penalty. The reward structure is standard for Atari — on_event
  score per collection. The exploit is an emulator bug that causes
  items to respawn when they should not.
Domain: Atari / discrete control

Structural encoding: We encode the score-per-collection as an
  on_event reward with respawn=NONE (the intended behavior). The
  actual exploit is that the emulator erroneously allows the key
  to respawn. If we encoded respawn=INFINITE, the tool would flag
  it — but that respawn behavior is a bug, not a design choice.

Advisory: The exploit is an emulator bug, not a reward structure
  problem. The reward correctly gives points for key collection;
  the emulator incorrectly allows re-collection.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


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
