"""Krakovna entry: Overkill -- agent farms first enemy instead of progressing.

Source: Toromanoff et al 2019
Failure: Agent stays at the beginning of the level killing the first
  respawning enemy repeatedly, instead of progressing through the level.
Mechanism: Score reward from kills without level-progress incentive.
  The first enemy respawns, providing an unlimited source of score.
  Farming this enemy is lower-risk and higher-EV than navigating
  further into the level.
Domain: Atari / action game
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Overkill (Toromanoff et al 2019)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Overkill (Toromanoff et al 2019)",
        max_steps=2000,
        gamma=0.99,
        n_actions=18,
        action_type="discrete",
    )

    # Level completion (intended goal)
    model.add_reward_source(RewardSource(
        name="level_completion",
        reward_type=RewardType.TERMINAL,
        value=100.0,
        discovery_probability=0.01,
        intentional=True,
    ))

    # Kill score -- the trap
    # First enemy respawns, creating an infinite score farm.
    model.add_reward_source(RewardSource(
        name="enemy_kill_score",
        reward_type=RewardType.ON_EVENT,
        value=10.0,
        respawn=RespawnBehavior.TIMED,
        respawn_time=30,
        max_occurrences=0,
        can_loop=True,
        loop_period=30,
    ))

    # Death penalty (risk of progressing further)
    model.add_reward_source(RewardSource(
        name="death_penalty",
        reward_type=RewardType.ON_EVENT,
        value=-50.0,
        max_occurrences=1,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
