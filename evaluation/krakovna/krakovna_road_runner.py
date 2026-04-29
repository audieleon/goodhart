"""Krakovna entry: Road Runner -- agent kills itself at end of level 1.

Source: Saunders et al 2017
Failure: Agent commits suicide at the end of level 1 to avoid losing
  points in the harder level 2.
Mechanism: Score reward without level-progress incentive. Level 2 is
  harder and the agent risks losing score. Dying after maximizing
  level 1 score preserves the high score. The agent learns that
  the expected value of continuing is negative.
Domain: Atari

Also in goodhart/examples/road_runner_replay.py
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Road Runner (Saunders et al 2017)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Road Runner (Saunders et al 2017)",
        max_steps=4500,
        gamma=0.99,
        n_actions=18,
        action_type="discrete",
    )

    # Level 1 score (collectible points)
    model.add_reward_source(RewardSource(
        name="level_1_score",
        reward_type=RewardType.ON_EVENT,
        value=5.0,
        max_occurrences=20,
        intentional=True,
    ))

    # Level 2 score (harder, risky)
    model.add_reward_source(RewardSource(
        name="level_2_score",
        reward_type=RewardType.ON_EVENT,
        value=5.0,
        max_occurrences=20,
        discovery_probability=0.1,
        prerequisite="level_1_score",
    ))

    # Death at level transition -- the exploit
    # Agent intentionally dies to avoid level 2's harder content.
    # Death resets via ON_DEATH, and the level 1 score is preserved
    # in the episode return.
    model.add_reward_source(RewardSource(
        name="intentional_death",
        reward_type=RewardType.ON_EVENT,
        value=0.0,
        max_occurrences=1,
        respawn=RespawnBehavior.ON_DEATH,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
