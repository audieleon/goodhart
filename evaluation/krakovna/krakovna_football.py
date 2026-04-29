"""Krakovna entry: Football -- agent kicks ball out of bounds.

Source: Kurach et al 2019 (Google Research Football)
Failure: Agent learns to kick the ball out of bounds to trigger a
  throw-in, which gives it possession closer to the goal.
Mechanism: Sparse goal reward incentivizes any strategy that leads to
  scoring. Kicking out of bounds yields a throw-in (possession reset)
  that is easier to exploit than dribbling through defenders. The
  agent discovers that throw-in positioning is a higher-EV strategy
  than direct play.
Domain: Simulated sports / multi-agent
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Football out-of-bounds exploit (Kurach et al 2019)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Football (Kurach et al 2019)",
        max_steps=3000,
        gamma=0.997,
        n_actions=19,
        action_type="discrete",
    )

    # Scoring a goal -- the intended reward
    model.add_reward_source(RewardSource(
        name="goal_scored",
        reward_type=RewardType.ON_EVENT,
        value=1.0,
        max_occurrences=0,
        discovery_probability=0.05,
        intentional=True,
    ))

    # Conceding a goal
    model.add_reward_source(RewardSource(
        name="goal_conceded",
        reward_type=RewardType.ON_EVENT,
        value=-1.0,
        max_occurrences=0,
    ))

    # Throw-in exploit -- the trap
    # Kicking out of bounds triggers a throw-in that grants favorable
    # positioning. This is effectively a repeatable event with
    # implicit positive value from the resulting field position.
    model.add_reward_source(RewardSource(
        name="throw_in_positioning",
        reward_type=RewardType.ON_EVENT,
        value=0.0,
        max_occurrences=0,
        can_loop=True,
        loop_period=30,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
