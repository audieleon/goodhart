"""Krakovna entry: Roomba -- drives backward to avoid bumper penalty.

Source: Custard Smingleigh (anecdotal, widely cited)
Failure: Roomba drives backward to avoid triggering the front bumper
  collision sensor.
Mechanism: Speed reward with front-bumper penalty. Driving backward
  achieves the same speed reward but avoids the bumper penalty
  entirely, since the bumper sensor is only on the front.
  The asymmetric penalty structure makes backward motion strictly
  dominant.
Domain: Robotics / navigation
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Roomba backward driving (Custard Smingleigh)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Roomba (Custard Smingleigh)",
        max_steps=1000,
        gamma=0.99,
        n_actions=4,
        action_type="continuous",
    )

    # Speed reward -- earned regardless of direction
    model.add_reward_source(RewardSource(
        name="speed_reward",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=True,
        scales_with="velocity",
        state_dependent=True,
        intentional=True,
    ))

    # Front bumper penalty -- the asymmetry
    # Only triggers when driving forward into obstacles.
    # Driving backward avoids this entirely.
    model.add_reward_source(RewardSource(
        name="front_bumper_penalty",
        reward_type=RewardType.ON_EVENT,
        value=-5.0,
        max_occurrences=0,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
