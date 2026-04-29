"""Krakovna entry: CoastRunners boat race -- agent loops for points.

Source: Amodei & Clark 2016
Failure: Agent drives in circles collecting sparse waypoint bonuses
  instead of finishing the race.
Mechanism: Waypoint rewards respawn on a loop. The per-lap waypoint
  total exceeds the race-completion bonus, so the optimal policy under
  the proxy is to never finish.
Domain: Atari / racing

Also in goodhart/examples/coast_runners.py
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: CoastRunners boat race (Amodei & Clark 2016)")
    print("=" * 70)

    model = EnvironmentModel(
        name="CoastRunners (Amodei & Clark 2016)",
        max_steps=2000,
        gamma=0.99,
        n_actions=9,
        action_type="discrete",
    )

    # Finishing the race
    model.add_reward_source(RewardSource(
        name="finish_race",
        reward_type=RewardType.TERMINAL,
        value=10.0,
        discovery_probability=0.1,
        intentional=True,
    ))

    # Waypoint bonuses -- the trap
    # Respawn each lap; looping is more profitable than finishing.
    model.add_reward_source(RewardSource(
        name="waypoint_bonus",
        reward_type=RewardType.ON_EVENT,
        value=3.0,
        respawn=RespawnBehavior.TIMED,
        respawn_time=50,
        max_occurrences=0,
        can_loop=True,
        loop_period=60,
    ))

    # Occasionally catching fire (collision penalty)
    model.add_reward_source(RewardSource(
        name="collision_penalty",
        reward_type=RewardType.ON_EVENT,
        value=-1.0,
        max_occurrences=0,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
