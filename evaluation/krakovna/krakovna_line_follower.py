"""Krakovna entry: Line following robot -- goes backward along the line.

Source: Vamplew 2004
Failure: Robot follows the line backward instead of forward.
Mechanism: On-track reward without directional constraint. The reward
  function gives a bonus for staying on the line but does not encode
  which direction to travel. Going backward is equally rewarded,
  and may be preferred if the forward path is harder.
Domain: Robotics / navigation
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Line following robot (Vamplew 2004)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Line follower (Vamplew 2004)",
        max_steps=1000,
        gamma=0.99,
        n_actions=3,
        action_type="discrete",
        death_probability=0.0,
    )

    # Intended: reach end of track
    model.add_reward_source(RewardSource(
        name="reach_end",
        reward_type=RewardType.TERMINAL,
        value=10.0,
        discovery_probability=0.02,
        intentional=True,
    ))

    # On-track bonus -- the trap
    # Rewards staying on the line regardless of direction.
    # Going backward collects the same per-step reward.
    model.add_reward_source(RewardSource(
        name="on_track_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.1,
        requires_action=True,
        can_loop=True,
        loop_period=2,
    ))

    # Off-track penalty
    model.add_reward_source(RewardSource(
        name="off_track_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.5,
        requires_action=False,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
