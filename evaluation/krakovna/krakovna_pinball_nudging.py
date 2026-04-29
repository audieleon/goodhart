"""Krakovna entry: Pinball table nudging.

Source: Lapuschkin et al 2019
Failure: Agent learns to nudge the pinball table so the ball
  infinitely triggers a high-scoring switch, rather than playing
  with the flippers.
Mechanism: Per-event score from flipper hits and target switches.
  Nudging the table is a legal action that repositions the ball
  to repeatedly trigger the same high-value switch in a loop.
Domain: Atari / discrete control

Structural encoding: We encode the per-event score from hitting
  targets. The key structural feature is that the high-scoring
  switch can be triggered repeatedly (effectively respawn=INFINITE
  via nudging), creating a farmable loop. The nudge action is a
  legitimate part of the action space.

Advisory: The nudging mechanic is part of real pinball. The exploit
  is structural — the agent found a repeatable loop to harvest a
  high-value event. This is encodable.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Pinball nudging (Lapuschkin et al 2019)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Pinball Nudging (Lapuschkin et al 2019)",
        max_steps=5000,
        gamma=0.99,
        n_actions=4,
        action_type="discrete",
        death_probability=0.01,
    )

    # High-scoring switch — can be triggered repeatedly via nudging.
    # This is the source the agent farms.
    model.add_reward_source(RewardSource(
        name="high_value_switch",
        reward_type=RewardType.ON_EVENT,
        value=50.0,
        respawn=RespawnBehavior.INFINITE,
        max_occurrences=0,
        requires_action=True,
        can_loop=True,
        loop_period=5,
    ))

    # Normal flipper-based scoring (bumpers, ramps, etc.)
    model.add_reward_source(RewardSource(
        name="flipper_score",
        reward_type=RewardType.ON_EVENT,
        value=10.0,
        respawn=RespawnBehavior.INFINITE,
        max_occurrences=0,
        requires_action=True,
    ))

    # Ball loss (drain)
    model.add_reward_source(RewardSource(
        name="ball_loss",
        reward_type=RewardType.TERMINAL,
        value=-50.0,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("The high_value_switch is farmable via nudging — the agent found")
    print("a repeatable loop that scores more than playing with flippers.")
    print("This is a structural exploit: the high-value event has no")
    print("cooldown or diminishing returns, so looping it dominates.")


if __name__ == "__main__":
    run_example()
