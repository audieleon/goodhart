"""Krakovna entry: Sonic wall clipping for forward progress.

Source: Hesse et al 2018
Failure: Agent clips through walls to move rightward faster, earning
  higher forward-progress score than intended platforming gameplay.
Mechanism: Per-step forward progress reward plus level completion
  bonus. The reward correctly incentivizes rightward movement, but
  the physics engine allows wall clipping. The agent discovers that
  clipping through walls is a faster way to accumulate forward
  progress than platforming.
Domain: Atari / platformer

Structural encoding: We encode per-step forward progress (position
  delta) and a terminal level-completion bonus. The reward structure
  is reasonable for a platformer. The exploit is a physics/collision
  bug — the game engine allows the agent to pass through walls.

Advisory: The wall clipping is a physics/collision bug in the game
  engine, not a reward structure problem. The forward progress reward
  is a standard and appropriate signal for platformers. Structural
  analysis would not flag this encoding.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Sonic wall clipping (Hesse et al 2018)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Sonic (Hesse et al 2018)",
        max_steps=4500,
        gamma=0.99,
        n_actions=8,
        action_type="discrete",
        death_probability=0.02,
    )

    # Per-step forward progress — rightward position delta.
    model.add_reward_source(RewardSource(
        name="forward_progress",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        value_type="proportional",
        value_params={"scale": 1.0},
        scales_with="position",
        state_dependent=True,
        requires_action=True,
        intentional=True,
        max_occurrences=0,
    ))

    # Level completion bonus
    model.add_reward_source(RewardSource(
        name="level_completion",
        reward_type=RewardType.TERMINAL,
        value=100.0,
        intentional=True,
        discovery_probability=0.1,
    ))

    # Death penalty
    model.add_reward_source(RewardSource(
        name="death_penalty",
        reward_type=RewardType.TERMINAL,
        value=-50.0,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("Advisory: physics_exploit — the wall clipping is a collision")
    print("detection bug in the Sonic engine. The reward structure")
    print("(forward progress + level completion) is appropriate for a")
    print("platformer. The agent simply found a faster way to move right")
    print("by exploiting broken collision physics.")


if __name__ == "__main__":
    run_example()
