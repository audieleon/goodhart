"""Krakovna entry: Hide-and-seek physics exploits.

Source: Baker et al 2019
Failure: Multi-agent hide-and-seek where agents discover physics
  exploits (box surfing, ramp exploitation, endless running) to win.
Mechanism: Binary win/lose terminal reward plus per-step survival.
  Hiders get +1 for surviving, seekers get +1 for finding. The reward
  structure itself is clean — the exploits come from physics engine
  bugs (box surfing, ramp clipping) that the agents discover through
  emergent multi-agent competition.
Domain: Multi-agent / navigation

Structural encoding: We encode the binary terminal outcome and
  per-step survival bonus. The reward structure is well-designed for
  the intended game. The actual exploits are physics/dynamics bugs,
  not reward specification problems.

Advisory: physics_exploit — the box surfing and ramp exploits are
  MuJoCo physics engine bugs. The reward structure correctly
  incentivizes hide/seek behavior; it is the environment dynamics
  that are exploitable.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Hide-and-seek (Baker et al 2019)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Hide-and-Seek (Baker et al 2019)",
        max_steps=240,
        gamma=0.998,
        n_actions=6,
        action_type="continuous",
        death_probability=0.0,
    )

    # Binary win/lose at episode end.
    # Hiders win if not found; seekers win if they find hiders.
    model.add_reward_source(RewardSource(
        name="win_reward",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        intentional=True,
        discovery_probability=0.5,
    ))

    model.add_reward_source(RewardSource(
        name="lose_penalty",
        reward_type=RewardType.TERMINAL,
        value=-1.0,
        intentional=True,
        discovery_probability=0.5,
    ))

    # Per-step survival signal for hiders (rewarded each step hidden)
    model.add_reward_source(RewardSource(
        name="survival_per_step",
        reward_type=RewardType.PER_STEP,
        value=0.1,
        requires_action=True,
        intentional=True,
        max_occurrences=0,
    ))

    engine.print_report(model)

    print()
    print("Advisory: physics_exploit — the reward structure is sound for")
    print("the intended game. Box surfing and ramp exploitation are MuJoCo")
    print("dynamics bugs, not reward specification problems. The structural")
    print("encoding shows a clean competitive reward; what broke was the")
    print("physics engine, not the incentive design.")


if __name__ == "__main__":
    run_example()
