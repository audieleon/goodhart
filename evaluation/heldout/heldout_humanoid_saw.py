"""Example: Humanoid SaW — held-out evaluation.

Held-out paper not used during tool development. Tests whether
the tool passes structurally sound locomotion rewards while
firing physics exploit advisories for continuous control.

Source: van Marum et al. 2024, "Revisiting Reward Design and
  Evaluation for Robust Humanoid Standing and Walking" (IROS 2024)

Expected result: PASS structurally. Alive bonus (0.5) is small
  relative to velocity tracking (1.0) and marked intentional
  (survival IS the goal in locomotion). Physics exploit advisory
  should fire since this is locomotion with continuous control.
  The paper's actual issues (hopping gaits, exploitation of
  physics simulator) are the kind of thing blind-spot advisories
  warn about but cannot detect structurally.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Humanoid SaW — held-out evaluation")
    print("=" * 70)
    print()
    print("Source: van Marum et al. 2024, IROS 2024")
    print("Humanoid standing and walking with velocity tracking.")
    print("Structurally sound; physics advisory should fire.")
    print()

    model = EnvironmentModel(
        name="Humanoid SaW (Held-Out)",
        max_steps=1000,
        gamma=0.99,
        n_actions=21,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="velocity_tracking",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="alive_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        requires_action=False,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="smoothness_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.1,
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="energy_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.05,
        requires_action=True,
    ))

    engine.print_report(model)

    print()
    print("The tool should pass structurally: alive_bonus (0.5) is")
    print("moderate relative to velocity_tracking (1.0), and the")
    print("penalties (-0.1 smoothness, -0.05 energy) are well-scaled.")
    print("The physics exploit advisory should fire for continuous")
    print("control locomotion, noting that sim-to-real gaps and")
    print("physics exploits (hopping gaits, ground glitching) are")
    print("possible but cannot be detected from structure alone.")


if __name__ == "__main__":
    run_example()
