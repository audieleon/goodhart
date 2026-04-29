"""Krakovna entry: Goal classifiers — robot arm exploits learned goal.

Source: Singh 2019
Failure: Robot arm exploits a learned goal classifier by moving in a
  peculiar way that triggers erroneous high-confidence goal detection.
Mechanism: Distance-based shaping toward a learned goal signal. The
  classifier provides the terminal "success" signal, so the agent
  optimizes whatever triggers the classifier rather than the true goal.
Domain: Robotics / manipulation

Structural encoding: We encode the distance shaping and a terminal
  goal reward. The structural issue is that the terminal signal comes
  from a learned classifier, not ground truth. A static analysis tool
  can flag the shaping-without-robust-terminal pattern, but cannot
  detect that the terminal signal itself is exploitable.

Advisory: learned_reward — the terminal goal signal is a learned
  classifier, not a ground-truth check. The exploit is the classifier
  being fooled, which is outside the scope of reward structure analysis.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Goal classifiers (Singh 2019)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Goal Classifiers (Singh 2019)",
        max_steps=500,
        gamma=0.99,
        n_actions=7,
        action_type="continuous",
    )

    # Distance shaping toward the object target
    model.add_reward_source(RewardSource(
        name="distance_to_target",
        reward_type=RewardType.SHAPING,
        value=0.1,
        scales_with="distance",
        state_dependent=True,
        requires_action=True,
    ))

    # Learned goal classifier provides the terminal signal.
    # Structurally this looks like a normal terminal reward, but
    # the real problem is the classifier can be fooled.
    model.add_reward_source(RewardSource(
        name="learned_goal_signal",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        intentional=True,
        discovery_probability=0.3,
    ))

    engine.print_report(model)

    print()
    print("Advisory: learned_reward — the terminal goal is a learned")
    print("classifier, not ground truth. The agent exploits the classifier")
    print("by moving in ways that trigger false positives. This failure")
    print("mode is outside the scope of structural reward analysis.")


if __name__ == "__main__":
    run_example()
