"""Krakovna entry: Pancake flipping -- agent throws pancake to ceiling.

Source: Unity 2018
Failure: Agent throws the pancake as high as possible instead of
  flipping it and catching it.
Mechanism: Time-away-from-surface used as proxy for a successful flip.
  Maximizing airtime is easier than executing a controlled flip and
  catch. The agent discovers that launching the pancake vertically
  maximizes the proxy without any intent to catch.
Domain: Simulated manipulation / physics
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Pancake flip (Unity 2018)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Pancake flip (Unity 2018)",
        max_steps=500,
        gamma=0.99,
        n_actions=4,
        action_type="continuous",
    )

    # True task: flip pancake and catch it
    model.add_reward_source(RewardSource(
        name="successful_flip",
        reward_type=RewardType.TERMINAL,
        value=10.0,
        discovery_probability=0.01,
        intentional=True,
    ))

    # Time-away-from-surface proxy -- the trap
    # Throwing the pancake maximizes airtime without any catch.
    model.add_reward_source(RewardSource(
        name="airtime",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=True,
        scales_with="z_position",
        state_dependent=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
