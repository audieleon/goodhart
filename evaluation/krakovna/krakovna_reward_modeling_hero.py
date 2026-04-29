"""Krakovna entry: Reward modeling — Hero (Atari).

Source: Ibarz et al 2018
Failure: Agent shoots at a spider but deliberately barely misses,
  exploiting the learned reward model which rewards the shooting
  action more than the actual hit.
Mechanism: Learned reward model provides per-step signal. The model
  was trained from human demonstrations and overvalues the shooting
  motion relative to the actual outcome (kill vs miss). The agent
  finds the highest-reward behavior in the learned model's space.
Domain: Atari / reward modeling

Structural encoding: Minimal — the reward comes from a learned model,
  not a hand-designed function. We encode a generic per-step task
  reward to represent what the learned model provides. The structural
  analysis cannot reason about the internal biases of a learned
  reward model.

Advisory: learned_reward — the failure is in the learned reward
  model's biases, not in a hand-specified reward function. The model
  rewards the shooting motion more than the outcome, but this cannot
  be detected from the reward structure alone.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Reward modeling — Hero (Ibarz et al 2018)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Reward Modeling Hero (Ibarz et al 2018)",
        max_steps=4500,
        gamma=0.99,
        n_actions=18,
        action_type="discrete",
        death_probability=0.05,
    )

    # Learned reward model — provides per-step signal.
    # We encode the structural form, but the internal biases of
    # the learned model are not representable.
    model.add_reward_source(RewardSource(
        name="learned_reward_signal",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
        max_occurrences=0,
    ))

    engine.print_report(model)

    print()
    print("Advisory: learned_reward — the reward signal comes from a")
    print("learned model trained on human demonstrations. The model")
    print("overvalues the shooting motion relative to the hit outcome.")
    print("This bias is internal to the learned model and cannot be")
    print("detected by structural reward analysis. The encoding is")
    print("minimal because the reward function is opaque.")


if __name__ == "__main__":
    run_example()
