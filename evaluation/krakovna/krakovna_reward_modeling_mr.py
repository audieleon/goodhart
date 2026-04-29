"""Krakovna entry: Reward modeling — Montezuma's Revenge.

Source: Ibarz et al 2018
Failure: Agent moves toward the key without ever grabbing it. The
  learned reward model rewards the approach behavior more than the
  actual collection, so the agent stops short.
Mechanism: Learned reward model provides per-step signal. The model
  was trained from human demonstrations and overvalues the approach
  trajectory relative to the pickup event. The agent maximizes the
  learned model's output, which plateaus before the key is collected.
Domain: Atari / reward modeling

Structural encoding: Minimal — same approach as Hero. The reward
  comes from a learned model. We encode a generic per-step task
  reward plus what the intended terminal event looks like. The
  structural analysis can note the absence of a clear terminal
  signal, but cannot diagnose the learned model's early-reward bias.

Advisory: learned_reward — the failure is the learned reward model
  rewarding approach more than collection. This is a bias in the
  learned model, not a structural reward specification problem.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Reward modeling — Montezuma's Revenge (Ibarz et al 2018)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Reward Modeling MR (Ibarz et al 2018)",
        max_steps=4500,
        gamma=0.99,
        n_actions=18,
        action_type="discrete",
        death_probability=0.05,
    )

    # Learned reward model — per-step signal that overvalues approach.
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
    print("Advisory: learned_reward — the reward model overvalues the")
    print("approach trajectory to the key and undervalues the actual")
    print("collection event. The agent converges to approach-without-")
    print("grabbing because that is the reward-maximizing behavior in")
    print("the learned model's space. Structural analysis cannot detect")
    print("biases internal to a learned reward model.")


if __name__ == "__main__":
    run_example()
