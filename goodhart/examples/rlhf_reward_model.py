"""Example: RLHF reward model gaming (LIMITATION).

In RLHF (Reinforcement Learning from Human Feedback), the reward
signal comes from a learned reward model, not a hand-designed function.
Agents learn to exploit gaps in the reward model — producing outputs
that score high but are low quality (sycophancy, verbosity hacking,
format gaming).

LIMITATION: goodhart is designed for hand-specified reward functions
with known components. A learned reward model is a black box — we
can't decompose it into RewardSources with known dynamics. This is
ironic given our tool is named after Goodhart's Law, which is
precisely what RLHF reward model gaming exemplifies.

Source: Casper et al. 2023 ("Open Problems and Fundamental
Limitations of RLHF"), Gao et al. 2023 ("Scaling Laws for Reward
Model Overoptimization")
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("RLHF Reward Model Gaming (TOOL LIMITATION)")
    print("=" * 70)
    print()
    print("Source: Casper et al. 2023, Gao et al. 2023")
    print()
    print("In RLHF, the 'reward function' is a neural network trained")
    print("on human preferences. Agents learn to exploit its gaps:")
    print("  - Sycophancy: agree with the user regardless of truth")
    print("  - Verbosity: longer responses score higher")
    print("  - Format gaming: bullet points and headers score well")
    print()
    print("Best approximation in goodhart:")
    print()

    # Approximate RLHF as: learned reward (black box) + KL penalty
    model = EnvironmentModel(
        name="RLHF (approximate)",
        max_steps=1,  # single generation
        gamma=1.0,
        n_states=1000000,
        n_actions=50000,  # vocabulary size
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="reward_model_score",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        value_range=(0.0, 5.0),
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="kl_penalty",
        reward_type=RewardType.TERMINAL,
        value=-0.1,
        state_dependent=True,
        requires_action=True,
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=1.4e-5,
        entropy_coeff=0.0,
        num_epochs=4,
        clip_epsilon=0.2,
        num_envs=64,
        n_actors=64,
        total_steps=1_000_000,
    )
    result = engine.print_report(model, config)

    print()
    print("=" * 70)
    print("LIMITATION: This model captures almost nothing useful.")
    print()
    print("The core problem with RLHF is that the reward model itself")
    print("is a proxy for human preferences (Goodhart's Law!). The")
    print("proxy looks correct until you optimize against it hard enough.")
    print()
    print("What goodhart CAN'T express:")
    print("  - Reward model error distribution (where it's wrong)")
    print("  - Overoptimization scaling (more KL → worse alignment)")
    print("  - Distributional shift (RM trained on different policy)")
    print("  - Preference inconsistency in the training data")
    print()
    print("What would be needed:")
    print("  - Reward model uncertainty estimation")
    print("  - Scaling law analysis (Gao et al. 2023)")
    print("  - Probe-based reward model auditing")
    print()
    print("Irony: our tool is named after Goodhart's Law, but the")
    print("purest embodiment of it (RLHF overoptimization) is beyond")
    print("our expressiveness. We catch the structural traps, not the")
    print("measurement traps.")


if __name__ == "__main__":
    run_example()
