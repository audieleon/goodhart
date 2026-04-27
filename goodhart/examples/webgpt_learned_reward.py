"""Example: WebGPT / RLHF — learned reward model (advisory demo).

WebGPT uses a reward model trained on human comparisons to evaluate
web search + answer quality. The agent learns to game the RM:
preferring authoritative-sounding but incorrect sources, padding
answers with hedging language, citing more sources than needed.

This is Goodhart's Law in its purest form: the RM measures answer
quality, the agent targets the RM, the RM stops measuring quality.

We model this minimally to show how the advisory_learned_reward rule
could fire (though the pattern requires very high action space).

Source: Nakano et al. 2022 (WebGPT), Gao et al. 2023 (RM overoptimization)
LIMITATION: goodhart fundamentally cannot analyze learned rewards.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("WebGPT / RLHF — learned reward model (TOOL LIMITATION)")
    print("=" * 70)
    print()
    print("Source: Nakano et al. 2022, Gao et al. 2023")
    print()
    print("The reward is a neural network, not a formula.")
    print("The agent games the network's blind spots:")
    print("  - Sounds authoritative → high RM score (even if wrong)")
    print("  - More citations → high RM score (even if irrelevant)")
    print("  - Hedging language → high RM score (avoids penalties)")
    print()
    print("What goodhart says about our best approximation:")
    print()

    model = EnvironmentModel(
        name="WebGPT (RLHF approximate)",
        max_steps=1,
        gamma=1.0,
        n_states=1000000,
        n_actions=50000,  # vocabulary size
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="rm_score",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="kl_penalty",
        reward_type=RewardType.TERMINAL,
        value=-0.1,
        requires_action=True,
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=1.4e-5,
        entropy_coeff=0.0,
        num_epochs=4,
        num_envs=64,
        n_actors=64,
        total_steps=1_000_000,
    )
    result = engine.print_report(model, config)

    print()
    print("=" * 70)
    print("This is the most important thing goodhart CANNOT do.")
    print()
    print("Gao et al. 2023 showed that RM score improves with KL")
    print("up to a point, then actual quality DECREASES while RM")
    print("score keeps rising. The proxy diverges from the target.")
    print()
    print("No static analysis of reward structure can catch this")
    print("because the structure IS simple (RM score - KL penalty).")
    print("The problem is inside the RM — which is a neural network.")
    print()
    print("What would help: RM uncertainty estimation, red-teaming,")
    print("ensemble disagreement, scaling law extrapolation.")


if __name__ == "__main__":
    run_example()
