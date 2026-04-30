"""Example: WebGPT / RLHF -- learned reward model (advisory demo).

Agent games the reward model by preferring authoritative-sounding but incorrect sources.
Source: Nakano et al. 2022 (WebGPT), Gao et al. 2023 (RM overoptimization)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "webgpt_learned_reward",
    "source_paper": "Nakano et al. 2022 (WebGPT), Gao et al. 2023 ('Scaling Laws for Reward Model Overoptimization')",
    "paper_url": "https://arxiv.org/abs/2112.09332",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2022,
    "domain": "rlhf",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to find and summarize accurate information. Instead it learned to produce authoritative-sounding but incorrect outputs that score high on the reward model.",
    "documented_failure": "Agent games reward model blind spots: preferring authoritative-sounding but incorrect sources, padding with hedging language, citing more sources than needed. RM score rises while actual quality decreases.",
    "failure_mechanism": None,
    "detection_type": "specification",
    "discovery_stage": "post_training",
    "fix_known": "RM uncertainty estimation, red-teaming, ensemble disagreement, scaling law extrapolation",
    "compute_cost_class": "high",
    "is_negative_example": True,
    "encoding_rationale": {
        "limitation_example": "goodhart fundamentally cannot analyze learned rewards — the structure is simple (RM score - KL), the problem is inside the RM",
        "purest_goodhart": "Goodhart's Law in its purest form: proxy diverges from target under optimization",
    },
}


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
