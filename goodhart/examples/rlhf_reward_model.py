"""Example: RLHF reward model gaming (LIMITATION).

Agents exploit gaps in learned reward models -- sycophancy, verbosity hacking, format gaming.
Source: Casper et al. 2023, Gao et al. 2023 ("Scaling Laws for Reward Model Overoptimization")
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "rlhf_reward_model",
    "source_paper": "Casper et al. 2023 ('Open Problems and Fundamental Limitations of RLHF'), Gao et al. 2023 ('Scaling Laws for Reward Model Overoptimization')",
    "paper_url": "https://arxiv.org/abs/2307.15217",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2023,
    "domain": "rlhf",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to produce high-quality answers. Instead it learned to produce outputs that score high on the reward model but are low quality (sycophancy, verbosity, format gaming).",
    "documented_failure": "Learned reward model is a black box proxy for human preferences. Agents exploit gaps: sycophancy, verbosity hacking, format gaming. RM score improves while actual quality decreases.",
    "failure_mechanism": None,
    "detection_type": "specification",
    "discovery_stage": "post_training",
    "fix_known": "RM uncertainty estimation, ensemble disagreement, scaling law extrapolation, red-teaming",
    "compute_cost_class": "high",
    "is_negative_example": True,
    "encoding_rationale": {
        "limitation_example": "goodhart cannot decompose a learned reward model into RewardSources with known dynamics",
        "purest_goodhart": "The purest embodiment of Goodhart's Law is beyond the tool's expressiveness",
    },
}


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
    engine.print_report(model, config)

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
