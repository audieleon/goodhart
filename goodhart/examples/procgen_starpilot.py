"""Example: Procgen StarPilot -- dense event rewards + generalization.

Dense event rewards (+1 enemy, +0.5 star) with level progression.
Source: Cobbe et al. 2020 (NeurIPS, Procgen Benchmark)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "procgen_starpilot",
    "source_paper": "Cobbe et al. 2020 (NeurIPS, Procgen Benchmark)",
    "paper_url": "https://arxiv.org/abs/1912.01588",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2020,
    "domain": "game_ai",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to progress through levels. Dense event rewards provide good learning signal; main challenge is generalization, not reward structure.",
    "documented_failure": "N/A — well-designed reward structure with dense events. The challenge is train-to-test generalization across procedural levels, not reward hacking.",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "high",
    "is_negative_example": True,
    "encoding_rationale": {
        "dense_events": "Multiple event rewards provide learning signal throughout the episode",
        "level_completion_bonus": "Terminal bonus incentivizes progression over farming",
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Procgen StarPilot — dense event rewards")
    print("=" * 70)
    print()
    print("Source: Cobbe et al. 2020 (NeurIPS, Procgen Benchmark)")
    print("Enemies give +1.0, stars give +0.5, per-level progression.")
    print()

    model = EnvironmentModel(
        name="Procgen StarPilot",
        max_steps=1000,
        gamma=0.999,
        n_states=100000,
        n_actions=15,
        death_probability=0.05,
    )
    model.add_reward_source(RewardSource(
        name="enemy_destroyed",
        reward_type=RewardType.ON_EVENT,
        value=1.0,
        max_occurrences=0,  # unlimited
        requires_action=True,
        discovery_probability=0.8,
    ))
    model.add_reward_source(RewardSource(
        name="star_collected",
        reward_type=RewardType.ON_EVENT,
        value=0.5,
        max_occurrences=0,
        requires_action=True,
        discovery_probability=0.9,
    ))
    model.add_reward_source(RewardSource(
        name="level_complete",
        reward_type=RewardType.TERMINAL,
        value=10.0,
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.3,
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=5e-4,
        entropy_coeff=0.01,
        num_epochs=3,
        clip_epsilon=0.2,
        num_envs=64,
        n_actors=64,
        total_steps=200_000_000,
        rollout_length=256,
    )
    engine.print_report(model, config)

    print()
    print("StarPilot's reward structure is well-designed: dense events")
    print("provide learning signal throughout the episode, and the")
    print("level completion bonus incentivizes progression over farming.")
    print("The main challenge is generalization across procedural levels,")
    print("not reward structure — which goodhart correctly doesn't flag.")


if __name__ == "__main__":
    run_example()
