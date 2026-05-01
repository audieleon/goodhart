"""Wildfire mitigation RL (Tapley et al. 2023, NeurIPS Climate).

Agent places firelines to limit fire spread. Reward minimizes
total burned area and preserves structures. Real terrain data.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "wildfire_mitigation",
    "source_paper": (
        'Tapley, Dotter, Doyle, Fennelly, Gandikota, Smith, Threet & Welsh, '
        '"Reinforcement Learning for Wildfire Mitigation in Simulated '
        'Disaster Environments," NeurIPS 2023 Climate Workshop'
    ),
    "paper_url": "https://arxiv.org/abs/2311.15925",
    "source_code_url": "https://github.com/mitrefireline/simharness",
    "reward_location": (
        "Section 3.2: reward is configurable via SimHarness API. "
        "Section 4: DQN agent trained on Mineral Fire 2020 (Coalinga, CA). "
        "128x128 grid, 30 sq meters per unit. Agent places firelines. "
        "Reward from code: negative burned area + structure preservation."
    ),
    "year": 2023,
    "domain": "safety_critical",
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "RL agent acts as a firefighter placing firelines to mitigate "
        "wildfire spread on real terrain data. Reward minimizes total "
        "fire damage (burned cells) while preserving high-value areas."
    ),
    "documented_failure": (
        "None documented — framework/benchmark paper. The reward is "
        "configurable. Default minimizes burned area."
    ),
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "medium",
    "is_negative_example": True,
    "encoding_rationale": {
        "reward_from_framework": (
            "Paper describes configurable reward API, not a fixed "
            "formula. Encoding based on default: negative burned area "
            "with optional structure value preservation. "
            "encoding_basis=code_derived from simharness repo."
        ),
        "real_world_terrain": (
            "Uses real LANDFIRE fuel/terrain data and Rothermel "
            "fire spread model. GPS coordinates: 36.09, -120.52. "
            "This is a real-world disaster scenario, not synthetic."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Agent places firelines on 128x128 grid
    # Reward: minimize fire damage
    # DQN, discrete actions (place fireline at location)

    model = EnvironmentModel(
        name="Wildfire Mitigation (Tapley et al. 2023)",
        max_steps=500,  # fire simulation steps
        gamma=0.99,
        n_actions=16384,  # 128x128 grid positions
        action_type="discrete",
        death_probability=0.0,
    )

    # Negative reward: fire damage (burned cells)
    # Each burned cell reduces reward
    model.add_reward_source(RewardSource(
        name="fire_damage",
        reward_type=RewardType.PER_STEP,
        value=-1.0,  # penalty per burned cell (normalized)
        requires_action=False,  # fire spreads regardless of action
        state_dependent=True,
    ))

    # Positive reward: saved area (cells protected by firelines)
    model.add_reward_source(RewardSource(
        name="area_preserved",
        reward_type=RewardType.PER_STEP,
        value=0.1,  # smaller positive for preserved cells
        requires_action=True,  # fireline placement required
        intentional=True,
        state_dependent=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
