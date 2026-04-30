"""Humanoid Parkour Learning — held-out evaluation (negative example).

Zhuang et al. 2024 humanoid with 17 reward components, no alive
bonus, well-calibrated penalties. No documented Goodhart failures.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "heldout_humanoid_parkour",
    "source_paper": (
        'Zhuang, Yao & Zhao 2024, "Humanoid Parkour Learning,"'
        " CoRL 2024"
    ),
    "paper_url": "https://arxiv.org/abs/2406.10759",
    "source_code_url": None,
    "reward_location": "Table 4",
    "year": 2024,
    "domain": "locomotion",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Well-structured 17-component reward with no alive bonus."
        " Intentional rewards dominate, diverse behaviors emerge"
        " from terrain curriculum."
    ),
    "documented_failure": "None — reward is well-designed.",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "high",
    "is_negative_example": True,
    "encoding_rationale": {
        "no_alive_bonus": (
            "Avoids the classic stand-still trap by omitting"
            " alive bonus entirely."
        ),
        "well_calibrated": (
            "Intentional rewards (2.5/step) dominate small"
            " regularization penalties."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Humanoid Parkour — held-out evaluation")
    print("=" * 70)
    print()
    print("Source: Zhuang, Yao, Zhao 2024 (CoRL), arXiv:2406.10759")
    print("Unitree H1 humanoid, 17 reward components from Table 4.")
    print("No alive bonus. Diverse parkour from terrain curriculum.")
    print()

    model = EnvironmentModel(
        name="Humanoid Parkour (Held-Out)",
        max_steps=1000,
        gamma=0.99,
        n_actions=19,
        action_type="continuous",
        death_probability=0.0,
    )
    # Task rewards (active, intentional)
    model.add_reward_source(RewardSource(
        name="linear_velocity_tracking",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=True,
        intentional=True,
        explore_fraction=0.3,
    ))
    model.add_reward_source(RewardSource(
        name="angular_velocity_tracking",
        reward_type=RewardType.PER_STEP,
        value=1.5,
        requires_action=True,
        intentional=True,
        explore_fraction=0.2,
    ))
    # Feet apart bonus (active auxiliary)
    model.add_reward_source(RewardSource(
        name="feet_apart",
        reward_type=RewardType.PER_STEP,
        value=0.4,
        requires_action=True,
        intentional=False,
    ))
    # Orientation penalty (passive)
    model.add_reward_source(RewardSource(
        name="orientation",
        reward_type=RewardType.PER_STEP,
        value=-2.0,
        requires_action=False,
    ))
    # Collision penalty (passive, large)
    model.add_reward_source(RewardSource(
        name="collision",
        reward_type=RewardType.PER_STEP,
        value=-10.0,
        requires_action=False,
    ))
    # Posture penalties (passive)
    model.add_reward_source(RewardSource(
        name="arm_dof_error",
        reward_type=RewardType.PER_STEP,
        value=-0.3,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="waist_dof_error",
        reward_type=RewardType.PER_STEP,
        value=-0.1,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="hip_yaw_dof_error",
        reward_type=RewardType.PER_STEP,
        value=-0.1,
        requires_action=False,
    ))
    # Energy/smoothness penalties (small, action-dependent)
    model.add_reward_source(RewardSource(
        name="energy",
        reward_type=RewardType.PER_STEP,
        value=-2.5e-7,
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="dof_velocity",
        reward_type=RewardType.PER_STEP,
        value=-1e-4,
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="dof_acceleration",
        reward_type=RewardType.PER_STEP,
        value=-2e-6,
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="action_rate",
        reward_type=RewardType.PER_STEP,
        value=-6e-3,
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="weighted_torques",
        reward_type=RewardType.PER_STEP,
        value=-1e-7,
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="contact_forces",
        reward_type=RewardType.PER_STEP,
        value=-3e-4,
        requires_action=False,
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-5,
        clip_epsilon=0.2,
        num_epochs=5,
        num_envs=4096,
        rollout_length=24,
    )
    engine.print_report(model, config)

    print()
    print("No alive bonus avoids the stand-still trap. Intentional rewards")
    print("(1.0 + 1.5 = 2.5/step) dominate small regularization penalties.")
    print("The large collision penalty (-10.0) is appropriate for a parkour")
    print("task where body contacts are dangerous. No documented Goodhart")
    print("failures; diverse behaviors emerge from terrain curriculum.")


if __name__ == "__main__":
    run_example()
