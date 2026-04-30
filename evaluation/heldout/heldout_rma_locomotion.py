"""Rapid Motor Adaptation — held-out evaluation (negative example).

Kumar et al. 2021 well-structured locomotion with bioenergetics
penalties. Deployed on real A1 with no documented Goodhart failures.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "heldout_rma_locomotion",
    "source_paper": (
        'Kumar et al. 2021, "Rapid Motor Adaptation for Legged'
        ' Robots," RSS 2021'
    ),
    "paper_url": "https://arxiv.org/abs/2107.04034",
    "source_code_url": None,
    "reward_location": "Paper reward description",
    "year": 2021,
    "domain": "locomotion",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Well-structured locomotion reward with bioenergetics-"
        "inspired penalties and curriculum. Successfully deployed"
        " on real Unitree A1 across diverse terrains."
    ),
    "documented_failure": "None — reward is well-designed.",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "high",
    "is_negative_example": True,
    "encoding_rationale": {
        "well_balanced": (
            "Forward velocity goal (effective 7.0/step) is"
            " well-balanced against lateral/rotation penalties."
        ),
        "curriculum": (
            "Penalties start at 0.03x and grow to 1.0x,"
            " preventing penalty avoidance during early training."
        ),
        "real_deployment": (
            "Successfully deployed on real hardware across"
            " diverse terrains with no Goodhart failures."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Rapid Motor Adaptation — held-out evaluation")
    print("=" * 70)
    print()
    print("Source: Kumar et al. 2021 (RSS), arXiv:2107.04034")
    print("Unitree A1, 11 reward components, PPO + adaptation module.")
    print("Bioenergetics-inspired penalties. Deployed on real hardware.")
    print()

    # Encode with final penalty weights (after curriculum convergence)
    model = EnvironmentModel(
        name="RMA Locomotion (Held-Out)",
        max_steps=1000,
        gamma=0.998,
        n_actions=12,
        action_type="continuous",
        death_probability=0.0,  # termination is a reset, not death
    )
    # Goal: forward velocity (clamped at 0.35 m/s, weight 20)
    # Effective max per-step value: 20 * 0.35 = 7.0
    model.add_reward_source(RewardSource(
        name="forward_velocity",
        reward_type=RewardType.PER_STEP,
        value=7.0,  # 20 * 0.35 (clamped maximum)
        requires_action=True,
        intentional=True,
        explore_fraction=0.3,
    ))
    # Lateral movement penalty (passive — penalizes drift)
    model.add_reward_source(RewardSource(
        name="lateral_movement",
        reward_type=RewardType.PER_STEP,
        value=-0.5,  # 20 * typical v_y^2 ~ 0.025
        value_range=(-2.0, 0.0),
        requires_action=False,
    ))
    # Rotation penalty (passive — penalizes yaw drift)
    model.add_reward_source(RewardSource(
        name="rotation",
        reward_type=RewardType.PER_STEP,
        value=-0.5,
        value_range=(-2.0, 0.0),
        requires_action=False,
    ))
    # Orientation penalty (passive — penalizes roll/pitch)
    model.add_reward_source(RewardSource(
        name="orientation",
        reward_type=RewardType.PER_STEP,
        value=-1.5,
        requires_action=False,
    ))
    # Z velocity penalty (passive — penalizes bouncing)
    model.add_reward_source(RewardSource(
        name="z_acceleration",
        reward_type=RewardType.PER_STEP,
        value=-2.0,
        requires_action=False,
    ))
    # Foot slip penalty (passive)
    model.add_reward_source(RewardSource(
        name="foot_slip",
        reward_type=RewardType.PER_STEP,
        value=-0.8,
        requires_action=False,
    ))
    # Energy/work penalty (action-dependent, bioenergetics-inspired)
    model.add_reward_source(RewardSource(
        name="work",
        reward_type=RewardType.PER_STEP,
        value=-0.002,
        requires_action=True,
    ))
    # Ground impact penalty (passive)
    model.add_reward_source(RewardSource(
        name="ground_impact",
        reward_type=RewardType.PER_STEP,
        value=-0.02,
        requires_action=False,
    ))
    # Smoothness penalty (action-dependent)
    model.add_reward_source(RewardSource(
        name="smoothness",
        reward_type=RewardType.PER_STEP,
        value=-0.001,
        requires_action=True,
    ))
    # Action magnitude penalty
    model.add_reward_source(RewardSource(
        name="action_magnitude",
        reward_type=RewardType.PER_STEP,
        value=-0.07,
        requires_action=True,
    ))
    # Joint speed penalty (passive)
    model.add_reward_source(RewardSource(
        name="joint_speed",
        reward_type=RewardType.PER_STEP,
        value=-0.002,
        requires_action=False,
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=1e-3,
        entropy_coeff=0.01,
        clip_epsilon=0.2,
        num_epochs=5,
        total_steps=200_000_000,
    )
    engine.print_report(model, config)

    print()
    print("Forward velocity goal (effective 7.0/step) is well-balanced")
    print("against lateral/rotation penalties. Bioenergetics penalties")
    print("(work, ground impact, smoothness) are small but prevent")
    print("high-energy gaits. No documented Goodhart failures;")
    print("successfully deployed on real A1 across diverse terrains.")
    print("The curriculum (penalties start at 0.03x, grow to 1.0x)")
    print("prevents penalty avoidance during early training.")


if __name__ == "__main__":
    run_example()
