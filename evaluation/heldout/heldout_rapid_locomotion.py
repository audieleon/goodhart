"""Rapid Locomotion — held-out evaluation.

Margolis et al. 2022 quadruped locomotion with body-sinking exploit.
Penalty-to-goal ratio is 30:1, creating energy-minimization shortcut.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "heldout_rapid_locomotion",
    "source_paper": (
        'Margolis et al. 2022, "Rapid Locomotion via Reinforcement'
        ' Learning," RSS 2022 / IJRR 2024'
    ),
    "paper_url": None,
    "source_code_url": None,
    "reward_location": "Table VI",
    "year": 2022,
    "domain": "locomotion",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Agent was supposed to walk at high speed. Instead it"
        " sinks its body and leans forward to minimize energy"
        " penalties — unsafe posture that fails on real hardware."
    ),
    "documented_failure": (
        "Base height penalty (-0.6) is 30x the goal reward (+0.02)."
        " Without the height fix, passive penalties dominate and"
        " create an energy-minimization exploit: the robot sinks"
        " its body and leans forward at high speed. This posture"
        " is reward-optimal but physically unstable and doesn't"
        " transfer to real hardware."
    ),
    "failure_mechanism": "shaping_dominance",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Added base_height_penalty (-0.6) and"
        " base_orientation_penalty (-0.002) to prevent sinking."
    ),
    "compute_cost_class": "high",
    "is_negative_example": False,
    "encoding_rationale": {
        "penalty_dominance": (
            "Base height penalty at 30x the goal reward"
            " creates extreme dominance of passive penalties."
        ),
        "encodes_fixed_version": (
            "Encodes the FIXED version (with height/orientation"
            " penalties) to test whether the tool flags the"
            " extreme penalty-to-goal ratio."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Rapid Locomotion — held-out evaluation (body-sinking exploit)")
    print("=" * 70)
    print()
    print("Source: Margolis et al. 2022 (RSS / IJRR 2024)")
    print("MIT Mini Cheetah, 12 reward components from Table VI.")
    print("Body-sinking exploit: robot crouches and leans at high speed.")
    print()

    # Encode the FIXED version (with height/orientation penalties)
    # to test whether the tool flags the extreme penalty-to-goal ratio
    model = EnvironmentModel(
        name="Rapid Locomotion (Held-Out)",
        max_steps=1000,
        gamma=0.99,
        n_actions=12,
        action_type="continuous",
    )
    # Goal rewards (tiny magnitude)
    model.add_reward_source(RewardSource(
        name="xy_velocity_tracking",
        reward_type=RewardType.PER_STEP,
        value=0.02,
        requires_action=True,
        intentional=True,
        explore_fraction=0.3,
    ))
    model.add_reward_source(RewardSource(
        name="yaw_velocity_tracking",
        reward_type=RewardType.PER_STEP,
        value=0.01,
        requires_action=True,
        intentional=True,
        explore_fraction=0.2,
    ))
    # Foot airtime bonus (active auxiliary)
    model.add_reward_source(RewardSource(
        name="foot_airtime",
        reward_type=RewardType.PER_STEP,
        value=0.02,
        requires_action=True,
        intentional=False,
    ))
    # Passive penalties (regularization)
    model.add_reward_source(RewardSource(
        name="z_velocity_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.04,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="roll_pitch_angular_vel",
        reward_type=RewardType.PER_STEP,
        value=-0.001,
        requires_action=False,
    ))
    # The FIX: base height penalty (30x the goal reward!)
    model.add_reward_source(RewardSource(
        name="base_height_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.6,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="base_orientation_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.002,
        requires_action=False,
    ))
    # Safety penalties
    model.add_reward_source(RewardSource(
        name="self_collision",
        reward_type=RewardType.PER_STEP,
        value=-0.02,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="joint_limit_violation",
        reward_type=RewardType.PER_STEP,
        value=-0.2,
        requires_action=False,
    ))
    # Smoothness penalties (action-dependent)
    model.add_reward_source(RewardSource(
        name="joint_torques",
        reward_type=RewardType.PER_STEP,
        value=-2e-7,
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="joint_accelerations",
        reward_type=RewardType.PER_STEP,
        value=-5e-9,
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="action_rate",
        reward_type=RewardType.PER_STEP,
        value=-2e-4,
        requires_action=True,
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=1e-3,
        entropy_coeff=0.01,
        clip_epsilon=0.2,
        num_epochs=5,
        total_steps=400_000_000,
        rollout_length=21,
        num_envs=4000,
    )
    engine.print_report(model, config)

    print()
    print("The base_height_penalty (-0.6) is 30x the goal reward (+0.02).")
    print("This extreme ratio exists because without it, the robot sinks")
    print("its body and leans forward at high speed — minimizing the energy")
    print("penalties while tracking velocity. The fix (adding the height")
    print("penalty) creates a new dominance pattern that the tool should flag.")


if __name__ == "__main__":
    run_example()
