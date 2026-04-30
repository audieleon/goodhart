"""Agile Soccer Skills — held-out evaluation.

Haarnoja et al. 2024 bipedal soccer reward with 20,000x magnitude
ratio between scoring and shaping. Documented knee gear breakage.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "heldout_agile_soccer",
    "source_paper": (
        'Haarnoja et al. 2024, "Learning Agile Soccer Skills for a'
        ' Bipedal Robot with Deep Reinforcement Learning,"'
        " Science Robotics 2024"
    ),
    "paper_url": "https://arxiv.org/abs/2304.13653",
    "source_code_url": None,
    "reward_location": "Table S3",
    "year": 2024,
    "domain": "locomotion",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Agent was supposed to play soccer with bipedal walking."
        " Instead it broke knee gears from aggressive kicking,"
        " fell forward on real hardware, and rolled to the ball"
        " instead of walking."
    ),
    "documented_failure": (
        "Scoring reward (1000) is 20,000x the velocity-to-ball"
        " shaping (0.05), creating extreme dominance. The agent"
        " sacrifices everything for goals: (1) aggressive kicking"
        " broke knee gears, (2) forward-leaning gait caused falls"
        " on real hardware, (3) without skill regularization the"
        " agent learned to roll to the ball instead of walking"
        " (Figure 7B). Fixes: joint_torque penalty (0.01) and"
        " upright reward (0.015) were added AFTER observing"
        " hardware damage."
    ),
    "failure_mechanism": "shaping_dominance",
    "detection_type": "structural",
    "discovery_stage": "post_training",
    "fix_known": (
        "Added joint_torque penalty (0.01) and upright reward"
        " (0.015) after observing hardware damage."
    ),
    "compute_cost_class": "high",
    "is_negative_example": False,
    "encoding_rationale": {
        "scoring_dominance": (
            "1000 vs 0.05 = 20,000x ratio; agent optimizes"
            " scoring at the cost of physical safety."
        ),
        "post_training_fix": (
            "Penalties were added after hardware damage,"
            " confirming the original reward was misspecified."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Agile Soccer Skills — held-out evaluation")
    print("=" * 70)
    print()
    print("Source: Haarnoja et al. 2024 (Science Robotics)")
    print("20-DOF bipedal robot, soccer skill stage, Table S3 weights.")
    print("Documented: knee gear breakage, forward-lean falls,")
    print("rolling-to-ball degenerate strategy.")
    print()

    # Encode the soccer skill stage (Stage 1)
    model = EnvironmentModel(
        name="Agile Soccer (Held-Out)",
        max_steps=2000,  # 50s at 40Hz
        gamma=0.99,
        n_actions=20,
        action_type="continuous",
        death_probability=0.01,  # falls/out-of-bounds terminate
    )
    # Scoring: sparse terminal goal (weight 1000)
    model.add_reward_source(RewardSource(
        name="scoring",
        reward_type=RewardType.TERMINAL,
        value=1000.0,
        requires_action=True,
        intentional=True,
        discovery_probability=0.01,  # rare event: must walk, aim, kick
    ))
    # Dense shaping: velocity toward ball
    model.add_reward_source(RewardSource(
        name="velocity_to_ball",
        reward_type=RewardType.PER_STEP,
        value=0.05,
        requires_action=True,
        intentional=False,
        explore_fraction=0.2,
    ))
    # Dense shaping: forward velocity
    model.add_reward_source(RewardSource(
        name="velocity",
        reward_type=RewardType.PER_STEP,
        value=0.1,
        requires_action=True,
        intentional=False,
        explore_fraction=0.3,
    ))
    # Interference penalty (near opponent)
    model.add_reward_source(RewardSource(
        name="interference",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
        requires_action=False,  # triggered by proximity
    ))
    # Upright posture reward (the fix for forward-lean)
    model.add_reward_source(RewardSource(
        name="upright",
        reward_type=RewardType.PER_STEP,
        value=0.015,
        requires_action=False,  # passive: being upright
        intentional=False,
    ))
    # Joint torque penalty (the fix for knee gear breakage)
    model.add_reward_source(RewardSource(
        name="joint_torque",
        reward_type=RewardType.PER_STEP,
        value=-0.01,
        requires_action=False,  # penalizes high knee torques
    ))

    engine.print_report(model)

    print()
    print("Scoring (1000) is 20,000x the velocity-to-ball shaping (0.05).")
    print("This extreme dominance means the agent sacrifices everything for")
    print("goals: aggressive kicking broke knee gears, forward-leaning gait")
    print("fell on real hardware, and without skill regularization the agent")
    print("learned to roll to the ball instead of walking (Figure 7B).")
    print("Fixes: joint_torque penalty (0.01) and upright reward (0.015)")
    print("were added AFTER observing hardware damage.")


if __name__ == "__main__":
    run_example()
