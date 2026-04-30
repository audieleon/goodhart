"""SKATER humanoid roller skating (Gu et al. 2026, USTC).

22-component reward from Table I: velocity tracking, alive bonus,
base/joint/posture/foot/wheel penalties, symmetry rewards.
25-DOF humanoid with passive wheels. 75% impact reduction vs walking.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "humanoid_skating",
    "source_paper": (
        'Gu, Yuan, Shi, Huang, Zhang, Zhang, Wang, Gao & Zhang, '
        '"SKATER: Synthesized Kinematics for Advanced Traversing '
        'Efficiency on a Humanoid Robot via Roller Skate Swizzles," '
        'arXiv:2601.04948, 2026'
    ),
    "paper_url": "https://arxiv.org/abs/2601.04948",
    "source_code_url": None,
    "reward_location": (
        "Section III-B.4, Table I: 22 reward terms across 8 categories. "
        "Task: velocity tracking (3.2, 1.2), alive (0.15). "
        "Base penalties: angular vel (-0.05), joint vel (-0.001), "
        "joint accel (-2.5e-7), action rate (-0.05), joint limits (-5.0), "
        "energy (-2e-5). Joint: arms (-0.4), waist (-2.0), head (-1.0), "
        "ankle (-0.2). Posture: orientation (-7.0), height (-2.0). "
        "Foot: near (-1.0), far (-5.0). Wheel: slip (-0.1), air (-1.0). "
        "Symmetry: leg (+0.5), arm (+0.5). Contact: undesired (-1.0). "
        "25 DOF, PPO, 4096 envs, Isaac Lab."
    ),
    "year": 2026,
    "domain": "locomotion",
    "encoding_basis": "primary_source",
    "verification_date": "2026-05-01",
    "brief_summary": (
        "25-DOF humanoid learns roller skate swizzle gaits via RL. "
        "22-component implicit gait reward — no explicit trajectories "
        "or phase timing. Symmetry constraints and foot distance bounds "
        "let swizzle gaits emerge naturally. 75.86% impact reduction, "
        "63.34% cost-of-transport reduction vs bipedal walking."
    ),
    "documented_failure": "None — well-designed reward, sim-to-real deployed.",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "post_deployment",
    "fix_known": None,
    "compute_cost_class": "medium",
    "is_negative_example": True,
    "encoding_rationale": {
        "22_components_simplified": (
            "22 terms simplified to representative groups: task "
            "tracking (positive), regularization (negative), "
            "symmetry (positive), safety (negative). Full Table I "
            "weights documented in reward_location."
        ),
        "implicit_gait": (
            "No explicit gait trajectories or phase timing. The "
            "swizzle pattern emerges from symmetry rewards + foot "
            "distance constraints (0.2m-0.5m). This is the key "
            "design insight — let the reward shape the gait."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Table I: 22 terms, simplified to representative groups
    # 25 DOF humanoid, PPO, Isaac Lab

    model = EnvironmentModel(
        name="SKATER Humanoid Skating (Gu et al. 2026)",
        max_steps=1000,  # 20s at 50Hz
        gamma=0.99,
        n_actions=25,  # 25 DOF
        action_type="continuous",
        death_probability=0.01,
    )

    # Task: velocity tracking (exp(-error²/σ²), weights 3.2 + 1.2)
    model.add_reward_source(RewardSource(
        name="velocity_tracking",
        reward_type=RewardType.PER_STEP,
        value=4.4,  # 3.2 linear + 1.2 angular
        requires_action=True,
        intentional=True,
        state_dependent=True,
        scales_with="velocity",
        value_type="exponential",
    ))

    # Alive bonus (constant 0.15)
    model.add_reward_source(RewardSource(
        name="alive_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.15,
        requires_action=False,
        intentional=True,
    ))

    # Symmetry rewards (leg + arm, each 0.5)
    model.add_reward_source(RewardSource(
        name="symmetry_reward",
        reward_type=RewardType.PER_STEP,
        value=1.0,  # 0.5 leg + 0.5 arm
        requires_action=True,
        state_dependent=True,
    ))

    # Regularization penalties (base + joint + posture + wheel)
    # Aggregated: orientation (-7.0 dominant), joint limits (-5.0),
    # feet too far (-5.0), waist (-2.0), height (-2.0), etc.
    model.add_reward_source(RewardSource(
        name="regularization_penalties",
        reward_type=RewardType.PER_STEP,
        value=-1.5,  # weighted average of 15 penalty terms
        requires_action=True,
        state_dependent=True,
    ))

    # Safety: undesired contacts + foot constraints
    model.add_reward_source(RewardSource(
        name="safety_penalties",
        reward_type=RewardType.ON_EVENT,
        value=-5.0,  # joint limits + foot bounds + contacts
        requires_action=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
