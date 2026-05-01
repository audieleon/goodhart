"""HuB: Extreme humanoid balance (Zhang et al. 2025, CoRL).

20+ reward terms for quasi-static balance on Unitree G1.
Balance shaping (COM, foot contact, close feet) + motion tracking
+ joint limit penalties + regularization. Well-designed reward
with careful penalty structure for sim-to-real transfer.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "hub_humanoid_balance",
    "source_paper": (
        'Zhang, Zheng, Nai, Hu, Wang, Chen, Lin, Li, Hong, Sreenath '
        '& Gao, "HuB: Learning Extreme Humanoid Balance," CoRL 2025'
    ),
    "paper_url": "https://arxiv.org/abs/2505.07294",
    "source_code_url": None,
    "reward_location": (
        "Appendix B.2, Table 5: 20+ reward components with exact "
        "weights. Balance Shaping: COM (w=160), foot contact mismatch "
        "(w=-250), close feet (w=-1000). Tracking: body pos (30), rot "
        "(20), vel (8), ang vel (8), DoF pos (32), DoF vel (16). "
        "Penalties: torque limits (-0.5), DoF pos limits (-30), DoF "
        "vel limits (-12), termination (-60). Regularization: torque "
        "(-2.5e-5), DoF vel (-1e-3), DoF acc (-3e-6), action rate "
        "(-1.5), feet air time (250), contact force (-0.2), stumble "
        "(-3e-4), slippage (-30), feet orientation (-62.5), in air (-50). "
        "Section 3.1: gamma=0.99, PPO, 29 DoF actions."
    ),
    "year": 2025,
    "domain": "locomotion",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Humanoid robot learns extreme quasi-static balance poses "
        "(single-leg stands, high kicks). Reward carefully designed "
        "with strong balance shaping, tracking, and extensive "
        "regularization. Successfully transfers sim-to-real."
    ),
    "documented_failure": None,
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": None,
    "fix_known": None,
    "compute_cost_class": "high",
    "is_negative_example": True,
    "encoding_rationale": {
        "com_reward": (
            "w=160, exp(-||p_xy - p_xy^com||^2/sigma_com^2) * "
            "indicator(||p_z^foot - p_z^foot||>0.05). Dominant "
            "positive term encouraging COM within support polygon."
        ),
        "foot_contact_mismatch": (
            "w=-250, penalizes wrong foot contact state vs reference. "
            "Large negative prevents unintended ground contact."
        ),
        "close_feet": (
            "w=-1000, max{0.16 - ||p^foot - p^foot||, 0}. Largest "
            "single penalty, prevents inter-foot collision."
        ),
        "tracking_rewards": (
            "Body pos (30), rot (20), vel (8), ang vel (8), DoF pos "
            "(32), DoF vel (16). All exp(-||error||^2/sigma^2) form "
            "with task-specific sigma tolerances."
        ),
        "regularization": (
            "11 regularization terms with small weights. Feet air "
            "time (250) is the notable positive regularizer. Most "
            "others are small negative penalties."
        ),
        "negative_example_rationale": (
            "Well-designed reward with hierarchical structure: large "
            "safety penalties > balance shaping > tracking > "
            "regularization. Successfully deployed on real robot."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Zhang et al. 2025: Unitree G1 humanoid balance
    # PPO, 29 DoF, gamma=0.99, teacher-student
    model = EnvironmentModel(
        name="HuB: Extreme Humanoid Balance (Zhang et al. 2025)",
        max_steps=1000,
        gamma=0.99,
        n_states=100000,
        n_actions=29,
        action_type="continuous",
        death_probability=0.01,
    )

    # Balance Shaping Rewards
    # COM reward: w=160, exp(-||error||^2/sigma^2)
    model.add_reward_source(RewardSource(
        name="com_reward",
        reward_type=RewardType.PER_STEP,
        value=160.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Foot contact mismatch penalty: w=-250
    model.add_reward_source(RewardSource(
        name="foot_contact_mismatch",
        reward_type=RewardType.PER_STEP,
        value=-250.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Close feet penalty: w=-1000
    model.add_reward_source(RewardSource(
        name="close_feet_penalty",
        reward_type=RewardType.PER_STEP,
        value=-1000.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Tracking rewards (aggregated): body pos/rot/vel/angvel/dof
    # Total tracking weight: 30+20+8+8+32+16 = 114
    model.add_reward_source(RewardSource(
        name="motion_tracking",
        reward_type=RewardType.PER_STEP,
        value=114.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Feet air time reward: w=250
    model.add_reward_source(RewardSource(
        name="feet_air_time",
        reward_type=RewardType.PER_STEP,
        value=250.0,
        state_dependent=True,
        requires_action=False,
        intentional=True,
    ))

    # Termination penalty: w=-60
    model.add_reward_source(RewardSource(
        name="termination_penalty",
        reward_type=RewardType.TERMINAL,
        value=-60.0,
        state_dependent=False,
        requires_action=False,
        intentional=True,
    ))

    # Joint limit penalties (torque + pos + vel): -0.5 + -30 + -12
    model.add_reward_source(RewardSource(
        name="joint_limit_penalties",
        reward_type=RewardType.PER_STEP,
        value=-42.5,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Regularization (small penalties): torque, dof_vel, dof_acc,
    # action_rate, contact_force, stumble, slippage, feet_orient, in_air
    # Total ~-175.7 at max (but most are tiny)
    model.add_reward_source(RewardSource(
        name="regularization",
        reward_type=RewardType.PER_STEP,
        value=-5.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
