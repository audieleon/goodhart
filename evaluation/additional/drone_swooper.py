"""Swooper high-speed aerial grasping (Huang et al. 2026, RA-L).

Two-stage reward: flight (Eq. 2-3) then grasping (Eq. 4).
Documented failure: training from scratch converges to 0%
due to flight/grasping reward conflict. Sim-to-real at 1.5 m/s.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "drone_swooper",
    "source_paper": (
        'Huang, Niu, Chai, Jin & Zou, "Swooper: Learning High-Speed '
        'Aerial Grasping with a Simple Gripper," '
        'IEEE RA-L, 2026'
    ),
    "paper_url": "https://arxiv.org/abs/2603.05935",
    "source_code_url": "https://zikenhuang.github.io/Swooper/",
    "reward_location": (
        "Section IV-B, Eq. 2-3: r_fly = r_P + r_psi + r_sm + r_safe. "
        "λ₁=1.0 (pos), λ₂=0.1 (yaw), λ₃=0.1 λ₄=0.4 (smooth), "
        "-10 (crash). Section IV-C, Eq. 4: r_grasp = r_phase + "
        "r_gp_instr + r_gp_sm + r_crash. λ₅=10.0 (phase), λ₆=4.0 "
        "(gripper), λ₇=5.0 (smooth). 5D actions (4 CTBR + 1 gripper). "
        "PPO, MLP 2×64. Two-stage curriculum."
    ),
    "year": 2026,
    "domain": "safety_critical",
    "encoding_basis": "primary_source",
    "verification_date": "2026-05-01",
    "brief_summary": (
        "Quadrotor learns to grasp objects at 1.5 m/s during high-speed "
        "flight. Two-stage reward: first learn to fly (position + yaw + "
        "smoothness), then fine-tune for grasping (phase completion + "
        "gripper instruction). Training from scratch fails — flight "
        "and grasping objectives conflict."
    ),
    "documented_failure": (
        "Training From Scratch (TFS) converges to 0% success rate. "
        "The agent learns gripper control before mastering flight, "
        "causing premature grasps that knock objects and trigger "
        "crash penalties (-10). This hinders flight learning. The "
        "two-stage curriculum (fly first, then grasp) is the fix. "
        "Ablation: removing r_phase causes catastrophic forgetting; "
        "removing r_gp_instr causes gripper-object collisions."
    ),
    "failure_mechanism": "compound_trap",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Two-stage curriculum: pre-train flight control (Eq. 2-3), "
        "then fine-tune with grasping reward (Eq. 4). Decouples "
        "conflicting objectives. <60 min total training on RTX 3060."
    ),
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "two_stage_conflict": (
            "The documented failure is the COMBINED reward (TFS). "
            "Encoding the grasping stage (Eq. 4) which includes "
            "the conflicting components. The flight reward (Eq. 2-3) "
            "works fine alone — the conflict is in combining them."
        ),
        "crash_penalty_trap": (
            "r_crash = -10 per violation. When combined with grasping "
            "rewards, premature grasp attempts trigger crashes, which "
            "then suppress exploration of flight skills. This is a "
            "compound_trap: two reward components interact to produce "
            "a worse outcome than either alone."
        ),
        "staged_rewards": (
            "r_phase uses hierarchical sparse rewards (λ₅=10.0) for "
            "approaching → grasping → lifting. Without this staged "
            "structure, the agent suffers catastrophic forgetting."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Combined reward (the failure mode when training from scratch)
    # r = r_fly + r_grasp = (r_P + r_psi + r_sm + r_safe) +
    #     (r_phase + r_gp_instr + r_gp_sm + r_crash)

    model = EnvironmentModel(
        name="Swooper Aerial Grasping (Huang et al. 2026)",
        max_steps=500,  # ~5s at 100Hz
        gamma=0.99,
        n_actions=5,  # 4 flight (CTBR) + 1 gripper
        action_type="continuous",
        death_probability=0.05,  # crash terminates
    )

    # Position tracking: r_P = -λ₁ ||p - p_d||, λ₁=1.0
    model.add_reward_source(RewardSource(
        name="position_tracking",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
        requires_action=True,
        intentional=True,
        state_dependent=True,
        scales_with="distance",
    ))

    # Yaw alignment: r_ψ = 1 - exp(λ₂ ||ψ-ψ_d||²), λ₂=0.1
    model.add_reward_source(RewardSource(
        name="yaw_alignment",
        reward_type=RewardType.PER_STEP,
        value=0.5,  # range [0, 1]
        requires_action=True,
        state_dependent=True,
        value_range=(0.0, 1.0),
    ))

    # Action smoothness: -λ₃||ω||² - λ₄||a-a₋₁||
    model.add_reward_source(RewardSource(
        name="smoothness_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.2,  # λ₃=0.1, λ₄=0.4
        requires_action=True,
        state_dependent=True,
    ))

    # Phase completion: λ₅ · 𝟙{phase completed}, λ₅=10.0
    model.add_reward_source(RewardSource(
        name="phase_completion",
        reward_type=RewardType.ON_EVENT,
        value=10.0,  # sparse bonus per phase
        requires_action=True,
        intentional=True,
        prerequisite="position_tracking",
    ))

    # Crash penalty: -10 · 𝟙{safety violated}
    model.add_reward_source(RewardSource(
        name="crash_penalty",
        reward_type=RewardType.ON_EVENT,
        value=-10.0,
        requires_action=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
