"""Extreme Parkour — held-out evaluation (NoClear ablation).

Cheng et al. 2024 quadruped parkour with edge-stepping exploit
when clearance penalty is removed. Energy minimization dominates.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "heldout_extreme_parkour",
    "source_paper": (
        'Cheng et al. 2024, "Extreme Parkour with Legged Robots,"'
        " ICRA 2024"
    ),
    "paper_url": "https://arxiv.org/abs/2309.14341",
    "source_code_url": "https://github.com/chengxuxin/extreme-parkour",
    "reward_location": "Codebase reward definitions",
    "year": 2024,
    "domain": "locomotion",
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Agent was supposed to traverse parkour obstacles. Instead,"
        " without clearance penalty, it places feet on terrain"
        " edges to minimize energy — physically unstable."
    ),
    "documented_failure": (
        "NoClear ablation: removing the clearance penalty causes"
        " energy-minimization penalties (-10.0 collision, -0.5"
        " hip_pos, -0.04 dof_error, -1e-5 torques) to incentivize"
        " short steps landing on terrain edges. This is reward-"
        "optimal but physically unstable: 2.7x more edge violations"
        " in simulation, 20-80% lower success rate on real hardware."
    ),
    "failure_mechanism": "idle_exploit",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Add feet_edge clearance penalty to penalize foot"
        " placement on terrain edges."
    ),
    "compute_cost_class": "high",
    "is_negative_example": False,
    "encoding_rationale": {
        "noclear_ablation": (
            "Encodes the NoClear ablation specifically, where"
            " removing clearance penalty reveals the exploit."
        ),
        "energy_minimization": (
            "Passive penalties dominate and create an energy-"
            "minimization shortcut via edge-stepping."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Extreme Parkour — held-out evaluation (NoClear ablation)")
    print("=" * 70)
    print()
    print("Source: Cheng et al. 2024 (ICRA), arXiv:2309.14341")
    print("Unitree A1 quadruped, 15 reward components, PPO.")
    print("NoClear ablation: removing clearance penalty causes")
    print("edge-stepping exploit (2.7x more edge violations).")
    print()

    # Encode the NoClear ablation (feet_edge penalty removed)
    model = EnvironmentModel(
        name="Extreme Parkour NoClear (Held-Out)",
        max_steps=1000,  # 20s at 50Hz
        gamma=0.99,
        n_actions=12,
        action_type="continuous",
        death_probability=0.02,
    )
    # Goal: velocity tracking toward waypoints (world frame)
    model.add_reward_source(RewardSource(
        name="tracking_goal_vel",
        reward_type=RewardType.PER_STEP,
        value=1.5,
        requires_action=True,
        intentional=True,
        explore_fraction=0.3,
    ))
    # Yaw tracking toward waypoints
    model.add_reward_source(RewardSource(
        name="tracking_yaw",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        requires_action=True,
        intentional=True,
        explore_fraction=0.2,
    ))
    # NOTE: feet_edge (clearance) penalty REMOVED in NoClear ablation
    # This is the ablation that causes the documented failure

    # Regularization penalties (all passive)
    model.add_reward_source(RewardSource(
        name="lin_vel_z",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="ang_vel_xy",
        reward_type=RewardType.PER_STEP,
        value=-0.05,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="dof_acc",
        reward_type=RewardType.PER_STEP,
        value=-2.5e-7,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="collision",
        reward_type=RewardType.PER_STEP,
        value=-10.0,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="action_rate",
        reward_type=RewardType.PER_STEP,
        value=-0.1,
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="torques",
        reward_type=RewardType.PER_STEP,
        value=-1e-5,
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="hip_pos",
        reward_type=RewardType.PER_STEP,
        value=-0.5,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="dof_error",
        reward_type=RewardType.PER_STEP,
        value=-0.04,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="feet_stumble",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
        requires_action=False,
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=2e-4,
        entropy_coeff=0.01,
        clip_epsilon=0.2,
        num_epochs=5,
        total_steps=50_000_000,
        rollout_length=24,
    )
    engine.print_report(model, config)

    print()
    print("Without the clearance penalty, energy-minimization penalties")
    print("(-10.0 collision, -0.5 hip_pos, -0.04 dof_error, -1e-5 torques)")
    print("incentivize short steps that land on terrain edges. This is")
    print("reward-optimal but physically unstable: 2.7x more edge violations")
    print("in simulation, 20-80% lower success rate on real hardware.")


if __name__ == "__main__":
    run_example()
