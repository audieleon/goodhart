"""Example: Extreme Parkour — held-out evaluation.

Held-out paper not used during tool development. Tests whether
the tool detects the energy-minimization exploit documented in
the NoClear ablation: without a clearance penalty, the robot
places feet on terrain edges to minimize energy, which is
physically unstable and causes falls in the real world.

Source: Cheng et al. 2024, "Extreme Parkour with Legged Robots"
  (ICRA 2024, arXiv:2309.14341). Weights from codebase:
  github.com/chengxuxin/extreme-parkour

Expected result: WARN or higher. The reward structure has a
  dominant collision penalty (-10.0) vs small goal reward (1.5),
  and numerous passive regularization penalties. The NoClear
  ablation shows removing the clearance penalty leads to
  edge-stepping exploit (energy minimization dominates safety).
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


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
