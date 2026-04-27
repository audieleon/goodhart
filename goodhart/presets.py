"""Preset environment configurations for common RL domains.

Each preset provides a realistic EnvironmentModel AND TrainingConfig
with typical hyperparameters for that domain. All 24 rules run on
presets — nothing is silently skipped.

Usage:
    from goodhart.presets import PRESETS
    model, config = PRESETS["atari"]
    engine.analyze(model, config)

CLI:
    goodhart --preset atari
"""

from goodhart.models import (
    EnvironmentModel,
    RewardSource,
    RewardType,
    RespawnBehavior,
    TrainingConfig,
)


def _atari():
    """Atari PPO defaults from Schulman et al. 2017 + Engstrom et al. 2020
    ("Implementation Matters in Deep RL"). State space ~100K unique frames,
    18 actions (standard Atari), 18K steps (108K frames / 6 frame skip)."""
    model = EnvironmentModel(
        name="Atari (typical)",
        max_steps=18000,
        gamma=0.99,
        n_states=100000,
        n_actions=18,
        death_probability=0.05,
        wall_probability=0.1,
    )
    model.add_reward_source(RewardSource(
        name="score",
        reward_type=RewardType.ON_EVENT,
        value=1.0,
        respawn=RespawnBehavior.NONE,
        max_occurrences=0,
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.3,
    ))
    config = TrainingConfig(
        algorithm="PPO",
        lr=2.5e-4,
        entropy_coeff=0.01,
        num_epochs=4,
        clip_epsilon=0.1,
        num_envs=8,
        n_actors=8,
        total_steps=10_000_000,
        rollout_length=128,
        minibatch_size=256,
    )
    return model, config


def _mujoco_locomotion():
    """MuJoCo locomotion defaults from Schulman et al. 2017 (PPO paper,
    Table 3) and Gymnasium documentation. Walker2d/HalfCheetah/Hopper:
    1000 steps, gamma=0.99, alive bonus + velocity + control penalty."""
    model = EnvironmentModel(
        name="MuJoCo locomotion",
        max_steps=1000,
        gamma=0.99,
        n_states=10000,
        n_actions=8,
        death_probability=0.02,
        wall_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="forward velocity",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        value_range=(-2.0, 8.0),  # actual range depends on speed
        state_dependent=True,
        scales_with="velocity",
        respawn=RespawnBehavior.INFINITE,
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="alive bonus",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        respawn=RespawnBehavior.INFINITE,
        requires_action=False,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="control penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.001,
        respawn=RespawnBehavior.INFINITE,
        requires_action=True,
    ))
    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.0,
        num_epochs=10,
        clip_epsilon=0.2,
        num_envs=1,
        n_actors=1,
        total_steps=1_000_000,
        rollout_length=2048,
        minibatch_size=64,
    )
    return model, config


def _mujoco_manipulation():
    """MuJoCo manipulation task structure from Plappert et al. 2018.
    Environment parameters (gamma=0.98, sparse goal, distance shaping)
    from the HER paper. Original paper uses DDPG+HER (off-policy).
    Training hyperparameters from Andrychowicz et al. 2017."""
    model = EnvironmentModel(
        name="MuJoCo manipulation",
        max_steps=200,
        gamma=0.98,
        n_states=50000,
        n_actions=8,
        death_probability=0.0,
        wall_probability=0.1,
    )
    model.add_reward_source(RewardSource(
        name="goal reached",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.01,
    ))
    model.add_reward_source(RewardSource(
        name="distance shaping",
        reward_type=RewardType.SHAPING,
        value=0.1,
        requires_action=True,
    ))
    config = TrainingConfig(
        algorithm="DDPG",
        lr=1e-3,
        critic_lr=1e-3,
        num_envs=16,
        n_actors=16,
        total_steps=5_000_000,
        replay_buffer_size=1_000_000,
        tau=0.05,
    )
    return model, config


def _minihack_navigation():
    """MiniHack navigation from Samvelyan et al. 2021 Appendix D.2.
    Room-5x5, Room-15x15. Paper uses IMPALA via TorchBeast; we model
    as APPO (structurally similar). Values from polybeast/config.yaml:
    lr=2e-4, entropy=1e-4, 256 actors, gamma=0.999, penalty=-0.001.
    Note: paper's penalty_step=-0.001 is conditional (only when game
    timer doesn't advance, e.g., bumping walls). We model as constant
    per-step, which slightly overstates the penalty's impact."""
    model = EnvironmentModel(
        name="MiniHack navigation",
        max_steps=500,
        gamma=0.999,
        n_states=20000,
        n_actions=8,
        death_probability=0.05,
        wall_probability=0.3,
    )
    model.add_reward_source(RewardSource(
        name="goal",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.05,
    ))
    model.add_reward_source(RewardSource(
        name="step penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.001,
        respawn=RespawnBehavior.INFINITE,
    ))
    config = TrainingConfig(
        algorithm="APPO",
        lr=2e-4,
        entropy_coeff=1e-4,
        num_epochs=1,
        clip_epsilon=0.1,
        num_envs=256,
        num_workers=1,
        n_actors=256,
        total_steps=10_000_000,
        use_rnn=True,
        rnn_type="lstm",
        rnn_size=256,
    )
    return model, config


def _minihack_skill():
    """MiniHack skill tasks from Samvelyan et al. 2021 (Appendix D.3,
    footnote 12). CorridorBattle, Room-Ultimate: 256 actors, lr=5e-5
    (lower than navigation), entropy=1e-4, gamma=0.999, LSTM-256."""
    model = EnvironmentModel(
        name="MiniHack skill",
        max_steps=1000,
        gamma=0.999,
        n_states=50000,
        n_actions=16,
        death_probability=0.1,
        wall_probability=0.2,
    )
    model.add_reward_source(RewardSource(
        name="task completion",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.01,
    ))
    model.add_reward_source(RewardSource(
        name="step penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.001,
        respawn=RespawnBehavior.INFINITE,
    ))
    config = TrainingConfig(
        algorithm="APPO",
        lr=5e-5,
        entropy_coeff=1e-4,
        num_epochs=1,
        clip_epsilon=0.1,
        num_envs=256,
        num_workers=1,
        n_actors=256,
        total_steps=50_000_000,
        use_rnn=True,
        rnn_type="lstm",
        rnn_size=256,
    )
    return model, config


def _sparse_goal():
    """Sparse-reward navigation in the style of MiniGrid FourRooms
    (Chevalier-Boisvert et al. 2023). Terminal +1 goal, no step penalty,
    low discovery rate. PPO hyperparameters from Schulman et al. 2017
    (Table 3, MuJoCo defaults). Representative of sparse-reward tasks
    like Montezuma's Revenge (Bellemare et al. 2013) at smaller scale."""
    model = EnvironmentModel(
        name="Sparse goal",
        max_steps=500,
        gamma=0.99,
        n_states=10000,
        n_actions=4,
        death_probability=0.0,
        wall_probability=0.2,
    )
    model.add_reward_source(RewardSource(
        name="goal",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.02,
    ))
    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.01,
        num_envs=16,
        n_actors=16,
        total_steps=10_000_000,
    )
    return model, config


def _dense_survival():
    """Dense survival task combining the alive-bonus structure of
    MuJoCo locomotion (Brockman et al. 2016; Walker2d healthy_reward=1.0)
    with foraging mechanics inspired by Crafter (Hafner 2022). PPO
    hyperparameters from Schulman et al. 2017. Both reward sources
    marked intentional (survival IS the objective, not an exploit)."""
    model = EnvironmentModel(
        name="Dense survival",
        max_steps=2000,
        gamma=0.99,
        n_states=5000,
        n_actions=4,
        death_probability=0.1,
        wall_probability=0.1,
    )
    model.add_reward_source(RewardSource(
        name="alive bonus",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        respawn=RespawnBehavior.INFINITE,
        requires_action=False,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="food",
        reward_type=RewardType.ON_EVENT,
        value=5.0,
        respawn=RespawnBehavior.TIMED,
        respawn_time=50,
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.5,
        intentional=True,
    ))
    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.01,
        num_envs=16,
        n_actors=16,
        total_steps=5_000_000,
    )
    return model, config


def _coast_runners():
    """CoastRunners boat racing from Clark & Amodei 2016 (OpenAI blog).
    Agent goes in circles collecting turbo powerups (+20 each, respawn
    every ~50 steps) instead of finishing the race (+100). The loop
    reward dominates the goal. Tool should catch: respawning_exploit."""
    model = EnvironmentModel(
        name="CoastRunners (reward loop)",
        max_steps=2000,
        gamma=0.99,
        n_states=50000,
        n_actions=6,
        death_probability=0.01,
    )
    model.add_reward_source(RewardSource(
        name="finish race",
        reward_type=RewardType.TERMINAL,
        value=100.0,
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.1,
    ))
    model.add_reward_source(RewardSource(
        name="turbo powerup",
        reward_type=RewardType.ON_EVENT,
        value=20.0,
        respawn=RespawnBehavior.TIMED,
        respawn_time=50,
        requires_action=True,
        can_loop=True,
        loop_period=50,
    ))
    config = TrainingConfig(
        algorithm="PPO",
        lr=2.5e-4,
        entropy_coeff=0.01,
        num_envs=8,
        n_actors=8,
        total_steps=10_000_000,
    )
    return model, config


def _mountain_car():
    """Mountain Car from Moore 1990 / Gymnasium classic control.
    -1 reward per step, goal at +1.0. With 200 max steps, total
    penalty (-200) dwarfs goal (+1). Agent can't bootstrap learning.
    Tool should catch: penalty_dominates_goal, death_beats_survival."""
    model = EnvironmentModel(
        name="Mountain Car (step penalty trap)",
        max_steps=200,
        gamma=1.0,
        n_states=500,
        n_actions=3,
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="reach flag",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.01,
    ))
    model.add_reward_source(RewardSource(
        name="step penalty",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
    ))
    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.01,
        num_envs=16,
        n_actors=16,
        total_steps=1_000_000,
    )
    return model, config


def _bipedal_walker():
    """BipedalWalker-v3 from Brockman et al. 2016. -100 penalty for
    falling, small per-step reward for forward motion. The fall penalty
    is so large that agents learn to stand still or take tiny steps.
    Tool should catch: death_beats_survival (net negative when falling)."""
    model = EnvironmentModel(
        name="BipedalWalker (fall penalty)",
        max_steps=1600,
        gamma=0.99,
        n_states=10000,
        n_actions=4,
        death_probability=0.05,
    )
    model.add_reward_source(RewardSource(
        name="forward progress",
        reward_type=RewardType.PER_STEP,
        value=0.3,
        requires_action=True,
        intentional=True,  # locomotion IS the goal
        explore_fraction=0.5,  # random falling earns ~50% of velocity reward
    ))
    model.add_reward_source(RewardSource(
        name="fall penalty",
        reward_type=RewardType.ON_EVENT,
        value=-100.0,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="torque penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.003,
        requires_action=True,  # only applies when joints are actuated
    ))
    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.0,
        num_epochs=10,
        num_envs=1,
        n_actors=1,
        total_steps=2_000_000,
        rollout_length=2048,
    )
    return model, config


def _anymal():
    """ANYmal quadruped from Hwangbo et al. 2019 (Science Robotics,
    Table S2). 8 reward components including alive bonus, velocity
    tracking, torque penalty. The alive bonus creates a stand-still
    optimum. Tool should catch: idle_exploit potential."""
    model = EnvironmentModel(
        name="ANYmal (alive bonus vs locomotion)",
        max_steps=1000,
        gamma=0.99,
        n_states=50000,
        n_actions=12,
        death_probability=0.02,
    )
    model.add_reward_source(RewardSource(
        name="alive bonus",
        reward_type=RewardType.PER_STEP,
        value=0.2,
        requires_action=False,
        intentional=False,  # NOT intentional — locomotion is the goal, not standing
    ))
    model.add_reward_source(RewardSource(
        name="velocity tracking",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        value_range=(0.0, 2.0),  # exp(-error), capped
        state_dependent=True,
        scales_with="velocity_error",
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="torque penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.002,
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="joint velocity penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.001,
        requires_action=True,
    ))
    config = TrainingConfig(
        algorithm="PPO",
        lr=1e-3,
        entropy_coeff=0.01,
        num_epochs=5,
        clip_epsilon=0.2,
        num_envs=4096,
        n_actors=4096,
        total_steps=100_000_000,
        rollout_length=24,
        minibatch_size=24576,
    )
    return model, config


def _coinrun():
    """CoinRun from Cobbe et al. 2019 (ICML). Procedurally generated
    platformer. Sparse +10 for collecting coin. High discovery rate
    in training (coin always reachable), but goal misgeneralization
    in test (agent learns 'go right' not 'collect coin').
    Tool should catch: clean config (no structural reward traps)."""
    model = EnvironmentModel(
        name="CoinRun (goal misgeneralization)",
        max_steps=1000,
        gamma=0.999,
        n_states=100000,
        n_actions=15,
        death_probability=0.02,
    )
    model.add_reward_source(RewardSource(
        name="coin",
        reward_type=RewardType.TERMINAL,
        value=10.0,
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.8,
    ))
    config = TrainingConfig(
        algorithm="PPO",
        lr=5e-4,
        entropy_coeff=0.01,
        num_epochs=3,
        clip_epsilon=0.2,
        num_envs=8,
        n_actors=8,
        total_steps=200_000_000,
        rollout_length=256,
    )
    return model, config


def _hand_manipulation():
    """Shadow Hand in-hand manipulation from Andrychowicz et al. 2020
    (IJRR, OpenAI). Rotation-distance reward d_t - d_{t+1} is
    potential-based (Phi=-d), so cycles net zero (can_loop=False).
    384 workers x 16 envs, zero entropy, 29B total steps.
    Tool should catch: idle exploit (fingertip contact), entropy warning."""
    model = EnvironmentModel(
        name="Shadow Hand (shaping loop risk)",
        max_steps=100,
        gamma=0.998,
        n_states=500000,
        n_actions=20,
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="rotation achieved",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.05,
    ))
    model.add_reward_source(RewardSource(
        name="rotation distance shaping",
        reward_type=RewardType.SHAPING,
        value=0.5,
        requires_action=False,  # d_t - d_{t+1} depends on state, not action
        can_loop=False,  # potential-based: cycles net zero by Ng 1999
    ))
    model.add_reward_source(RewardSource(
        name="fingertip contact bonus",
        reward_type=RewardType.PER_STEP,
        value=0.1,
        requires_action=False,
        intentional=False,  # contact is a means, not the goal
    ))
    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.0,
        num_epochs=5,
        clip_epsilon=0.2,
        num_envs=6144,
        n_actors=6144,
        total_steps=29_000_000_000,
        rollout_length=10,
    )
    return model, config


def _legged_gym():
    """Legged Gym ANYmal from Rudin et al. 2022 (RSS). 19 reward terms.
    Massively parallel PPO in Isaac Gym. Known issues: alive-bonus idle
    exploit, exponential velocity tracking saturation, penalty imbalance."""
    model = EnvironmentModel(
        name="Legged Gym ANYmal (19 reward terms)",
        max_steps=1000,
        gamma=0.99,
        n_states=100000,
        n_actions=12,
        death_probability=0.02,
    )
    model.add_reward_source(RewardSource(
        name="tracking_lin_vel",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        value_type="exponential",
        value_params={"sigma": 0.25},
        value_range=(0.0, 1.0),
        state_dependent=True,
        scales_with="velocity_error",
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="tracking_ang_vel",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        value_type="exponential",
        value_params={"sigma": 0.25},
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="feet_air_time",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=False,
        intentional=False,  # gait incentive, not the goal
    ))
    model.add_reward_source(RewardSource(
        name="lin_vel_z_penalty",
        reward_type=RewardType.PER_STEP,
        value=-2.0,
        state_dependent=True,
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="ang_vel_xy_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.05,
        state_dependent=True,
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="torques_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.00001,
        state_dependent=True,
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="dof_acc_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.0000025,
        state_dependent=True,
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="action_rate_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.01,
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="collision_penalty",
        reward_type=RewardType.ON_EVENT,
        value=-1.0,
        requires_action=False,
    ))
    config = TrainingConfig(
        algorithm="PPO",
        lr=1e-3,
        entropy_coeff=0.01,
        num_epochs=5,
        clip_epsilon=0.2,
        num_envs=4096,
        n_actors=4096,
        total_steps=100_000_000,
        rollout_length=24,
        minibatch_size=24576,
    )
    return model, config


def _robosuite_pick_place():
    """Robosuite Pick-and-Place from Zhu et al. 2020 (CoRL).
    4 staged rewards: grasp → lift → hover → place. Prerequisite
    gates create learning plateaus between stages."""
    model = EnvironmentModel(
        name="Robosuite Pick-and-Place (staged rewards)",
        max_steps=200,
        gamma=0.99,
        n_states=50000,
        n_actions=8,
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="grasp",
        reward_type=RewardType.ON_EVENT,
        value=0.35,
        requires_action=True,
        discovery_probability=0.3,
    ))
    model.add_reward_source(RewardSource(
        name="lift",
        reward_type=RewardType.ON_EVENT,
        value=0.15,
        requires_action=True,
        prerequisite="grasp",
        discovery_probability=0.1,
    ))
    model.add_reward_source(RewardSource(
        name="hover",
        reward_type=RewardType.ON_EVENT,
        value=0.2,
        requires_action=True,
        prerequisite="lift",
        discovery_probability=0.05,
    ))
    model.add_reward_source(RewardSource(
        name="place",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        requires_action=True,
        prerequisite="hover",
        discovery_probability=0.01,
    ))
    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.01,
        num_epochs=10,
        num_envs=16,
        n_actors=16,
        total_steps=5_000_000,
    )
    return model, config


def _metadrive():
    """MetaDrive autonomous driving from Li et al. 2021. Progress reward
    + speed bonus + collision penalties. The lateral factor modifier
    scales the driving reward by lane position."""
    model = EnvironmentModel(
        name="MetaDrive (driving + safety)",
        max_steps=1000,
        gamma=0.99,
        n_states=100000,
        n_actions=2,
        death_probability=0.01,
    )
    model.add_reward_source(RewardSource(
        name="driving_progress",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        value_type="proportional",
        state_dependent=True,
        scales_with="distance",
        requires_action=True,
        intentional=True,
        explore_fraction=0.3,
    ))
    model.add_reward_source(RewardSource(
        name="lateral_factor",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        modifies="driving_progress",
        modifier_type="multiplicative",
        state_dependent=True,
    ))
    model.add_reward_source(RewardSource(
        name="speed_reward",
        reward_type=RewardType.PER_STEP,
        value=0.1,
        state_dependent=True,
        scales_with="speed",
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="success",
        reward_type=RewardType.TERMINAL,
        value=10.0,
        discovery_probability=0.2,
    ))
    model.add_reward_source(RewardSource(
        name="crash_penalty",
        reward_type=RewardType.ON_EVENT,
        value=-5.0,
        requires_action=False,
    ))
    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.01,
        num_envs=16,
        n_actors=16,
        total_steps=10_000_000,
    )
    return model, config


def _smac():
    """SMAC StarCraft Micromanagement from Samvelyan et al. 2019.
    Kill/death events + proportional damage. Asymmetric neg_scale
    (0.5x) means killing is valued 2x over protecting allies."""
    model = EnvironmentModel(
        name="SMAC 3m (StarCraft micro)",
        max_steps=120,
        gamma=0.99,
        n_states=50000,
        n_actions=11,
        death_probability=0.1,
    )
    model.add_reward_source(RewardSource(
        name="enemy_killed",
        reward_type=RewardType.ON_EVENT,
        value=10.0,
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="ally_killed",
        reward_type=RewardType.ON_EVENT,
        value=-5.0,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="damage_dealt",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        state_dependent=True,
        scales_with="damage",
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="damage_received",
        reward_type=RewardType.PER_STEP,
        value=-0.25,
        state_dependent=True,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="win_bonus",
        reward_type=RewardType.TERMINAL,
        value=200.0,
        discovery_probability=0.5,
    ))
    config = TrainingConfig(
        algorithm="PPO",
        lr=5e-4,
        entropy_coeff=0.01,
        num_epochs=5,
        num_envs=8,
        n_actors=8,
        total_steps=10_000_000,
    )
    return model, config


def _highway_env():
    """highway-env from Leurent 2018. Default agent uses DQN.
    Simple reward components but the ratio of collision (-1.0) to
    speed (+0.4) means 2.5 steps of driving offsets a crash."""
    model = EnvironmentModel(
        name="highway-env (driving)",
        max_steps=40,
        gamma=0.8,
        n_states=1000,
        n_actions=5,
        death_probability=0.05,
    )
    model.add_reward_source(RewardSource(
        name="collision",
        reward_type=RewardType.ON_EVENT,
        value=-1.0,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="high_speed",
        reward_type=RewardType.PER_STEP,
        value=0.4,
        state_dependent=True,
        scales_with="speed",
        requires_action=True,
        intentional=True,
        explore_fraction=0.5,
    ))
    model.add_reward_source(RewardSource(
        name="right_lane",
        reward_type=RewardType.PER_STEP,
        value=0.1,
        state_dependent=True,
        requires_action=True,
    ))
    config = TrainingConfig(
        algorithm="DQN",
        lr=5e-4,
        entropy_coeff=0.01,
        num_envs=4,
        n_actors=4,
        total_steps=100_000,
        replay_buffer_size=15000,
        target_update_freq=50,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=10000,
    )
    return model, config


def _humanoid():
    """MuJoCo Humanoid-v4 from Todorov et al. 2012. healthy_reward=5.0
    per step dominates velocity reward (~1.25/step average). Known idle
    exploit: agent learns to stand still. Towers et al. 2023 (Gymnasium)
    document this as a common pitfall."""
    model = EnvironmentModel(
        name="Humanoid-v4 (idle exploit)",
        max_steps=1000,
        gamma=0.99,
        n_states=100000,
        n_actions=17,
        death_probability=0.03,
    )
    model.add_reward_source(RewardSource(
        name="healthy_reward",
        reward_type=RewardType.PER_STEP,
        value=5.0,
        respawn=RespawnBehavior.INFINITE,
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="forward_velocity",
        reward_type=RewardType.PER_STEP,
        value=1.25,
        value_range=(-1.0, 3.0),
        state_dependent=True,
        scales_with="velocity",
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="ctrl_cost",
        reward_type=RewardType.PER_STEP,
        value=-0.1,
        requires_action=True,
    ))
    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.0,
        num_epochs=10,
        num_envs=1,
        n_actors=1,
        total_steps=1_000_000,
        rollout_length=2048,
        minibatch_size=64,
    )
    return model, config


def _lunar_lander():
    """LunarLander-v2 from Brockman et al. 2016. Potential-based shaping
    (distance + velocity decrease). By Ng 1999, this preserves optimal
    policy. Landing bonus +100, crash penalty -100. Should pass clean."""
    model = EnvironmentModel(
        name="LunarLander-v2",
        max_steps=1000,
        gamma=0.99,
        n_states=10000,
        n_actions=4,
        death_probability=0.1,
    )
    model.add_reward_source(RewardSource(
        name="distance_shaping",
        reward_type=RewardType.SHAPING,
        value=1.0,
        value_range=(-1.0, 1.0),
        state_dependent=True,
        scales_with="distance",
        requires_action=True,
        can_loop=False,
    ))
    model.add_reward_source(RewardSource(
        name="landing_bonus",
        reward_type=RewardType.TERMINAL,
        value=100.0,
        requires_action=True,
        discovery_probability=0.3,
    ))
    model.add_reward_source(RewardSource(
        name="crash_penalty",
        reward_type=RewardType.ON_EVENT,
        value=-100.0,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="fuel_cost",
        reward_type=RewardType.PER_STEP,
        value=-0.03,
        requires_action=True,
    ))
    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.01,
        num_envs=16,
        n_actors=16,
        total_steps=1_000_000,
    )
    return model, config


def _taxi():
    """Taxi-v3 from Dietterich 2000 (MAXQ). Step penalty -1/step with
    +20 dropoff goal. Total penalty (-200) is 10x goal (+20). Classic
    penalty-dominates-goal structure. Tabular RL solves it easily but
    function approximation struggles."""
    model = EnvironmentModel(
        name="Taxi-v3 (penalty trap)",
        max_steps=200,
        gamma=0.99,
        n_states=500,
        n_actions=6,
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="step_penalty",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
    ))
    model.add_reward_source(RewardSource(
        name="dropoff_success",
        reward_type=RewardType.TERMINAL,
        value=20.0,
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.1,
    ))
    model.add_reward_source(RewardSource(
        name="illegal_action",
        reward_type=RewardType.ON_EVENT,
        value=-10.0,
        requires_action=True,
    ))
    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.01,
        num_envs=16,
        n_actors=16,
        total_steps=500_000,
    )
    return model, config


def _football():
    """Google Research Football from Kurach et al. 2020 (ICML). Sparse
    +1 goal with checkpoint shaping (+0.1 each, 10 zones). APPO with
    128 actors, 500M total steps. Checkpoints are one-time per episode."""
    model = EnvironmentModel(
        name="Google Research Football",
        max_steps=3000,
        gamma=0.997,
        n_states=100000,
        n_actions=19,
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="goal_scored",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.1,
    ))
    model.add_reward_source(RewardSource(
        name="checkpoint_shaping",
        reward_type=RewardType.ON_EVENT,
        value=0.1,
        max_occurrences=10,
        respawn=RespawnBehavior.NONE,
        requires_action=True,
        can_loop=False,
    ))
    config = TrainingConfig(
        algorithm="APPO",
        lr=2.5e-4,
        entropy_coeff=0.003,
        num_epochs=2,
        clip_epsilon=0.115,
        num_envs=16,
        num_workers=8,
        n_actors=128,
        total_steps=500_000_000,
        rollout_length=512,
        minibatch_size=4096,
    )
    return model, config


def _cartpole():
    """CartPole-v1 from Barto et al. 1983. +1/step for balancing.
    Well-designed: survival IS the objective (intentional=True).
    Should pass clean — no structural reward traps."""
    model = EnvironmentModel(
        name="CartPole-v1",
        max_steps=500,
        gamma=0.99,
        n_states=500,
        n_actions=2,
        death_probability=0.05,
    )
    model.add_reward_source(RewardSource(
        name="alive_reward",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        respawn=RespawnBehavior.INFINITE,
        requires_action=False,
        intentional=True,
    ))
    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.01,
        num_envs=8,
        n_actors=8,
        total_steps=100_000,
    )
    return model, config


PRESETS = {
    # Standard benchmarks
    "atari": _atari(),
    "mujoco-locomotion": _mujoco_locomotion(),
    "mujoco-manipulation": _mujoco_manipulation(),
    "minihack-navigation": _minihack_navigation(),
    "minihack-skill": _minihack_skill(),
    # Reward structure archetypes
    "sparse-goal": _sparse_goal(),
    "dense-survival": _dense_survival(),
    # Documented reward failures
    "coast-runners": _coast_runners(),
    "mountain-car": _mountain_car(),
    "bipedal-walker": _bipedal_walker(),
    "anymal": _anymal(),
    "coinrun": _coinrun(),
    "hand-manipulation": _hand_manipulation(),
    # Complex multi-component rewards
    "legged-gym": _legged_gym(),
    "robosuite-pick-place": _robosuite_pick_place(),
    "metadrive": _metadrive(),
    "smac": _smac(),
    "highway-env": _highway_env(),
    # Classic control & locomotion
    "humanoid": _humanoid(),
    "lunar-lander": _lunar_lander(),
    "taxi": _taxi(),
    "cartpole": _cartpole(),
    # Multi-agent & game AI
    "football": _football(),
}
