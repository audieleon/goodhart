"""Tests for off-policy training rules (DQN, SAC, DDPG, TD3)."""

from goodhart.models import EnvironmentModel, RewardSource, RewardType, TrainingConfig
from goodhart.engine import TrainingAnalysisEngine


def _analyze(config, n_states=1000, max_steps=200):
    """Helper: run analysis with a minimal model and given config."""
    model = EnvironmentModel(name="test", max_steps=max_steps, n_states=n_states)
    model.add_reward_source(RewardSource("goal", RewardType.TERMINAL, 1.0,
                                         discovery_probability=0.1))
    engine = TrainingAnalysisEngine().add_all_rules()
    return engine.analyze(model, config)


def _rules_fired(result):
    """Return set of rule names that produced verdicts."""
    return {v.rule_name for v in result.verdicts}


# =====================================================================
# Replay Buffer Ratio
# =====================================================================

def test_replay_buffer_too_small():
    config = TrainingConfig(
        algorithm="DQN", lr=1e-3,
        replay_buffer_size=100,  # way too small for 200-step episodes
        total_steps=100_000,
    )
    result = _analyze(config)
    assert "replay_buffer_ratio" in _rules_fired(result)


def test_replay_buffer_adequate():
    config = TrainingConfig(
        algorithm="DQN", lr=1e-3,
        replay_buffer_size=100_000,
        total_steps=1_000_000,
    )
    result = _analyze(config)
    assert "replay_buffer_ratio" not in _rules_fired(result)


def test_replay_buffer_bigger_than_budget():
    config = TrainingConfig(
        algorithm="SAC", lr=3e-4,
        replay_buffer_size=10_000_000,
        total_steps=1_000_000,  # budget < buffer
    )
    result = _analyze(config)
    fired = [v for v in result.verdicts if v.rule_name == "replay_buffer_ratio"]
    assert any("smaller than replay buffer" in v.message for v in fired)


def test_replay_buffer_silent_on_ppo():
    """On-policy algorithms have no replay buffer — rule should be silent."""
    config = TrainingConfig(algorithm="PPO", lr=3e-4)
    result = _analyze(config)
    assert "replay_buffer_ratio" not in _rules_fired(result)


# =====================================================================
# Target Network Update
# =====================================================================

def test_target_update_too_frequent():
    config = TrainingConfig(
        algorithm="DQN", lr=1e-3,
        target_update_freq=10,  # too frequent
        total_steps=100_000,
    )
    result = _analyze(config)
    assert "target_network_update" in _rules_fired(result)


def test_target_update_adequate():
    config = TrainingConfig(
        algorithm="DQN", lr=1e-3,
        target_update_freq=1000,
        total_steps=1_000_000,
    )
    result = _analyze(config)
    assert "target_network_update" not in _rules_fired(result)


def test_target_update_too_rare():
    config = TrainingConfig(
        algorithm="DQN", lr=1e-3,
        target_update_freq=50_000,  # only 2 updates in 100K steps
        total_steps=100_000,
    )
    result = _analyze(config)
    fired = [v for v in result.verdicts if v.rule_name == "target_network_update"]
    assert any("Fewer than 10 updates" in v.message for v in fired)


def test_target_update_silent_without_target():
    """No target network configured — rule should be silent."""
    config = TrainingConfig(algorithm="PPO", lr=3e-4, target_update_freq=0)
    result = _analyze(config)
    assert "target_network_update" not in _rules_fired(result)


# =====================================================================
# Epsilon Schedule
# =====================================================================

def test_epsilon_decays_too_fast():
    config = TrainingConfig(
        algorithm="DQN", lr=1e-3,
        epsilon_start=1.0, epsilon_end=0.01,
        epsilon_decay_steps=1000,  # 1% of 100K budget
        total_steps=100_000,
    )
    result = _analyze(config)
    assert "epsilon_schedule" in _rules_fired(result)


def test_epsilon_decay_adequate():
    config = TrainingConfig(
        algorithm="DQN", lr=1e-3,
        epsilon_start=1.0, epsilon_end=0.01,
        epsilon_decay_steps=50_000,  # 50% of budget
        total_steps=100_000,
    )
    result = _analyze(config)
    fired = [v for v in result.verdicts if v.rule_name == "epsilon_schedule"
             and v.severity.value == "warning"]
    assert len(fired) == 0


def test_epsilon_end_zero():
    config = TrainingConfig(
        algorithm="DQN", lr=1e-3,
        epsilon_start=1.0, epsilon_end=0.0,  # no residual exploration
        epsilon_decay_steps=50_000,
        total_steps=100_000,
    )
    result = _analyze(config)
    fired = [v for v in result.verdicts if v.rule_name == "epsilon_schedule"]
    assert any("zero exploration" in v.message for v in fired)


def test_epsilon_silent_without_schedule():
    """No epsilon schedule configured — rule should be silent."""
    config = TrainingConfig(algorithm="PPO", lr=3e-4, epsilon_decay_steps=0)
    result = _analyze(config)
    assert "epsilon_schedule" not in _rules_fired(result)


# =====================================================================
# Soft Update Rate
# =====================================================================

def test_tau_too_high():
    config = TrainingConfig(algorithm="SAC", lr=3e-4, tau=0.5)
    result = _analyze(config)
    assert "soft_update_rate" in _rules_fired(result)


def test_tau_adequate():
    config = TrainingConfig(algorithm="SAC", lr=3e-4, tau=0.005)
    result = _analyze(config)
    assert "soft_update_rate" not in _rules_fired(result)


def test_tau_silent_on_ppo():
    config = TrainingConfig(algorithm="PPO", lr=3e-4, tau=0.5)
    result = _analyze(config)
    assert "soft_update_rate" not in _rules_fired(result)


def test_tau_fires_on_ddpg():
    config = TrainingConfig(algorithm="DDPG", lr=1e-3, tau=0.5)
    result = _analyze(config)
    assert "soft_update_rate" in _rules_fired(result)


def test_tau_fires_on_td3():
    config = TrainingConfig(algorithm="TD3", lr=3e-4, tau=0.5)
    result = _analyze(config)
    assert "soft_update_rate" in _rules_fired(result)


# =====================================================================
# SAC Alpha
# =====================================================================

def test_sac_alpha_too_high():
    config = TrainingConfig(algorithm="SAC", lr=3e-4, alpha=5.0, auto_alpha=False)
    result = _analyze(config)
    assert "sac_alpha" in _rules_fired(result)


def test_sac_alpha_too_low():
    config = TrainingConfig(algorithm="SAC", lr=3e-4, alpha=0.001, auto_alpha=False)
    result = _analyze(config)
    assert "sac_alpha" in _rules_fired(result)


def test_sac_alpha_adequate():
    config = TrainingConfig(algorithm="SAC", lr=3e-4, alpha=0.2, auto_alpha=False)
    result = _analyze(config)
    fired = [v for v in result.verdicts if v.rule_name == "sac_alpha"
             and v.severity.value == "warning"]
    assert len(fired) == 0


def test_sac_auto_alpha():
    config = TrainingConfig(algorithm="SAC", lr=3e-4, auto_alpha=True)
    result = _analyze(config)
    fired = [v for v in result.verdicts if v.rule_name == "sac_alpha"]
    # auto_alpha should produce an INFO, not a warning
    assert all(v.severity.value == "info" for v in fired)


def test_sac_alpha_silent_on_ppo():
    config = TrainingConfig(algorithm="PPO", lr=3e-4, alpha=5.0)
    result = _analyze(config)
    assert "sac_alpha" not in _rules_fired(result)


# =====================================================================
# LR Regime (algorithm-specific thresholds)
# =====================================================================

def test_lr_regime_dqn_high():
    config = TrainingConfig(algorithm="DQN", lr=0.01)
    result = _analyze(config)
    assert "lr_regime" in _rules_fired(result)


def test_lr_regime_sac_normal():
    config = TrainingConfig(algorithm="SAC", lr=3e-4)
    result = _analyze(config)
    assert "lr_regime" not in _rules_fired(result)


def test_lr_regime_ddpg_normal():
    config = TrainingConfig(algorithm="DDPG", lr=1e-3)
    result = _analyze(config)
    assert "lr_regime" not in _rules_fired(result)
