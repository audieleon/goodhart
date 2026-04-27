"""Test that all 11 known failures are caught.

7 published failures + 4 from our own experiments.
"""

import pytest
from goodhart.models import (
    EnvironmentModel, RewardSource, RewardType, RespawnBehavior,
    Severity, TrainingConfig,
)
from goodhart.engine import TrainingAnalysisEngine


def _run_analysis(model, config=None):
    engine = TrainingAnalysisEngine().add_all_rules()
    result = engine.analyze(model, config)
    return result.verdicts


def _has_critical(verdicts):
    return any(v.severity == Severity.CRITICAL for v in verdicts)


def _has_warning(verdicts):
    return any(v.severity == Severity.WARNING for v in verdicts)


# --- Published failures ---

def test_coast_runners_loop():
    """OpenAI 2016: agent loops green blocks instead of racing."""
    model = EnvironmentModel(name="CoastRunners", max_steps=1000)
    model.add_reward_source(RewardSource(
        name="finish", reward_type=RewardType.TERMINAL,
        value=10.0, discovery_probability=0.1,
    ))
    model.add_reward_source(RewardSource(
        name="blocks", reward_type=RewardType.ON_EVENT, value=0.5,
        respawn=RespawnBehavior.TIMED, respawn_time=2,
        can_loop=True, loop_period=3,
    ))
    verdicts = _run_analysis(model)
    assert _has_critical(verdicts), "Should catch respawning loop exploit"


def test_cartpole_suicide():
    """Classic: agent dies immediately to avoid step penalty."""
    model = EnvironmentModel(name="CartPole", max_steps=200)
    model.add_reward_source(RewardSource(
        name="penalty", reward_type=RewardType.PER_STEP, value=-0.1,
        requires_action=False,  # constant step penalty
    ))
    verdicts = _run_analysis(model)
    assert _has_critical(verdicts), "Should catch death beats survival"


def test_road_runner_replay():
    """Atari 2017: agent dies to replay level 1 collectibles."""
    model = EnvironmentModel(name="RoadRunner", max_steps=10000)
    model.add_reward_source(RewardSource(
        name="level", reward_type=RewardType.TERMINAL,
        value=100.0, discovery_probability=0.5,
    ))
    model.add_reward_source(RewardSource(
        name="collectibles", reward_type=RewardType.ON_EVENT,
        value=80.0, respawn=RespawnBehavior.ON_DEATH,
        discovery_probability=0.9,
    ))
    verdicts = _run_analysis(model)
    assert _has_critical(verdicts), "Should catch death reset exploit"


def test_bicycle_orbiting():
    """Weng 2024: agent orbits goal via shaping loop."""
    model = EnvironmentModel(name="Bicycle", max_steps=1000)
    model.add_reward_source(RewardSource(
        name="goal", reward_type=RewardType.TERMINAL,
        value=1.0, discovery_probability=0.05,
    ))
    model.add_reward_source(RewardSource(
        name="distance", reward_type=RewardType.SHAPING,
        value=0.1, can_loop=True, loop_period=4,
    ))
    verdicts = _run_analysis(model)
    assert _has_critical(verdicts), "Should catch shaping loop exploit"


def test_qbert_score_loop():
    """Atari 2018: Q*bert infinite score via respawning platforms."""
    model = EnvironmentModel(name="Q*bert", max_steps=50000)
    model.add_reward_source(RewardSource(
        name="level", reward_type=RewardType.TERMINAL,
        value=1000.0, discovery_probability=0.3,
    ))
    model.add_reward_source(RewardSource(
        name="platforms", reward_type=RewardType.ON_EVENT, value=25.0,
        respawn=RespawnBehavior.INFINITE, can_loop=True, loop_period=2,
    ))
    verdicts = _run_analysis(model)
    assert _has_critical(verdicts), "Should catch respawning exploit"


def test_walker2d_standing():
    """MuJoCo: agent stands still to collect alive bonus."""
    model = EnvironmentModel(name="Walker2d", max_steps=1000)
    model.add_reward_source(RewardSource(
        name="alive", reward_type=RewardType.PER_STEP, value=1.0,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="velocity", reward_type=RewardType.SHAPING, value=0.5,
    ))
    model.add_reward_source(RewardSource(
        name="energy", reward_type=RewardType.PER_STEP, value=-0.001,
    ))
    verdicts = _run_analysis(model)
    assert _has_critical(verdicts), "Should catch idle exploit"


def test_shazeer_expert_collapse():
    """Shazeer 2017: MoE without load balancing -> expert collapse."""
    model = EnvironmentModel(name="MoE", max_steps=500)
    model.add_reward_source(RewardSource(
        name="task", reward_type=RewardType.TERMINAL,
        value=1.0, discovery_probability=0.5,
    ))
    config = TrainingConfig(num_specialists=4, routing_floor=0.0)
    verdicts = _run_analysis(model, config)
    assert _has_critical(verdicts), "Should catch expert collapse"


# --- Our own failures ---

def test_multiroom_stand_still():
    """Scenario: agent stands still with action-conditional penalty."""
    model = EnvironmentModel(name="MultiRoom", max_steps=120)
    model.add_reward_source(RewardSource(
        name="goal", reward_type=RewardType.TERMINAL,
        value=1.0, discovery_probability=0.01,
    ))
    model.add_reward_source(RewardSource(
        name="penalty", reward_type=RewardType.PER_STEP,
        value=-0.01, requires_action=True,
    ))
    verdicts = _run_analysis(model)
    # With action-conditional penalty, idle exploit should fire
    assert _has_critical(verdicts), "Should catch stand-still incentive"


def test_multiroom_die_fast():
    """Scenario: agent dies fast with always-on penalty."""
    model = EnvironmentModel(name="MultiRoom", max_steps=120)
    model.add_reward_source(RewardSource(
        name="goal", reward_type=RewardType.TERMINAL,
        value=1.0, discovery_probability=0.01,
    ))
    model.add_reward_source(RewardSource(
        name="penalty", reward_type=RewardType.PER_STEP, value=-0.01,
    ))
    verdicts = _run_analysis(model)
    assert _has_critical(verdicts), "Should catch death beats survival"


def test_entry047_expert_collapse():
    """Scenario: 95% MLP, 0% Transformer on MiniHack."""
    model = EnvironmentModel(name="MiniHack", max_steps=120)
    config = TrainingConfig(
        num_specialists=3, routing_floor=0.0, balance_coef=0.0,
    )
    verdicts = _run_analysis(model, config)
    assert _has_critical(verdicts), "Should catch expert collapse"


def test_multiroom_reduced_penalty():
    """Scenario: reduced penalty still trapped."""
    model = EnvironmentModel(name="MultiRoom", max_steps=120, n_states=2000)
    model.add_reward_source(RewardSource(
        name="goal", reward_type=RewardType.TERMINAL,
        value=1.0, discovery_probability=0.01,
    ))
    model.add_reward_source(RewardSource(
        name="penalty", reward_type=RewardType.PER_STEP, value=-0.001,
    ))
    verdicts = _run_analysis(model)
    # Should catch that exploration threshold is not met
    has_issue = _has_critical(verdicts) or _has_warning(verdicts)
    assert has_issue, "Should catch exploration difficulty"
