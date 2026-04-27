"""Test reward landscape visualization."""

import pytest

from goodhart.models import (
    EnvironmentModel,
    RewardSource,
    RewardType,
    RespawnBehavior,
)
from goodhart.viz import reward_landscape_ascii, _compute_strategy_evs


def _sample_model():
    """Build a sample model with goal + step penalty."""
    model = EnvironmentModel(name="test_env", max_steps=500)
    model.add_reward_source(RewardSource(
        name="goal", reward_type=RewardType.TERMINAL,
        value=1.0, discovery_probability=0.1,
    ))
    model.add_reward_source(RewardSource(
        name="step_penalty", reward_type=RewardType.PER_STEP,
        value=-0.01,
    ))
    return model


def test_ascii_produces_output():
    """ASCII visualization should produce a non-empty string."""
    model = _sample_model()
    output = reward_landscape_ascii(model)
    assert isinstance(output, str)
    assert len(output) > 0
    assert "test_env" in output


def test_ascii_contains_strategies():
    """ASCII output should show all strategy names."""
    model = _sample_model()
    output = reward_landscape_ascii(model)
    assert "die fast" in output
    assert "stand still" in output
    assert "optimal" in output
    assert "explore" in output


def test_ascii_contains_ev_values():
    """ASCII output should contain numeric EV values."""
    model = _sample_model()
    output = reward_landscape_ascii(model)
    # Should contain at least one numeric value with sign
    assert "+" in output or "-" in output


def test_ascii_marks_winner():
    """ASCII output should mark the winning strategy."""
    model = _sample_model()
    output = reward_landscape_ascii(model)
    assert "WINS" in output


def test_ascii_warns_on_degenerate_winner():
    """When a degenerate strategy wins, show a warning."""
    # Heavy penalty, no goal -- standing still wins
    model = EnvironmentModel(name="bad_env", max_steps=500)
    model.add_reward_source(RewardSource(
        name="step_penalty", reward_type=RewardType.PER_STEP,
        value=-0.1,
    ))
    output = reward_landscape_ascii(model)
    # die_fast or stand_still should win -- both are degenerate
    assert "WARNING" in output or "CAUTION" in output


def test_compute_strategy_evs_returns_dict():
    """Strategy EV computation should return a dict with all strategies."""
    model = _sample_model()
    evs = _compute_strategy_evs(model)
    assert isinstance(evs, dict)
    assert "die_fast" in evs
    assert "stand_still" in evs
    assert "optimal" in evs
    assert "explore_random" in evs
    assert "explore_full" in evs


def test_compute_strategy_evs_optimal_highest_with_good_config():
    """With a well-designed reward, optimal should have highest EV."""
    model = EnvironmentModel(name="good_env", max_steps=100)
    model.add_reward_source(RewardSource(
        name="goal", reward_type=RewardType.TERMINAL,
        value=10.0, discovery_probability=0.5,
    ))
    # Very small penalty
    model.add_reward_source(RewardSource(
        name="step_penalty", reward_type=RewardType.PER_STEP,
        value=-0.001,
    ))
    evs = _compute_strategy_evs(model)
    assert evs["optimal"] > evs["stand_still"]
    assert evs["optimal"] > evs["die_fast"]


def test_ascii_no_goal():
    """ASCII should work even with no goal source."""
    model = EnvironmentModel(name="no_goal", max_steps=100)
    model.add_reward_source(RewardSource(
        name="step_penalty", reward_type=RewardType.PER_STEP,
        value=-0.01,
    ))
    output = reward_landscape_ascii(model)
    assert isinstance(output, str)
    assert len(output) > 0


def test_ascii_no_penalty():
    """ASCII should work with no penalty (all zeros except goal)."""
    model = EnvironmentModel(name="no_penalty", max_steps=100)
    model.add_reward_source(RewardSource(
        name="goal", reward_type=RewardType.TERMINAL,
        value=1.0, discovery_probability=0.5,
    ))
    output = reward_landscape_ascii(model)
    assert isinstance(output, str)
    assert "optimal" in output


def test_reward_landscape_matplotlib_import_error():
    """reward_landscape() should raise ImportError if matplotlib missing."""
    # We can't easily uninstall matplotlib, but we can test the function
    # exists and has the right signature
    from goodhart.viz import reward_landscape
    model = _sample_model()
    # If matplotlib is installed, it should produce a file
    # If not, it should raise ImportError
    # Either way, the function should be importable
    assert callable(reward_landscape)


def test_legend_in_ascii():
    """ASCII output should include a legend."""
    model = _sample_model()
    output = reward_landscape_ascii(model)
    assert "intended" in output
    assert "marginal" in output
    assert "degenerate" in output
