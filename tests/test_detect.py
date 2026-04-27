"""Test auto-detection from Gymnasium environments."""

import pytest


gymnasium = pytest.importorskip("gymnasium", reason="gymnasium not installed")


from goodhart.detect import detect_env
from goodhart.models import EnvironmentModel


def test_detect_cartpole():
    """Detect reward structure from CartPole-v1."""
    model, stats = detect_env("CartPole-v1", n_episodes=50)

    assert isinstance(model, EnvironmentModel)
    assert model.name == "CartPole-v1"

    # CartPole has max_episode_steps=500
    assert stats["max_episode_steps"] == 500

    # CartPole has 2 discrete actions
    assert stats["n_actions"] == 2

    # Stats should have expected keys
    assert "mean_episode_length" in stats
    assert "mean_reward" in stats
    assert "max_reward" in stats
    assert "min_reward" in stats
    assert "discovery_probability" in stats
    assert "death_probability" in stats

    # Random CartPole should have positive mean reward
    # (agent gets +1 per step)
    assert stats["mean_reward"] > 0

    # CartPole almost always terminates early with random policy
    assert stats["death_probability"] > 0.5

    # Model should have reward sources
    assert len(model.reward_sources) > 0


def test_detect_returns_tuple():
    """detect_env should return a 2-tuple."""
    result = detect_env("CartPole-v1", n_episodes=10)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_detect_bad_env():
    """detect_env should raise RuntimeError for invalid env."""
    with pytest.raises(RuntimeError, match="Could not create"):
        detect_env("NonExistentEnv-v999", n_episodes=5)


def test_detect_stats_types():
    """Stats values should be numeric."""
    _, stats = detect_env("CartPole-v1", n_episodes=10)
    assert isinstance(stats["mean_episode_length"], float)
    assert isinstance(stats["mean_reward"], float)
    assert isinstance(stats["max_reward"], float)
    assert isinstance(stats["min_reward"], float)
    assert isinstance(stats["discovery_probability"], float)
    assert isinstance(stats["death_probability"], float)
