"""Tests for the @reward_function decorator."""

import pytest
from goodhart import reward_function, analyze_function, RewardSource, RewardType, RespawnBehavior


@reward_function(
    max_steps=500, gamma=0.99, n_states=1000, n_actions=4,
    sources=[
        RewardSource("alive", RewardType.PER_STEP, 1.0,
                     requires_action=False, intentional=True),
    ],
)
def simple_reward(obs, action, info):
    return 1.0


@reward_function(
    max_steps=1000, gamma=0.99, n_states=100000, n_actions=17,
    sources=[
        RewardSource("healthy", RewardType.PER_STEP, 5.0,
                     respawn=RespawnBehavior.INFINITE,
                     requires_action=False, intentional=False),
        RewardSource("velocity", RewardType.PER_STEP, 1.25,
                     requires_action=True, intentional=True),
    ],
    lr=3e-4, num_envs=1, total_steps=1_000_000,
)
def idle_exploit_reward(obs, action, info):
    return 5.0 + 1.25


def test_decorator_preserves_function():
    """The decorated function should work normally."""
    assert simple_reward({}, None, {}) == 1.0


def test_decorator_attaches_model():
    """The decorator should attach goodhart_model."""
    assert hasattr(simple_reward, 'goodhart_model')
    assert simple_reward.goodhart_model.name == "simple_reward"
    assert simple_reward.goodhart_model.max_steps == 500


def test_decorator_attaches_config():
    """Training config is attached when lr is provided."""
    assert simple_reward.goodhart_config is None
    assert idle_exploit_reward.goodhart_config is not None
    assert idle_exploit_reward.goodhart_config.lr == 3e-4


def test_goodhart_passed():
    """Simple intentional survival should pass."""
    assert simple_reward.goodhart_passed() is True


def test_goodhart_failed():
    """Humanoid-style idle exploit should fail."""
    assert idle_exploit_reward.goodhart_passed() is False


def test_goodhart_check_returns_result():
    """goodhart_check should return a Result object."""
    result = simple_reward.goodhart_check(quiet=True)
    assert hasattr(result, 'passed')
    assert hasattr(result, 'criticals')
    assert hasattr(result, 'warnings')


def test_analyze_function():
    """analyze_function should work on decorated functions."""
    result = analyze_function(simple_reward)
    assert result.passed is True


def test_analyze_function_undecorated():
    """analyze_function should raise on undecorated functions."""
    def plain_fn():
        return 1.0
    with pytest.raises(AttributeError, match="not decorated"):
        analyze_function(plain_fn)


def test_sources_attached():
    """Sources should be accessible."""
    assert len(simple_reward.goodhart_sources) == 1
    assert simple_reward.goodhart_sources[0].name == "alive"


def test_variable_driven_decorator():
    """Constants can drive both decorator and function body."""
    BONUS = 2.0

    @reward_function(
        max_steps=100, n_states=100, n_actions=2,
        sources=[RewardSource("bonus", RewardType.PER_STEP, BONUS,
                              requires_action=False, intentional=True)],
    )
    def var_reward(obs, action, info):
        return BONUS

    assert var_reward({}, None, {}) == BONUS
    assert var_reward.goodhart_model.reward_sources[0].value == BONUS
