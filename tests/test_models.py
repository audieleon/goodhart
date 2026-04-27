"""Test input validation on models."""

import pytest
from goodhart.models import (
    EnvironmentModel, RewardSource, RewardType, RespawnBehavior,
    TrainingConfig, Severity, Verdict, Result,
)


def test_environment_model_valid():
    model = EnvironmentModel(name="test", max_steps=100, gamma=0.99)
    assert model.name == "test"
    assert model.max_steps == 100


def test_environment_model_invalid_gamma():
    with pytest.raises(ValueError, match="gamma"):
        EnvironmentModel(name="test", gamma=1.5)


def test_environment_model_invalid_gamma_negative():
    with pytest.raises(ValueError, match="gamma"):
        EnvironmentModel(name="test", gamma=-0.1)


def test_environment_model_invalid_max_steps():
    with pytest.raises(ValueError, match="max_steps"):
        EnvironmentModel(name="test", max_steps=0)


def test_environment_model_invalid_n_states():
    with pytest.raises(ValueError, match="n_states"):
        EnvironmentModel(name="test", n_states=-1)


def test_reward_source_valid():
    src = RewardSource(name="r", reward_type=RewardType.TERMINAL, value=1.0)
    assert src.name == "r"


def test_reward_source_invalid_discovery_probability():
    with pytest.raises(ValueError, match="discovery_probability"):
        RewardSource(name="r", reward_type=RewardType.TERMINAL,
                     value=1.0, discovery_probability=1.5)


def test_reward_source_invalid_max_occurrences():
    with pytest.raises(ValueError, match="max_occurrences"):
        RewardSource(name="r", reward_type=RewardType.TERMINAL,
                     value=1.0, max_occurrences=-1)


def test_reward_source_invalid_loop_period():
    with pytest.raises(ValueError, match="loop_period"):
        RewardSource(name="r", reward_type=RewardType.TERMINAL,
                     value=1.0, loop_period=-1)


def test_environment_model_properties():
    model = EnvironmentModel(name="test")
    model.add_reward_source(RewardSource(
        name="goal", reward_type=RewardType.TERMINAL, value=10.0,
    ))
    model.add_reward_source(RewardSource(
        name="penalty", reward_type=RewardType.PER_STEP, value=-0.01,
    ))
    assert len(model.goal_sources) == 1
    assert len(model.penalty_sources) == 1
    assert model.max_goal_reward == 10.0
    assert model.total_step_penalty == -0.01


def test_verdict_str():
    v = Verdict(rule_name="test", severity=Severity.CRITICAL,
                message="bad thing", recommendation="fix it")
    s = str(v)
    assert "[test]" in s
    assert "bad thing" in s
    assert "fix it" in s


def test_training_config_defaults():
    config = TrainingConfig()
    assert config.algorithm == "PPO"
    assert config.lr == 3e-4
    assert config.num_specialists == 1


# --- Result tests ---

def test_result_to_dict():
    verdicts = [
        Verdict(rule_name="r1", severity=Severity.CRITICAL,
                message="bad", recommendation="fix it"),
        Verdict(rule_name="r2", severity=Severity.WARNING,
                message="careful", recommendation="watch out"),
        Verdict(rule_name="r3", severity=Severity.INFO,
                message="fyi"),
    ]
    result = Result(verdicts=verdicts, passed=False)
    d = result.to_dict()
    assert d["passed"] is False
    assert len(d["criticals"]) == 1
    assert d["criticals"][0]["rule"] == "r1"
    assert d["criticals"][0]["recommendation"] == "fix it"
    assert len(d["warnings"]) == 1
    assert d["warnings"][0]["rule"] == "r2"
    assert len(d["infos"]) == 1
    assert d["infos"][0]["rule"] == "r3"


def test_result_properties():
    verdicts = [
        Verdict(rule_name="a", severity=Severity.CRITICAL, message="x"),
        Verdict(rule_name="b", severity=Severity.WARNING, message="y"),
        Verdict(rule_name="c", severity=Severity.INFO, message="z"),
    ]
    result = Result(verdicts=verdicts, passed=False)
    assert result.has_criticals
    assert result.has_warnings
    assert len(result.criticals) == 1
    assert len(result.warnings) == 1
    assert len(result.infos) == 1


def test_result_passed_no_criticals():
    verdicts = [
        Verdict(rule_name="b", severity=Severity.WARNING, message="y"),
    ]
    result = Result(verdicts=verdicts, passed=True)
    assert result.passed
    assert not result.has_criticals
    assert result.has_warnings


def test_result_to_dict_empty():
    result = Result(verdicts=[], passed=True)
    d = result.to_dict()
    assert d["passed"] is True
    assert d["criticals"] == []
    assert d["warnings"] == []
    assert d["infos"] == []
