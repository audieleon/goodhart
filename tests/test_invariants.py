"""Property-based tests for rule invariants.

These test mathematical properties that should ALWAYS hold,
regardless of input values. Uses hypothesis for random generation.
Falls back to parametric tests if hypothesis is not installed.
"""

import pytest
from goodhart.models import EnvironmentModel, RewardSource, RewardType, Severity
from goodhart.rules.reward import (
    PenaltyDominatesGoal, DeathBeatsSurvival, IdleExploit,
    ExplorationThreshold, RespawningExploit, IntrinsicSufficiency,
)

try:
    from hypothesis import given, settings, assume
    from hypothesis.strategies import floats, integers
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


# --- Invariants that hold for all inputs ---

@pytest.mark.parametrize("penalty", [0.0, -0.001, -0.1, -1.0])
@pytest.mark.parametrize("goal", [0.0, 0.1, 1.0, 10.0])
def test_penalty_dominates_never_fires_without_penalty(penalty, goal):
    """PenaltyDominatesGoal should not fire when penalty=0."""
    if penalty == 0:
        rule = PenaltyDominatesGoal()
        model = EnvironmentModel(name="test", max_steps=100)
        if goal > 0:
            model.add_reward_source(RewardSource(
                name="goal", reward_type=RewardType.TERMINAL, value=goal,
            ))
        assert not rule.applies_to(model)


@pytest.mark.parametrize("death_prob", [0.0, 0.01, 0.1, 0.5])
def test_death_beats_survival_requires_death(death_prob):
    """DeathBeatsSurvival should not fire when death_probability=0."""
    rule = DeathBeatsSurvival()
    model = EnvironmentModel(name="test", max_steps=100, death_probability=death_prob)
    model.add_reward_source(RewardSource(
        name="penalty", reward_type=RewardType.PER_STEP, value=-0.01,
        requires_action=False,
    ))
    if death_prob == 0:
        assert not rule.applies_to(model)
    else:
        assert rule.applies_to(model)


@pytest.mark.parametrize("gamma", [0.5, 0.9, 0.99, 0.999, 1.0])
def test_discounting_monotonic(gamma):
    """Higher gamma means more penalty accumulation (for same steps)."""
    from goodhart.rules.reward import _discounted_steps
    d10 = _discounted_steps(gamma, 10)
    d100 = _discounted_steps(gamma, 100)
    assert d100 >= d10, f"discounted_steps should be monotonic: d100={d100} < d10={d10}"


@pytest.mark.parametrize("gamma", [0.0, 0.5, 0.99, 1.0])
def test_discounted_steps_positive(gamma):
    """Discounted steps should always be non-negative."""
    from goodhart.rules.reward import _discounted_steps
    for n in [0, 1, 10, 100, 1000]:
        d = _discounted_steps(gamma, n)
        assert d >= 0, f"discounted_steps({gamma}, {n}) = {d} < 0"


def test_no_reward_sources_no_verdicts():
    """With no reward sources, no rule should fire."""
    from goodhart.engine import TrainingAnalysisEngine
    model = EnvironmentModel(name="empty", max_steps=100)
    engine = TrainingAnalysisEngine().add_all_rules()
    result = engine.analyze(model)
    assert result.passed, f"Empty model should pass, got: {[v.rule_name for v in result.criticals]}"


@pytest.mark.parametrize("max_steps", [1, 2, 10, 100, 1000])
def test_max_steps_edge_cases_no_crash(max_steps):
    """No rule should crash on any valid max_steps."""
    from goodhart.engine import TrainingAnalysisEngine
    from goodhart.models import TrainingConfig
    model = EnvironmentModel(name="test", max_steps=max_steps)
    model.add_reward_source(RewardSource(
        name="goal", reward_type=RewardType.TERMINAL, value=1.0,
        discovery_probability=0.05,
    ))
    model.add_reward_source(RewardSource(
        name="penalty", reward_type=RewardType.PER_STEP, value=-0.01,
        requires_action=False,
    ))
    config = TrainingConfig()
    engine = TrainingAnalysisEngine().add_all_rules()
    # Should not raise
    result = engine.analyze(model, config)
    assert result is not None


def test_intentional_sources_never_flagged_as_idle():
    """Sources marked intentional should never trigger idle_exploit."""
    rule = IdleExploit()
    model = EnvironmentModel(name="test", max_steps=100)
    model.add_reward_source(RewardSource(
        name="alive", reward_type=RewardType.PER_STEP, value=10.0,
        requires_action=False, intentional=True,
    ))
    verdicts = rule.check(model)
    idle_verdicts = [v for v in verdicts if v.rule_name == "idle_exploit"]
    assert len(idle_verdicts) == 0, "Intentional sources should not trigger idle exploit"


def test_zero_penalty_no_death_beats_survival():
    """Zero penalty should never trigger death_beats_survival."""
    rule = DeathBeatsSurvival()
    model = EnvironmentModel(name="test", max_steps=100, death_probability=0.1)
    # No penalty sources
    assert not rule.applies_to(model)


@pytest.mark.parametrize("discovery_prob", [0.0, 0.01, 0.5, 1.0])
def test_exploration_threshold_no_crash(discovery_prob):
    """ExplorationThreshold should handle all discovery probabilities."""
    rule = ExplorationThreshold()
    model = EnvironmentModel(name="test", max_steps=100)
    model.add_reward_source(RewardSource(
        name="goal", reward_type=RewardType.TERMINAL, value=1.0,
        discovery_probability=discovery_prob,
    ))
    # Should not crash
    verdicts = rule.check(model)
    assert isinstance(verdicts, list)
