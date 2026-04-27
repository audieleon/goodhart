"""Tests for blind-spot advisory rules."""

from goodhart.models import (
    EnvironmentModel, RewardSource, RewardType, RespawnBehavior,
)
from goodhart.engine import TrainingAnalysisEngine


def _get_advisories(model, config=None):
    """Run analysis and return only advisory verdicts."""
    engine = TrainingAnalysisEngine().add_all_rules()
    result = engine.analyze(model, config)
    return [v for v in result.verdicts if v.rule_name.startswith("advisory_")]


def test_physics_exploit_fires_on_rich_env():
    """Complex continuous env with many rewards should trigger advisory."""
    m = EnvironmentModel(name="test", max_steps=1000,
                         n_states=100000, n_actions=12, death_probability=0.01)
    for i in range(5):
        m.add_reward_source(RewardSource(
            f"r{i}", RewardType.PER_STEP, 0.5,
            requires_action=True, state_dependent=True, scales_with="x"))
    advs = _get_advisories(m)
    names = {v.rule_name for v in advs}
    assert "advisory_physics_exploit" in names


def test_physics_exploit_silent_on_simple_env():
    """Simple gridworld should not trigger physics advisory."""
    m = EnvironmentModel(name="test", max_steps=100,
                         n_states=100, n_actions=4)
    m.add_reward_source(RewardSource("goal", RewardType.TERMINAL, 1.0))
    advs = _get_advisories(m)
    names = {v.rule_name for v in advs}
    assert "advisory_physics_exploit" not in names


def test_misgeneralization_fires_on_easy_goal():
    """High discovery rate single goal should trigger advisory."""
    m = EnvironmentModel(name="test", max_steps=1000,
                         n_states=100000, n_actions=15, death_probability=0.02)
    m.add_reward_source(RewardSource(
        "coin", RewardType.TERMINAL, 10.0,
        discovery_probability=0.8, requires_action=True))
    advs = _get_advisories(m)
    names = {v.rule_name for v in advs}
    assert "advisory_goal_misgeneralization" in names


def test_misgeneralization_silent_on_competitive():
    """Symmetric win/lose (competitive) should NOT trigger misgeneralization."""
    m = EnvironmentModel(name="test", max_steps=100, n_states=5000, n_actions=5)
    m.add_reward_source(RewardSource(
        "win", RewardType.TERMINAL, 1.0,
        discovery_probability=0.5, requires_action=True))
    m.add_reward_source(RewardSource(
        "lose", RewardType.TERMINAL, -1.0, requires_action=True))
    advs = _get_advisories(m)
    names = {v.rule_name for v in advs}
    assert "advisory_goal_misgeneralization" not in names


def test_nonstationarity_fires_on_symmetric_terminals():
    """Symmetric win/lose where both require action → competitive."""
    m = EnvironmentModel(name="test", max_steps=100, n_states=5000, n_actions=5)
    m.add_reward_source(RewardSource(
        "win", RewardType.TERMINAL, 10.0,
        discovery_probability=0.5, requires_action=True))
    m.add_reward_source(RewardSource(
        "lose", RewardType.TERMINAL, -10.0, requires_action=True))
    advs = _get_advisories(m)
    names = {v.rule_name for v in advs}
    assert "advisory_nonstationarity" in names


def test_nonstationarity_silent_on_survive_die():
    """+15/-15 survive/die where death is passive → NOT competitive."""
    m = EnvironmentModel(name="test", max_steps=48,
                         n_states=50000, n_actions=25, death_probability=0.15)
    m.add_reward_source(RewardSource(
        "survive", RewardType.TERMINAL, 15.0,
        requires_action=True, discovery_probability=0.7))
    m.add_reward_source(RewardSource(
        "die", RewardType.TERMINAL, -15.0, requires_action=False))
    advs = _get_advisories(m)
    names = {v.rule_name for v in advs}
    assert "advisory_nonstationarity" not in names


def test_credit_assignment_fires_on_deep_sparse():
    """Very sparse reward with no shaping in long episodes."""
    m = EnvironmentModel(name="test", max_steps=100000,
                         n_states=1000000, n_actions=77, death_probability=0.01)
    m.add_reward_source(RewardSource(
        "amulet", RewardType.TERMINAL, 10000.0,
        requires_action=True, requires_exploration=True,
        discovery_probability=0.0000001))
    advs = _get_advisories(m)
    names = {v.rule_name for v in advs}
    assert "advisory_credit_assignment" in names


def test_constrained_rl_fires_on_small_passive_penalty():
    """Small passive negative event should trigger constrained RL advisory."""
    m = EnvironmentModel(name="test", max_steps=500,
                         n_states=10000, n_actions=4)
    m.add_reward_source(RewardSource(
        "progress", RewardType.PER_STEP, 0.5,
        requires_action=True, intentional=True))
    m.add_reward_source(RewardSource(
        "boundary", RewardType.ON_EVENT, -0.05, requires_action=False))
    advs = _get_advisories(m)
    names = {v.rule_name for v in advs}
    assert "advisory_constrained_rl" in names


def test_constrained_rl_silent_on_death_penalty():
    """Large death penalty should NOT trigger constrained RL."""
    m = EnvironmentModel(name="test", max_steps=500,
                         n_states=10000, n_actions=4)
    m.add_reward_source(RewardSource(
        "score", RewardType.ON_EVENT, 1.0, requires_action=True))
    m.add_reward_source(RewardSource(
        "death", RewardType.ON_EVENT, -5.0, requires_action=False))
    advs = _get_advisories(m)
    names = {v.rule_name for v in advs}
    assert "advisory_constrained_rl" not in names


def test_missing_constraint_fires_on_continuous_all_positive():
    """All-positive continuous control with few sources → advisory."""
    m = EnvironmentModel(name="test", max_steps=1000,
                         n_states=100000, n_actions=10, death_probability=0.0)
    m.add_reward_source(RewardSource(
        "shape", RewardType.PER_STEP, 1.0,
        requires_action=True, intentional=True, scales_with="shape_error"))
    m.add_reward_source(RewardSource(
        "current", RewardType.PER_STEP, 0.5,
        requires_action=True, intentional=True))
    advs = _get_advisories(m)
    names = {v.rule_name for v in advs}
    assert "advisory_missing_constraint" in names


def test_missing_constraint_silent_on_discrete():
    """Discrete game with few sources → no advisory."""
    m = EnvironmentModel(name="test", max_steps=1000,
                         n_states=100000, n_actions=15, death_probability=0.02)
    m.add_reward_source(RewardSource(
        "coin", RewardType.TERMINAL, 10.0,
        discovery_probability=0.8, requires_action=True))
    advs = _get_advisories(m)
    names = {v.rule_name for v in advs}
    assert "advisory_missing_constraint" not in names


def test_aggregation_trap_fires_on_small_perstep():
    """Small per-step rewards with no terminal → aggregation advisory."""
    m = EnvironmentModel(name="test", max_steps=252,
                         n_states=100000, n_actions=3)
    m.add_reward_source(RewardSource(
        "return", RewardType.PER_STEP, 0.001,
        requires_action=True, intentional=True))
    advs = _get_advisories(m)
    names = {v.rule_name for v in advs}
    assert "advisory_aggregation_trap" in names


def test_aggregation_trap_silent_with_terminal():
    """Small per-step + terminal goal → no aggregation advisory."""
    m = EnvironmentModel(name="test", max_steps=252,
                         n_states=100000, n_actions=3)
    m.add_reward_source(RewardSource(
        "return", RewardType.PER_STEP, 0.001,
        requires_action=True, intentional=True))
    m.add_reward_source(RewardSource(
        "goal", RewardType.TERMINAL, 10.0, requires_action=True))
    advs = _get_advisories(m)
    names = {v.rule_name for v in advs}
    assert "advisory_aggregation_trap" not in names
