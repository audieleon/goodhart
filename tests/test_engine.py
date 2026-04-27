"""Tests for engine.py — analysis composition, enrichment, contradictions."""

from goodhart.engine import AnalysisEngine, TrainingAnalysisEngine, Rule
from goodhart.models import (
    EnvironmentModel, RewardSource, RewardType, Severity, Verdict,
)


class DummyRule(Rule):
    @property
    def name(self): return "dummy"
    @property
    def description(self): return "A dummy rule"
    def check(self, model, config=None):
        return [Verdict("dummy", Severity.WARNING, "dummy warning",
                        recommendation="fix it")]


class CrashingRule(Rule):
    @property
    def name(self): return "crasher"
    @property
    def description(self): return "A rule that crashes"
    def check(self, model, config=None):
        raise RuntimeError("boom")


def _simple_model():
    m = EnvironmentModel(name="test", max_steps=100)
    m.add_reward_source(RewardSource("goal", RewardType.TERMINAL, 1.0))
    return m


def test_add_rule():
    engine = AnalysisEngine()
    engine.add_rule(DummyRule())
    assert len(engine.rules) == 1


def test_add_rule_dedup():
    engine = AnalysisEngine()
    engine.add_rule(DummyRule())
    engine.add_rule(DummyRule())
    assert len(engine.rules) == 1


def test_analyze_returns_result():
    engine = AnalysisEngine()
    engine.add_rule(DummyRule())
    result = engine.analyze(_simple_model())
    assert result.passed is True  # warning, not critical
    assert len(result.warnings) == 1


def test_crashing_rule_handled():
    engine = AnalysisEngine()
    engine.add_rule(CrashingRule())
    result = engine.analyze(_simple_model())
    assert len(result.warnings) == 1
    assert "crashed" in result.warnings[0].message


def test_enrichment_adds_learn_more():
    """Verdicts should get learn_more from explanations DB."""
    engine = TrainingAnalysisEngine().add_all_rules()
    m = EnvironmentModel(name="test", max_steps=200, gamma=1.0,
                         n_states=500, n_actions=3, death_probability=0.0)
    m.add_reward_source(RewardSource(
        "goal", RewardType.TERMINAL, 1.0,
        requires_action=True, requires_exploration=True,
        discovery_probability=0.01))
    m.add_reward_source(RewardSource(
        "penalty", RewardType.PER_STEP, -1.0))
    result = engine.analyze(m)
    # penalty_dominates_goal should fire and have learn_more
    pdg = [v for v in result.verdicts if v.rule_name == "penalty_dominates_goal"]
    assert len(pdg) > 0
    assert pdg[0].learn_more is not None


def test_training_engine_has_all_categories():
    engine = TrainingAnalysisEngine().add_all_rules()
    names = {r.name for r in engine.rules}
    # Should have rules from all categories
    assert "penalty_dominates_goal" in names  # reward
    assert "lr_regime" in names               # training
    assert "embed_dim_capacity" in names      # architecture
    assert "advisory_physics_exploit" in names  # advisory


def test_contradiction_detection():
    """Engine should detect contradictory recommendations."""
    engine = TrainingAnalysisEngine().add_all_rules()
    # Humanoid-v4 style: healthy_reward creates idle exploit + compound trap
    # which produce contradictory "add penalty" vs "remove penalty" recs
    from goodhart.models import RespawnBehavior
    m = EnvironmentModel(name="test", max_steps=1000,
                         n_states=100000, n_actions=17, death_probability=0.03)
    m.add_reward_source(RewardSource(
        "healthy", RewardType.PER_STEP, 5.0,
        respawn=RespawnBehavior.INFINITE,
        requires_action=False, intentional=False))
    m.add_reward_source(RewardSource(
        "velocity", RewardType.PER_STEP, 1.25,
        requires_action=True, intentional=True))
    m.add_reward_source(RewardSource(
        "ctrl", RewardType.PER_STEP, -0.1, requires_action=True))
    result = engine.analyze(m)
    contradiction = [v for v in result.verdicts if v.rule_name == "contradiction"]
    assert len(contradiction) > 0
