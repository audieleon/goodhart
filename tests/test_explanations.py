"""Tests for the explanations database — coverage and consistency."""

from goodhart.rules.explanations import EXPLANATIONS, get_explanation, get_learn_more, get_related_examples
from goodhart.engine import TrainingAnalysisEngine
import pkgutil
import goodhart.examples


def test_every_rule_has_explanation():
    """All 35 rules should have an entry in EXPLANATIONS."""
    engine = TrainingAnalysisEngine().add_all_rules()
    all_rules = {r.name for r in engine.rules}
    explained = set(EXPLANATIONS.keys())
    missing = all_rules - explained
    assert missing == set(), f"Rules without explanations: {missing}"


def test_every_explanation_has_learn_more():
    """Every explanation entry should have a learn_more field."""
    for name, entry in EXPLANATIONS.items():
        assert "learn_more" in entry, f"{name} missing learn_more"
        assert len(entry["learn_more"]) > 50, f"{name} learn_more too short"


def test_referenced_examples_exist():
    """Examples listed in explanations should exist as actual modules."""
    actual = {m.name for m in pkgutil.iter_modules(goodhart.examples.__path__)
              if m.name != "__init__"}
    for name, entry in EXPLANATIONS.items():
        for ex in entry.get("examples", []):
            assert ex in actual, f"Explanation '{name}' references non-existent example '{ex}'"


def test_get_explanation():
    """get_explanation should return dict or None."""
    assert get_explanation("idle_exploit") is not None
    assert get_explanation("nonexistent_rule") is None


def test_get_learn_more():
    """get_learn_more should return string or None."""
    text = get_learn_more("penalty_dominates_goal")
    assert text is not None
    assert "agent" in text.lower()


def test_get_related_examples():
    """get_related_examples should return list."""
    examples = get_related_examples("idle_exploit")
    assert isinstance(examples, list)
    assert len(examples) > 0
