"""Preset verdict regression tests.

Each test verifies that a specific preset produces the expected
findings. These catch regressions from rule changes.
"""

import pytest
from goodhart.presets import PRESETS
from goodhart.engine import TrainingAnalysisEngine
from goodhart.models import Severity


def _analyze_preset(name):
    model, config = PRESETS[name]
    engine = TrainingAnalysisEngine().add_all_rules()
    return engine.analyze(model, config)


def _critical_rules(result):
    return {v.rule_name for v in result.criticals}


def _warning_rules(result):
    return {v.rule_name for v in result.warnings}


# --- Clean presets (should pass) ---

def test_atari_clean():
    result = _analyze_preset("atari")
    assert result.passed, f"Atari should pass, got criticals: {_critical_rules(result)}"


def test_mujoco_locomotion_clean():
    result = _analyze_preset("mujoco-locomotion")
    assert result.passed, f"MuJoCo locomotion should pass, got: {_critical_rules(result)}"


def test_sparse_goal_clean():
    result = _analyze_preset("sparse-goal")
    assert result.passed


def test_dense_survival_clean():
    result = _analyze_preset("dense-survival")
    assert result.passed


def test_coinrun_clean():
    """CoinRun has a clean reward structure; the issue is generalization."""
    result = _analyze_preset("coinrun")
    assert result.passed


def test_bipedal_walker_clean():
    """BipedalWalker: clean with explore_fraction=0.5 on forward progress.
    Random exploration earns ~50% of velocity reward, which outweighs
    the torque penalty."""
    result = _analyze_preset("bipedal-walker")
    assert result.passed, f"BipedalWalker should pass, got: {_critical_rules(result)}"


# --- Presets with known issues ---

def test_coast_runners_loop_exploit():
    """CoastRunners: loop EV beats goal (Clark & Amodei 2016)."""
    result = _analyze_preset("coast-runners")
    assert not result.passed
    assert "respawning_exploit" in _critical_rules(result)


def test_mountain_car_desert():
    """Mountain Car: reward desert, no gradient signal."""
    result = _analyze_preset("mountain-car")
    assert not result.passed
    assert "penalty_dominates_goal" in _critical_rules(result)
    # Should identify as desert, not trap
    desert_verdicts = [v for v in result.criticals
                       if v.rule_name == "exploration_threshold"
                       and "desert" in v.message.lower()]
    assert len(desert_verdicts) > 0, "Should identify reward desert"


def test_anymal_idle_exploit():
    """ANYmal: alive bonus creates idle exploit (Hwangbo 2019)."""
    result = _analyze_preset("anymal")
    assert not result.passed
    assert "idle_exploit" in _critical_rules(result)


def test_minihack_navigation_exploration_warning():
    """MiniHack navigation: exploration threshold warning (Samvelyan 2021).
    Step penalty is action-dependent, so death_beats_survival doesn't fire.
    The real issue is sparse reward + exploration difficulty."""
    result = _analyze_preset("minihack-navigation")
    assert "exploration_threshold" in _warning_rules(result)


def test_minihack_skill_exploration_warning():
    """MiniHack skill: same issue, harder task."""
    result = _analyze_preset("minihack-skill")
    assert "exploration_threshold" in _warning_rules(result)


def test_hand_manipulation_exploration():
    """Shadow Hand: exploration impossible (Andrychowicz 2020).
    The fingertip contact bonus (+0.1/step) and sparse goal make
    exploration impractical under the current reward structure."""
    result = _analyze_preset("hand-manipulation")
    assert not result.passed
    assert "exploration_threshold" in _critical_rules(result)


# --- Verify no false positives on clean presets ---

def test_no_false_criticals():
    """Clean presets should never produce criticals."""
    clean_presets = ["atari", "mujoco-locomotion", "sparse-goal",
                     "dense-survival", "coinrun", "bipedal-walker"]
    for name in clean_presets:
        result = _analyze_preset(name)
        assert result.passed, (
            f"Preset {name} should be clean but got criticals: "
            f"{_critical_rules(result)}"
        )
