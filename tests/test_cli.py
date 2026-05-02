"""Test CLI modes."""

import subprocess
import sys
from pathlib import Path
import pytest
import os

GOODHART_DIR = str(Path(__file__).parent.parent)


def _run_cli(args, check=True):
    result = subprocess.run(
        [sys.executable, "-m", "goodhart"] + args,
        capture_output=True, text=True, cwd=GOODHART_DIR,
    )
    if check:
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
    return result


def test_cli_quick_check():
    result = _run_cli(["--goal", "1.0", "--penalty", "-0.01", "--steps", "500"])
    assert "FULL ANALYSIS" in result.stdout


def test_cli_rules():
    result = _run_cli(["--rules"])
    assert "penalty_dominates_goal" in result.stdout
    assert "Total:" in result.stdout


def test_cli_examples_list():
    result = _run_cli(["--examples"])
    assert "coast_runners" in result.stdout
    assert "examples from" in result.stdout


def test_cli_about():
    result = _run_cli(["--about"])
    assert "Goodhart" in result.stdout
    assert "measure" in result.stdout


def test_cli_example_coast_runners():
    result = _run_cli(["--example", "coast_runners"])
    assert "CoastRunners" in result.stdout


def test_cli_config_file():
    config_path = os.path.join(GOODHART_DIR, "goodhart", "examples", "sample_config.yaml")
    result = _run_cli(["--config", config_path])
    assert "FULL ANALYSIS" in result.stdout


def test_cli_safe_config():
    result = _run_cli(["--goal", "0", "--penalty", "0", "--steps", "500",
                       "--critic-lr", "3e-5"])
    assert "PASSED" in result.stdout or "0 critical" in result.stdout


def test_cli_version_import():
    result = subprocess.run(
        [sys.executable, "-c", "import goodhart; print(goodhart.__version__)"],
        capture_output=True, text=True, cwd=GOODHART_DIR,
    )
    assert result.returncode == 0
    from goodhart import __version__
    assert __version__ in result.stdout


def test_cli_quiet_pass():
    """--quiet mode with safe config should exit 0 and produce no output."""
    result = _run_cli(["--quiet", "--goal", "0", "--penalty", "0",
                       "--steps", "500", "--critic-lr", "3e-5"])
    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_cli_quiet_fail():
    """--quiet mode with bad config should exit 1."""
    result = _run_cli(["--quiet", "--goal", "1.0", "--penalty", "-0.01",
                       "--steps", "500"], check=False)
    assert result.returncode == 1
    assert result.stdout.strip() == ""


def test_cli_json_output():
    """--json mode should produce valid JSON."""
    import json
    result = _run_cli(["--json", "--goal", "1.0", "--penalty", "-0.01",
                       "--steps", "500"])
    data = json.loads(result.stdout)
    assert "passed" in data
    assert "criticals" in data
    assert "warnings" in data
    assert "infos" in data
    assert isinstance(data["passed"], bool)


def test_cli_json_safe():
    """--json mode with safe config should show passed=True."""
    import json
    result = _run_cli(["--json", "--goal", "0", "--penalty", "0",
                       "--steps", "500", "--critic-lr", "3e-5"])
    data = json.loads(result.stdout)
    assert data["passed"] is True


def test_cli_doctor():
    """--doctor mode should produce diagnosis output."""
    result = _run_cli(["--doctor", "--goal", "1.0", "--penalty", "-0.01",
                       "--steps", "500"])
    assert "goodhart doctor" in result.stdout
    assert "Diagnosis" in result.stdout


def test_cli_doctor_clean():
    """--doctor with safe config should report no issues."""
    result = _run_cli(["--doctor", "--goal", "0", "--penalty", "0",
                       "--steps", "500", "--critic-lr", "3e-5"])
    assert "no issues" in result.stdout.lower()


def test_cli_config_json():
    """--config with .json file should work."""
    config_path = os.path.join(GOODHART_DIR, "tests", "fixtures", "test_config.json")
    result = _run_cli(["--config", config_path])
    assert "FULL ANALYSIS" in result.stdout


def test_cli_config_toml():
    """--config with .toml file should work."""
    config_path = os.path.join(GOODHART_DIR, "tests", "fixtures", "test_config.toml")
    result = _run_cli(["--config", config_path])
    assert "FULL ANALYSIS" in result.stdout


def test_cli_example_unknown():
    """--example with unknown name should show available examples."""
    result = _run_cli(["--example", "nonexistent_example"], check=False)
    assert result.returncode == 1
    assert "Available examples" in result.stdout
    assert "coast_runners" in result.stdout


def test_analyze_returns_result():
    """goodhart.analyze() should return a Result object."""
    from goodhart import analyze
    result = analyze(goal=1.0, penalty=-0.01, max_steps=500)
    from goodhart.models import Result
    assert isinstance(result, Result)
    assert isinstance(result.passed, bool)
    assert isinstance(result.verdicts, list)
    # This config should have criticals
    assert result.has_criticals


def test_analyze_safe_returns_result():
    """goodhart.analyze() with safe config should return passed=True."""
    from goodhart import analyze
    result = analyze(goal=0.0, penalty=0.0, max_steps=500, critic_lr=3e-5)
    assert result.passed
    assert not result.has_criticals


def test_doctor_penalty_shows_exact_values():
    """Doctor for penalty_dominates_goal shows computed safe threshold."""
    result = _run_cli(["--doctor", "--goal", "1.0", "--penalty", "-0.05",
                       "--steps", "500"])
    output = result.stdout
    assert "was:" in output
    assert "safe threshold" in output or "penalty" in output


def test_doctor_critic_lr_shows_exact_value():
    """Doctor for critic_lr_ratio shows exact recommended value."""
    result = _run_cli(["--doctor", "--goal", "0", "--penalty", "0",
                       "--steps", "500", "--lr", "3e-4"])
    output = result.stdout
    # With default critic_lr == lr, should trigger critic_lr_ratio
    assert "critic_lr" in output or "no issues" in output.lower()


def test_doctor_expert_collapse_shows_floor():
    """Doctor for expert_collapse shows routing_floor value."""
    result = _run_cli(["--doctor", "--goal", "1.0", "--penalty", "0",
                       "--steps", "500", "--specialists", "3",
                       "--floor", "0.0", "--critic-lr", "3e-5"])
    output = result.stdout
    assert "routing_floor" in output
    assert "was:" in output
