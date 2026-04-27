"""Tests for the MCP server handlers."""

from goodhart.mcp_server import handle_check, handle_list_rules


def _all_rules(result):
    """Extract all rule names from a check result."""
    rules = []
    for key in ("criticals", "warnings", "infos"):
        for v in result.get(key, []):
            rules.append(v["rule"])
    return rules


def test_handle_check_basic():
    """Basic check with goal and penalty."""
    result = handle_check({
        "goal_reward": 1.0,
        "step_penalty": -0.01,
        "max_steps": 500,
    })
    assert "passed" in result
    assert isinstance(result["criticals"], list)


def test_handle_check_clean():
    """Clean config should pass."""
    result = handle_check({
        "goal_reward": 1.0,
        "max_steps": 100,
    })
    assert result["passed"] is True


def test_handle_check_with_training():
    """Check with training params including expert collapse."""
    result = handle_check({
        "goal_reward": 1.0,
        "max_steps": 500,
        "lr": 3e-4,
        "num_specialists": 3,
        "routing_floor": 0.0,
    })
    assert "passed" in result
    assert "expert_collapse" in _all_rules(result)


def test_handle_check_with_reward_sources():
    """Check with loopable reward source."""
    result = handle_check({
        "max_steps": 500,
        "reward_sources": [
            {"name": "goal", "type": "terminal", "value": 10.0},
            {"name": "loop", "type": "on_event", "value": 5.0,
             "can_loop": True, "loop_period": 10},
        ],
    })
    assert "respawning_exploit" in _all_rules(result)


def test_handle_check_empty():
    """Empty params should not crash."""
    result = handle_check({})
    assert "passed" in result


def test_handle_list_rules():
    """List rules returns all rules."""
    result = handle_list_rules({})
    assert "rules" in result
    assert "total" in result
    assert result["total"] >= 24
    for rule in result["rules"]:
        assert "name" in rule
        assert "description" in rule


def test_handle_list_rules_proof_info():
    """Rules with proofs include proof info."""
    result = handle_list_rules({})
    proved = [r for r in result["rules"] if "proof_name" in r]
    assert len(proved) >= 10
