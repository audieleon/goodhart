"""Tests for builders.py — config file loading and model construction."""

import os
import json
import tempfile
import pytest
from goodhart.builders import build_model_and_config, load_config_file, build_from_config_dict


def test_build_simple():
    model, config = build_model_and_config(goal=1.0, penalty=-0.01, max_steps=500)
    assert model.max_steps == 500
    assert model.max_goal_reward == 1.0
    assert model.total_step_penalty == -0.01


def test_build_no_goal():
    model, config = build_model_and_config(goal=0.0, penalty=0.0, max_steps=500)
    assert len(model.reward_sources) == 0


def test_build_with_training():
    model, config = build_model_and_config(
        goal=1.0, penalty=-0.01, max_steps=500,
        lr=1e-3, n_specialists=3, routing_floor=0.1)
    assert config.lr == 1e-3
    assert config.num_specialists == 3
    assert config.routing_floor == 0.1


def test_load_yaml():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("name: test\nmax_steps: 200\n")
        f.flush()
        try:
            cfg = load_config_file(f.name)
            assert cfg["name"] == "test"
            assert cfg["max_steps"] == 200
        finally:
            os.unlink(f.name)


def test_load_json():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"name": "test", "max_steps": 100}, f)
        f.flush()
        try:
            cfg = load_config_file(f.name)
            assert cfg["name"] == "test"
        finally:
            os.unlink(f.name)


def test_load_unsupported_format():
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(b"hello")
        f.flush()
        try:
            with pytest.raises(ValueError, match="Unsupported"):
                load_config_file(f.name)
        finally:
            os.unlink(f.name)


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config_file("/nonexistent/file.yaml")


def test_build_from_dict_nested():
    cfg = {
        "environment": {
            "name": "test",
            "max_steps": 300,
            "reward_sources": [
                {"name": "goal", "type": "terminal", "value": 1.0},
            ],
        },
        "training": {"lr": 1e-4},
    }
    model, config = build_from_config_dict(cfg)
    assert model.name == "test"
    assert model.max_steps == 300
    assert len(model.reward_sources) == 1
    assert config.lr == 1e-4


def test_build_from_dict_flat():
    cfg = {
        "name": "flat_test",
        "max_steps": 100,
        "reward_sources": [],
    }
    model, config = build_from_config_dict(cfg)
    assert model.name == "flat_test"


def test_build_from_dict_empty():
    with pytest.raises(ValueError, match="empty"):
        build_from_config_dict(None)


def test_build_from_dict_not_dict():
    with pytest.raises(ValueError, match="must be a dict"):
        build_from_config_dict("not a dict")


def test_build_action_type():
    cfg = {
        "name": "test",
        "action_type": "continuous",
        "reward_sources": [],
    }
    model, config = build_from_config_dict(cfg)
    assert model.action_type == "continuous"
