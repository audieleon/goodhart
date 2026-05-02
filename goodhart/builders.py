"""Model builders — construct EnvironmentModel and TrainingConfig from various inputs.

This module provides the construction logic used by both the CLI and
the Python API. Separated from cli.py to keep the library API independent
of the CLI dispatch layer.
"""

import json

from goodhart.models import (
    EnvironmentModel,
    RewardSource,
    TrainingConfig,
    RewardType,
    RespawnBehavior,
)


def build_model_and_config(
    goal=0.0,
    penalty=0.0,
    max_steps=500,
    discovery_prob=0.05,
    n_actors=64,
    total_steps=20_000_000,
    lr=3e-4,
    critic_lr=None,
    entropy=0.01,
    n_specialists=1,
    routing_floor=0.0,
    n_states=1000,
    gamma=0.99,
    name="experiment",
):
    """Build EnvironmentModel and TrainingConfig from simple keyword args.

    This is the construction path used by the quick API (check/analyze)
    and the CLI when using --goal/--penalty/--steps flags.
    """
    model = EnvironmentModel(name=name, max_steps=max_steps, n_states=n_states, gamma=gamma)

    if goal > 0:
        model.add_reward_source(
            RewardSource(
                name="goal",
                reward_type=RewardType.TERMINAL,
                value=goal,
                discovery_probability=discovery_prob,
            )
        )

    if penalty != 0:
        model.add_reward_source(
            RewardSource(
                name="step penalty",
                reward_type=RewardType.PER_STEP,
                value=penalty,
                requires_action=False,  # CLI penalty is constant (passive)
            )
        )

    config = TrainingConfig(
        lr=lr,
        critic_lr=critic_lr,
        entropy_coeff=entropy,
        num_specialists=n_specialists,
        routing_floor=routing_floor,
        num_envs=n_actors,
        n_actors=n_actors,
        total_steps=total_steps,
    )

    return model, config


def load_config_file(path):
    """Load config from YAML, JSON, or TOML file based on extension.

    Use path='-' to read from stdin (YAML or JSON auto-detected).
    """
    import os

    # Read from stdin
    if path == "-" or path == "/dev/stdin":
        import sys

        content = sys.stdin.read()
        if not content.strip():
            raise ValueError("Empty input on stdin")
        # Try JSON first, fall back to YAML
        try:
            return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            import yaml

            return yaml.safe_load(content)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    if os.path.getsize(path) > 1_000_000:
        raise ValueError(
            f"Config file too large (>{os.path.getsize(path) // 1_000_000}MB). Max 1MB to prevent denial-of-service."
        )
    if path.endswith((".yaml", ".yml")):
        import yaml

        # safe_load blocks code execution. YAML alias bombs are not
        # exploitable here: PyYAML creates shared Python references
        # (not copies), and build_from_config_dict only reads known
        # keys, so alias-expanded data is never traversed.
        with open(path) as f:
            return yaml.safe_load(f)
    elif path.endswith(".json"):
        with open(path) as f:
            return json.load(f)
    elif path.endswith(".toml"):
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib
        with open(path, "rb") as f:
            return tomllib.load(f)
    else:
        raise ValueError(f"Unsupported config format: {path}. Use .yaml, .yml, .json, or .toml")


def build_from_config_dict(cfg, fallback_name="config"):
    """Build EnvironmentModel and TrainingConfig from a config dict.

    Supports two YAML layouts:
    - Nested: { environment: { ... }, training: { ... } }
    - Flat: { name: ..., max_steps: ..., reward_sources: [...], training: { ... } }
    """
    if cfg is None:
        raise ValueError("Config file is empty or invalid")
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a dict, got {type(cfg).__name__}")
    if "environment" in cfg:
        env_cfg = cfg["environment"]
    else:
        env_cfg = cfg
    train_cfg = cfg.get("training", {})

    model = EnvironmentModel(
        name=env_cfg.get("name", fallback_name),
        max_steps=env_cfg.get("max_steps", 500),
        gamma=float(env_cfg.get("gamma", 0.99)),
        n_states=env_cfg.get("n_states", 1000),
        n_actions=env_cfg.get("n_actions", 8),
        action_type=env_cfg.get("action_type", "auto"),
        death_probability=float(env_cfg.get("death_probability", 0.01)),
        wall_probability=float(env_cfg.get("wall_probability", 0.3)),
    )

    for i, src in enumerate(env_cfg.get("reward_sources", [])):
        if not isinstance(src, dict):
            raise ValueError(f"reward_sources[{i}] must be a dict, got {type(src).__name__}")
        rt_str = src.get("reward_type", src.get("type", "terminal"))
        model.add_reward_source(
            RewardSource(
                name=src.get("name", f"source_{i}"),
                reward_type=RewardType(rt_str),
                value=src.get("value", 0.0),
                respawn=RespawnBehavior(src.get("respawn", "none")),
                respawn_time=src.get("respawn_time", 0),
                max_occurrences=src.get("max_occurrences", 1),
                requires_action=src.get("requires_action", True),
                requires_exploration=src.get("requires_exploration", False),
                discovery_probability=float(src.get("discovery_probability", 1.0)),
                can_loop=src.get("can_loop", False),
                loop_period=src.get("loop_period", 0),
                intentional=src.get("intentional", False),
                value_range=tuple(src["value_range"][:2])
                if "value_range" in src and len(src["value_range"]) >= 2
                else None,
                value_type=src.get("value_type", "constant"),
                value_params=src.get("value_params"),
                state_dependent=src.get("state_dependent", False),
                scales_with=src.get("scales_with", None),
                explore_fraction=float(src.get("explore_fraction", 0.0)),
                prerequisite=src.get("prerequisite"),
                modifies=src.get("modifies"),
                modifier_type=src.get("modifier_type", "none"),
            )
        )

    config = TrainingConfig(
        algorithm=train_cfg.get("algorithm", "PPO"),
        lr=float(train_cfg.get("lr", 3e-4)),
        critic_lr=float(train_cfg["critic_lr"]) if "critic_lr" in train_cfg else None,
        entropy_coeff=float(train_cfg.get("entropy_coeff", 0.01)),
        entropy_coeff_final=float(train_cfg.get("entropy_coeff_final", 0.001)),
        num_envs=train_cfg.get("num_envs", 16),
        num_workers=train_cfg.get("num_workers", 1),
        n_actors=train_cfg.get("n_actors", 16),
        total_steps=train_cfg.get("total_steps", 10_000_000),
        num_specialists=train_cfg.get("num_specialists", 1),
        routing_floor=float(train_cfg.get("routing_floor", 0.0)),
        balance_coef=float(train_cfg.get("balance_coef", 0.0)),
        model_params=train_cfg.get("model_params", 1_000_000),
        num_epochs=train_cfg.get("num_epochs", 4),
        minibatch_size=train_cfg.get("minibatch_size", 512),
        rollout_length=train_cfg.get("rollout_length", 128),
        clip_epsilon=float(train_cfg.get("clip_epsilon", 0.2)),
        target_kl=float(train_cfg["target_kl"]) if "target_kl" in train_cfg else None,
        max_grad_norm=float(train_cfg.get("max_grad_norm", 0.5)),
        value_coef=float(train_cfg.get("value_coef", 0.5)),
        gae_lambda=float(train_cfg.get("gae_lambda", 0.95)),
        embed_dim=train_cfg.get("embed_dim", 256),
        use_rnn=train_cfg.get("use_rnn", False),
        rnn_type=train_cfg.get("rnn_type", "lstm"),
        rnn_size=train_cfg.get("rnn_size", 256),
        # Off-policy
        replay_buffer_size=train_cfg.get("replay_buffer_size", 0),
        target_update_freq=train_cfg.get("target_update_freq", 0),
        tau=float(train_cfg.get("tau", 0.005)),
        epsilon_start=float(train_cfg.get("epsilon_start", 1.0)),
        epsilon_end=float(train_cfg.get("epsilon_end", 0.01)),
        epsilon_decay_steps=train_cfg.get("epsilon_decay_steps", 0),
        # SAC
        alpha=float(train_cfg.get("alpha", 0.2)),
        auto_alpha=train_cfg.get("auto_alpha", False),
        # TD3
        policy_noise=float(train_cfg.get("policy_noise", 0.2)),
        noise_clip=float(train_cfg.get("noise_clip", 0.5)),
        policy_delay=train_cfg.get("policy_delay", 2),
    )

    return model, config
