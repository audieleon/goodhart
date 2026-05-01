"""MCP (Model Context Protocol) server for goodhart.

Allows AI assistants (Claude Code, Cursor, etc.) to run goodhart
checks inline during conversations about RL experiments.

Usage:
    # In Claude Code's MCP config:
    {
        "mcpServers": {
            "goodhart": {
                "command": "python",
                "args": ["-m", "goodhart.mcp_server"]
            }
        }
    }

    # Then in conversation:
    User: "I'm setting up a sparse reward task with penalty -0.01..."
    Claude: [calls goodhart_check tool]
    Claude: "Goodhart analysis shows 4 critical issues..."
"""

import json
import sys

from goodhart import __version__
from goodhart.models import (
    EnvironmentModel, RewardSource, RewardType, RespawnBehavior,
    TrainingConfig,
)
from goodhart.engine import TrainingAnalysisEngine


def _build_reward_source(src: dict) -> RewardSource:
    """Build a RewardSource from a dict, handling all fields."""
    kwargs = {
        "name": src["name"],
        "reward_type": RewardType(src.get("type", "terminal")),
        "value": src.get("value", 0),
        "respawn": RespawnBehavior(src.get("respawn", "none")),
        "respawn_time": src.get("respawn_time", 0),
        "max_occurrences": src.get("max_occurrences", 1),
        "requires_action": src.get("requires_action", True),
        "requires_exploration": src.get("requires_exploration", False),
        "discovery_probability": float(src.get("discovery_probability", 1.0)),
        "can_loop": src.get("can_loop", False),
        "loop_period": src.get("loop_period", 0),
        "intentional": src.get("intentional", False),
        "explore_fraction": float(src.get("explore_fraction", 0.0)),
        "state_dependent": src.get("state_dependent", False),
        "scales_with": src.get("scales_with"),
        "value_type": src.get("value_type", "constant"),
        "prerequisite": src.get("prerequisite"),
        "modifies": src.get("modifies"),
        "modifier_type": src.get("modifier_type", "none"),
    }
    if "value_range" in src:
        kwargs["value_range"] = tuple(src["value_range"])
    if "value_params" in src:
        kwargs["value_params"] = src["value_params"]
    return RewardSource(**kwargs)


def _build_training_config(params: dict) -> TrainingConfig:
    """Build TrainingConfig from params, reading all supported fields."""
    kwargs = {}
    field_map = {
        "algorithm": str, "lr": float, "critic_lr": float,
        "entropy_coeff": float, "entropy_coeff_final": float,
        "num_specialists": int, "routing_floor": float, "balance_coef": float,
        "n_actors": int, "num_envs": int, "num_workers": int,
        "total_steps": int, "num_epochs": int, "minibatch_size": int,
        "rollout_length": int, "clip_epsilon": float, "target_kl": float,
        "model_params": int, "embed_dim": int,
        "use_rnn": bool, "rnn_type": str, "rnn_size": int,
        "max_grad_norm": float, "value_coef": float, "gae_lambda": float,
        # Off-policy
        "replay_buffer_size": int, "target_update_freq": int,
        "tau": float, "epsilon_start": float, "epsilon_end": float,
        "epsilon_decay_steps": int,
        # SAC
        "alpha": float, "auto_alpha": bool,
        # TD3
        "policy_noise": float, "noise_clip": float, "policy_delay": int,
    }
    for field, type_fn in field_map.items():
        if field in params:
            kwargs[field] = type_fn(params[field])
    if "num_envs" in kwargs and "n_actors" not in kwargs:
        kwargs["n_actors"] = kwargs["num_envs"]
    return TrainingConfig(**kwargs)


def handle_check(params: dict) -> dict:
    """Run a goodhart check and return structured results."""
    model = EnvironmentModel(
        name=params.get("name", "MCP check"),
        max_steps=params.get("max_steps", 500),
        gamma=float(params.get("gamma", 0.99)),
        n_states=params.get("n_states", 1000),
        n_actions=params.get("n_actions", 8),
        action_type=params.get("action_type", "auto"),
        death_probability=float(params.get("death_probability", 0.01)),
        wall_probability=float(params.get("wall_probability", 0.3)),
    )

    # Shorthand: goal_reward and step_penalty at top level
    if params.get("goal_reward", 0) > 0:
        model.add_reward_source(RewardSource(
            name="goal",
            reward_type=RewardType.TERMINAL,
            value=params["goal_reward"],
            discovery_probability=float(params.get("discovery_probability", 0.1)),
        ))

    if params.get("step_penalty", 0) != 0:
        model.add_reward_source(RewardSource(
            name="step_penalty",
            reward_type=RewardType.PER_STEP,
            value=params["step_penalty"],
        ))

    # Full reward sources
    for src in params.get("reward_sources", []):
        model.add_reward_source(_build_reward_source(src))

    # Training config (built if any training field present)
    training_fields = {"lr", "entropy_coeff", "num_specialists", "num_envs",
                       "total_steps", "algorithm", "num_epochs", "clip_epsilon",
                       "use_rnn", "n_actors"}
    config = None
    if any(k in params for k in training_fields):
        config = _build_training_config(params)

    engine = TrainingAnalysisEngine().add_all_rules()
    result = engine.analyze(model, config)

    verbose = params.get("verbose", True)  # default True for AI assistants
    return result.to_dict(verbose=verbose)


def handle_list_rules(params: dict) -> dict:
    """List all available rules with descriptions and proof status."""
    engine = TrainingAnalysisEngine().add_all_rules()
    rules = []
    for rule in engine.rules:
        entry = {
            "name": rule.name,
            "description": rule.description,
            "has_proof": rule.proof is not None,
        }
        if rule.proof:
            entry["proof_name"] = rule.proof.proof_name
            entry["proof_strength"] = rule.proof.strength
            entry["proof_statement"] = rule.proof.statement
        rules.append(entry)
    return {"rules": rules, "total": len(rules)}


def handle_explain(params: dict) -> dict:
    """Get detailed explanation for a specific rule."""
    from goodhart.rules.explanations import get_explanation

    rule_name = params.get("rule", "")
    engine = TrainingAnalysisEngine().add_all_rules()
    rule_map = {r.name: r for r in engine.rules}

    rule = rule_map.get(rule_name)
    if not rule:
        return {"error": f"Unknown rule: {rule_name}",
                "available": sorted(rule_map.keys())}

    result = {
        "name": rule.name,
        "description": rule.description,
    }

    entry = get_explanation(rule_name)
    if entry:
        result["learn_more"] = entry.get("learn_more", "")
        result["examples"] = entry.get("examples", [])
        result["papers"] = entry.get("papers", [])
        result["see_also"] = entry.get("see_also", [])
        if "proof" in entry:
            result["formal_basis"] = entry["proof"]

    if rule.proof:
        result["proof"] = {
            "name": rule.proof.proof_name,
            "strength": rule.proof.strength,
            "statement": rule.proof.statement,
        }

    return result


def handle_list_examples(params: dict) -> dict:
    """List cookbook examples with descriptions."""
    import pkgutil
    import goodhart.examples

    examples = []
    for m in sorted(pkgutil.iter_modules(goodhart.examples.__path__),
                    key=lambda m: m.name):
        if m.name == "__init__" or m.name.startswith("sample"):
            continue
        try:
            mod = __import__(f"goodhart.examples.{m.name}", fromlist=[m.name])
            doc = (mod.__doc__ or "").strip().split("\n")[0]
            doc = doc.replace("Example: ", "").replace('"""', "")
        except Exception:
            doc = ""
        examples.append({"name": m.name, "description": doc})
    return {"examples": examples, "total": len(examples)}


def handle_get_example(params: dict) -> dict:
    """Get full detail for a specific example.

    Returns the docstring (context, source, what it demonstrates),
    and if the example has a @reward_function-decorated function,
    also returns the analysis results.
    """
    import importlib

    name = params.get("name", "")
    try:
        mod = importlib.import_module(f"goodhart.examples.{name}")
    except ModuleNotFoundError:
        import pkgutil
        import goodhart.examples
        available = sorted(m.name for m in pkgutil.iter_modules(
            goodhart.examples.__path__) if m.name != "__init__"
            and not m.name.startswith("sample"))
        return {"error": f"Unknown example: {name}",
                "available": available}

    result = {
        "name": name,
        "docstring": (mod.__doc__ or "").strip(),
    }

    # Check for @reward_function-decorated functions in the module
    for attr_name in dir(mod):
        obj = getattr(mod, attr_name)
        if callable(obj) and hasattr(obj, 'goodhart_model'):
            model = obj.goodhart_model
            config = obj.goodhart_config
            engine = TrainingAnalysisEngine().add_all_rules()
            analysis = engine.analyze(model, config)
            result["has_reward_function"] = True
            result["function_name"] = attr_name
            result["environment"] = {
                "name": model.name,
                "max_steps": model.max_steps,
                "gamma": model.gamma,
                "n_actions": model.n_actions,
                "n_sources": len(model.reward_sources),
                "sources": [
                    {"name": s.name, "type": s.reward_type.value,
                     "value": s.value, "intentional": s.intentional}
                    for s in model.reward_sources
                ],
            }
            result["analysis"] = analysis.to_dict(verbose=True)
            break
    else:
        result["has_reward_function"] = False

    # Check the explanations DB for related rules
    from goodhart.rules.explanations import EXPLANATIONS
    related_rules = []
    for rule_name, entry in EXPLANATIONS.items():
        if name in entry.get("examples", []):
            related_rules.append(rule_name)
    if related_rules:
        result["demonstrates_rules"] = related_rules

    return result


def handle_doctor(params: dict) -> dict:
    """Diagnose issues and suggest a fixed configuration.

    Runs analysis, then for each critical/warning, suggests specific
    parameter changes that would resolve it.
    """
    # Build model from params (same as handle_check)
    model = EnvironmentModel(
        name=params.get("name", "doctor"),
        max_steps=params.get("max_steps", 500),
        gamma=float(params.get("gamma", 0.99)),
        n_states=params.get("n_states", 1000),
        n_actions=params.get("n_actions", 8),
        action_type=params.get("action_type", "auto"),
        death_probability=float(params.get("death_probability", 0.01)),
        wall_probability=float(params.get("wall_probability", 0.3)),
    )

    if params.get("goal_reward", 0) > 0:
        model.add_reward_source(RewardSource(
            name="goal", reward_type=RewardType.TERMINAL,
            value=params["goal_reward"],
            discovery_probability=float(params.get("discovery_probability", 0.1)),
        ))
    if params.get("step_penalty", 0) != 0:
        model.add_reward_source(RewardSource(
            name="step_penalty", reward_type=RewardType.PER_STEP,
            value=params["step_penalty"],
        ))
    for src in params.get("reward_sources", []):
        model.add_reward_source(_build_reward_source(src))

    training_fields = {"lr", "entropy_coeff", "num_specialists", "num_envs",
                       "total_steps", "algorithm", "num_epochs", "clip_epsilon",
                       "use_rnn", "n_actors", "replay_buffer_size",
                       "target_update_freq", "tau", "alpha", "auto_alpha"}
    config = None
    if any(k in params for k in training_fields):
        config = _build_training_config(params)

    engine = TrainingAnalysisEngine().add_all_rules()
    result = engine.analyze(model, config)

    # Build fix suggestions
    fixes = []
    for v in result.criticals + result.warnings:
        fix = {
            "rule": v.rule_name,
            "severity": v.severity.value,
            "issue": v.message,
        }
        if v.recommendation:
            fix["recommendation"] = v.recommendation
        if v.learn_more:
            fix["explanation"] = v.learn_more
        fixes.append(fix)

    return {
        "passed": result.passed,
        "n_criticals": len(result.criticals),
        "n_warnings": len(result.warnings),
        "fixes": fixes,
    }


# MCP tool definitions
TOOLS = [
    {
        "name": "goodhart_check",
        "description": (
            "Analyze an RL reward configuration for structural traps before "
            "training. Detects idle exploits, respawning loops, penalty traps, "
            "shaping issues, and more. Returns verdicts with severity, "
            "explanation, and fix recommendations."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Experiment name"},
                "goal_reward": {"type": "number", "description": "Terminal goal reward (shorthand)"},
                "step_penalty": {"type": "number", "description": "Per-step penalty (shorthand, negative)"},
                "max_steps": {"type": "integer", "description": "Max steps per episode"},
                "gamma": {"type": "number", "description": "Discount factor"},
                "discovery_probability": {"type": "number", "description": "P(finding goal per episode)"},
                "n_states": {"type": "integer", "description": "Approximate state space size"},
                "n_actions": {"type": "integer", "description": "Number of actions/actuators"},
                "action_type": {"type": "string", "enum": ["discrete", "continuous", "auto"]},
                "death_probability": {"type": "number", "description": "P(death per step)"},
                "lr": {"type": "number", "description": "Learning rate"},
                "entropy_coeff": {"type": "number", "description": "Entropy coefficient"},
                "num_envs": {"type": "integer", "description": "Number of parallel environments"},
                "total_steps": {"type": "integer", "description": "Total training steps"},
                "num_epochs": {"type": "integer", "description": "PPO reuse epochs"},
                "clip_epsilon": {"type": "number", "description": "PPO clip epsilon"},
                "num_specialists": {"type": "integer", "description": "Number of specialist networks"},
                "routing_floor": {"type": "number", "description": "Min routing weight per specialist"},
                "algorithm": {"type": "string", "enum": ["PPO", "APPO", "A2C", "IMPALA", "DQN", "SAC", "DDPG", "TD3"],
                              "description": "Training algorithm"},
                "use_rnn": {"type": "boolean", "description": "Whether to use recurrent network"},
                "replay_buffer_size": {"type": "integer", "description": "Off-policy replay buffer size (0=on-policy)"},
                "target_update_freq": {"type": "integer", "description": "DQN target network update frequency"},
                "tau": {"type": "number", "description": "Soft update coefficient (SAC/DDPG/TD3)"},
                "epsilon_start": {"type": "number", "description": "DQN initial exploration epsilon"},
                "epsilon_end": {"type": "number", "description": "DQN final exploration epsilon"},
                "epsilon_decay_steps": {"type": "integer", "description": "DQN epsilon decay duration"},
                "alpha": {"type": "number", "description": "SAC entropy temperature"},
                "auto_alpha": {"type": "boolean", "description": "SAC: learn alpha automatically"},
                "verbose": {"type": "boolean", "description": "Include learn_more explanations (default: true)"},
                "reward_sources": {
                    "type": "array",
                    "description": "Reward source components",
                    "items": {
                        "type": "object",
                        "required": ["name", "type", "value"],
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": ["terminal", "per_step", "on_event", "shaping"]},
                            "value": {"type": "number"},
                            "respawn": {"type": "string", "enum": ["none", "timed", "on_death", "on_episode", "infinite"]},
                            "respawn_time": {"type": "integer"},
                            "max_occurrences": {"type": "integer"},
                            "requires_action": {"type": "boolean", "description": "Agent must act to earn this"},
                            "requires_exploration": {"type": "boolean"},
                            "discovery_probability": {"type": "number"},
                            "can_loop": {"type": "boolean"},
                            "loop_period": {"type": "integer"},
                            "intentional": {"type": "boolean", "description": "This reward IS the goal"},
                            "explore_fraction": {"type": "number", "description": "Fraction earned by random exploration (0-1)"},
                            "state_dependent": {"type": "boolean"},
                            "scales_with": {"type": "string"},
                            "value_range": {"type": "array", "items": {"type": "number"}, "description": "[min, max]"},
                            "prerequisite": {"type": "string", "description": "Name of prerequisite reward source"},
                        },
                    },
                },
            },
        },
    },
    {
        "name": "goodhart_list_rules",
        "description": "List all 44 analysis rules with descriptions and formal proof status.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "goodhart_explain",
        "description": (
            "Get a detailed explanation of a specific rule: what it checks, "
            "why it matters, how to fix issues it finds, related examples, "
            "and formal proof basis."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["rule"],
            "properties": {
                "rule": {"type": "string", "description": "Rule name (e.g., idle_exploit)"},
            },
        },
    },
    {
        "name": "goodhart_list_examples",
        "description": "List 57 cookbook examples from published papers, each demonstrating a reward design pattern or failure mode.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "goodhart_get_example",
        "description": (
            "Get full detail for a specific example: docstring with context "
            "and source paper, the reward structure, analysis results, and "
            "which rules it demonstrates. Use this to explain a finding by "
            "showing a concrete real-world case."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string", "description": "Example name (e.g., humanoid_idle, coast_runners)"},
            },
        },
    },
    {
        "name": "goodhart_doctor",
        "description": (
            "Diagnose reward structure issues and suggest specific fixes. "
            "Like goodhart_check but returns actionable fix recommendations "
            "with full explanations for each issue found."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Experiment name"},
                "goal_reward": {"type": "number"},
                "step_penalty": {"type": "number"},
                "max_steps": {"type": "integer"},
                "gamma": {"type": "number"},
                "discovery_probability": {"type": "number"},
                "n_states": {"type": "integer"},
                "n_actions": {"type": "integer"},
                "action_type": {"type": "string", "enum": ["discrete", "continuous", "auto"]},
                "death_probability": {"type": "number"},
                "lr": {"type": "number"},
                "entropy_coeff": {"type": "number"},
                "num_envs": {"type": "integer"},
                "total_steps": {"type": "integer"},
                "reward_sources": {
                    "type": "array",
                    "items": {"type": "object"},
                },
            },
        },
    },
]

HANDLERS = {
    "goodhart_check": handle_check,
    "goodhart_list_rules": handle_list_rules,
    "goodhart_explain": handle_explain,
    "goodhart_list_examples": handle_list_examples,
    "goodhart_get_example": handle_get_example,
    "goodhart_doctor": handle_doctor,
}


def main():
    """Run as MCP server (stdio transport)."""
    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line.strip())
            request_id = request.get("id")
            method = request.get("method", "")

            if method == "initialize":
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {
                            "name": "goodhart",
                            "version": __version__,
                        },
                    },
                }

            elif method == "notifications/initialized":
                # Client confirms initialization — no response needed
                continue

            elif method == "tools/list":
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"tools": TOOLS},
                }

            elif method == "tools/call":
                tool_name = request["params"]["name"]
                tool_args = request["params"].get("arguments", {})

                if tool_name in HANDLERS:
                    result = HANDLERS[tool_name](tool_args)
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [{
                                "type": "text",
                                "text": json.dumps(result, indent=2),
                            }],
                        },
                    }
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
                    }

            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {},
                }

            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": str(e)},
            }
            sys.stdout.write(json.dumps(error_response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
