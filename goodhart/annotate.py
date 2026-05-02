"""Decorator-based reward annotation for Python reward functions.

Attach reward structure metadata to a function so goodhart can
analyze it without parsing the function body. The decorator is
the bridge between "reward function in Python" and "EnvironmentModel
that the engine can analyze."

Usage:

    from goodhart import reward_function, RewardSource

    @reward_function(
        max_steps=1000,
        gamma=0.99,
        sources=[
            RewardSource("alive", RewardType.PER_STEP, 1.0,
                         requires_action=False, intentional=True),
            RewardSource("velocity", RewardType.PER_STEP, 0.5,
                         state_dependent=True, intentional=True),
            RewardSource("ctrl_cost", RewardType.PER_STEP, -0.001,
                         requires_action=True),
            RewardSource("goal", RewardType.TERMINAL, 10.0,
                         discovery_probability=0.05),
        ],
        n_states=10000,
        n_actions=8,
        action_type="continuous",
    )
    def compute_reward(obs, action, info):
        reward = 1.0
        reward += obs["velocity"] * 0.5
        reward -= 0.001 * np.linalg.norm(action)
        if info.get("goal_reached"):
            reward += 10.0
        return reward

Then:
    # Python API
    from goodhart import analyze_function
    result = analyze_function(compute_reward)

    # CLI
    goodhart --check my_module:compute_reward

    # Or just call the analysis directly
    compute_reward.goodhart_check()
"""

import functools
from typing import List, Optional

from goodhart.models import (
    EnvironmentModel,
    RewardSource,
    TrainingConfig,
    Result,
)
from goodhart.engine import TrainingAnalysisEngine


def reward_function(
    sources: List[RewardSource],
    max_steps: int = 500,
    gamma: float = 0.99,
    n_states: int = 1000,
    n_actions: int = 8,
    action_type: str = "auto",
    death_probability: float = 0.01,
    wall_probability: float = 0.3,
    name: Optional[str] = None,
    # Training config (optional — if provided, training rules also run)
    lr: Optional[float] = None,
    entropy_coeff: Optional[float] = None,
    num_envs: Optional[int] = None,
    total_steps: Optional[int] = None,
    algorithm: str = "PPO",
    **training_kwargs,
):
    """Decorate a reward function with structural metadata.

    The decorated function works identically to the original — calling
    it runs the reward computation. The metadata is attached as attributes
    and used by goodhart's analysis engine.

    Args:
        sources: List of RewardSource objects describing each component.
        max_steps: Maximum episode length.
        gamma: Discount factor.
        n_states: Approximate state space size.
        n_actions: Number of actions (discrete) or actuators (continuous).
        action_type: "discrete", "continuous", or "auto".
        death_probability: Per-step termination probability.
        wall_probability: Per-step wasted action probability.
        name: Environment name (defaults to function name).
        lr: Learning rate (enables training rules if provided).
        entropy_coeff: Entropy coefficient.
        num_envs: Number of parallel environments.
        total_steps: Total training budget.
        algorithm: Training algorithm name.
        **training_kwargs: Additional TrainingConfig fields.
    """

    def decorator(fn):
        env_name = name or fn.__name__

        model = EnvironmentModel(
            name=env_name,
            max_steps=max_steps,
            gamma=gamma,
            n_states=n_states,
            n_actions=n_actions,
            action_type=action_type,
            death_probability=death_probability,
            wall_probability=wall_probability,
        )
        for source in sources:
            model.add_reward_source(source)

        config = None
        if lr is not None:
            config_args = dict(
                algorithm=algorithm,
                lr=lr,
                **training_kwargs,
            )
            if entropy_coeff is not None:
                config_args["entropy_coeff"] = entropy_coeff
            if num_envs is not None:
                config_args["num_envs"] = num_envs
                config_args["n_actors"] = num_envs
            if total_steps is not None:
                config_args["total_steps"] = total_steps
            config = TrainingConfig(**config_args)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        # Attach metadata for analysis
        wrapper.goodhart_model = model
        wrapper.goodhart_config = config
        wrapper.goodhart_sources = sources

        def goodhart_check(verbose=False, quiet=False) -> Result:
            """Run goodhart analysis on this function's reward structure."""
            engine = TrainingAnalysisEngine().add_all_rules()
            if not quiet:
                return engine.print_report(model, config, verbose=verbose)
            return engine.analyze(model, config)

        def goodhart_passed() -> bool:
            """Quick check: returns True if no criticals."""
            engine = TrainingAnalysisEngine().add_all_rules()
            return engine.analyze(model, config).passed

        wrapper.goodhart_check = goodhart_check
        wrapper.goodhart_passed = goodhart_passed

        return wrapper

    return decorator


def analyze_function(fn, verbose=False, print_report=False) -> Result:
    """Analyze a @reward_function-decorated function.

    Returns a Result without printing anything by default.
    Use print_report=True to print the formatted report.

    Args:
        fn: A function decorated with @reward_function.
        verbose: If True and print_report=True, includes learn_more context.
        print_report: If True, prints the formatted report to stdout.

    Returns:
        Result with verdicts, passed status, etc.

    Raises:
        AttributeError: If fn is not decorated with @reward_function.
    """
    if not hasattr(fn, "goodhart_model"):
        raise AttributeError(
            f"Function '{fn.__name__}' is not decorated with @reward_function. "
            f"Add @reward_function(sources=[...]) to annotate it."
        )
    if print_report:
        return fn.goodhart_check(verbose=verbose)
    else:
        return fn.goodhart_check(quiet=True)


def load_annotated_function(module_path: str):
    """Load and return a @reward_function-decorated function from a module path.

    Args:
        module_path: "module.path:function_name" or "path/to/file.py:function_name"

    Returns:
        The decorated function with .goodhart_model and .goodhart_config attached.
    """
    if ":" not in module_path:
        raise ValueError(
            f"Expected 'module:function' format, got '{module_path}'. Example: goodhart --check my_env:compute_reward"
        )
    module_str, func_name = module_path.rsplit(":", 1)

    # Handle file paths
    if module_str.endswith(".py"):
        import importlib.util

        spec = importlib.util.spec_from_file_location("_goodhart_target", module_str)
        if spec is None or spec.loader is None:
            raise FileNotFoundError(f"Cannot load module from: {module_str}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        import importlib

        module = importlib.import_module(module_str)

    fn = getattr(module, func_name, None)
    if fn is None:
        raise AttributeError(f"Module '{module_str}' has no attribute '{func_name}'")
    if not hasattr(fn, "goodhart_model"):
        raise AttributeError(f"Function '{func_name}' in '{module_str}' is not decorated with @reward_function.")
    return fn
