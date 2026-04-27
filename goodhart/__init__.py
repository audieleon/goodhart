"""goodhart -- catch reward traps before training.

"When a measure becomes a target, it ceases to be a good measure."
-- Charles Goodhart (1975), generalized by Marilyn Strathern (1997)

Usage (quick):
    from goodhart import check
    check(goal=1.0, penalty=-0.01, max_steps=500)

Usage (decorator — annotate a reward function):
    from goodhart import reward_function, RewardSource, RewardType

    @reward_function(
        max_steps=1000, gamma=0.99,
        sources=[
            RewardSource("alive", RewardType.PER_STEP, 1.0, requires_action=False),
            RewardSource("goal", RewardType.TERMINAL, 10.0, discovery_probability=0.05),
        ],
    )
    def compute_reward(obs, action, info):
        ...

    # Then:
    compute_reward.goodhart_check()          # print full report
    assert compute_reward.goodhart_passed()  # CI gate

Usage (analyze and get Result):
    from goodhart import analyze, analyze_function
    result = analyze(goal=1.0, penalty=-0.01, max_steps=500)
    result = analyze_function(compute_reward)
"""

from goodhart.cli import preflight_check as check
from goodhart.engine import AnalysisEngine, TrainingAnalysisEngine
from goodhart.presets import PRESETS
from goodhart.models import (
    EnvironmentModel,
    RewardSource,
    TrainingConfig,
    RewardType,
    RespawnBehavior,
    Severity,
    Verdict,
    Result,
)
from goodhart.annotate import reward_function, analyze_function

__version__ = "0.1.0"


def analyze(goal=0.0, penalty=0.0, max_steps=500, **kwargs) -> Result:
    """Quick analysis returning typed Result. Same args as check() but
    returns Result instead of bool, and doesn't print."""
    from goodhart.builders import build_model_and_config

    model, config = build_model_and_config(
        goal=goal, penalty=penalty, max_steps=max_steps, **kwargs,
    )
    engine = TrainingAnalysisEngine().add_all_rules()
    return engine.analyze(model, config)


__all__ = [
    "check",
    "analyze",
    "analyze_function",
    "reward_function",
    "EnvironmentModel",
    "RewardSource",
    "TrainingConfig",
    "RewardType",
    "RespawnBehavior",
    "Severity",
    "Verdict",
    "Result",
    "AnalysisEngine",
    "TrainingAnalysisEngine",
    "PRESETS",
    "__version__",
]
