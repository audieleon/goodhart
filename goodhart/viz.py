"""Reward landscape visualization.

Generates charts showing the expected value of different agent strategies
given an EnvironmentModel's reward structure. Helps visualize why
degenerate strategies can dominate intended behavior.
"""

from typing import Dict, List, Optional, Tuple

from goodhart.models import EnvironmentModel, RewardType, TrainingConfig


# =====================================================================
# Strategy EV calculations
# =====================================================================

def _discounted_steps(gamma: float, n: int) -> float:
    """Sum of discounted steps (local copy to avoid circular import)."""
    if gamma >= 1.0 or abs(1.0 - gamma) < 1e-12:
        return float(n)
    return (1.0 - gamma ** n) / (1.0 - gamma)


def _compute_strategy_evs(model: EnvironmentModel) -> Dict[str, float]:
    """Compute expected value for canonical agent strategies.

    Returns a dict mapping strategy name to expected total episode return.
    Uses discounted EVs consistent with the analysis rules.
    """
    step_penalty = model.total_step_penalty  # negative or zero
    goal_reward = model.max_goal_reward      # positive or zero
    gamma = model.gamma

    # Goal discovery probability (use first goal source, or 0)
    goal_sources = model.goal_sources
    disc_prob = goal_sources[0].discovery_probability if goal_sources else 0.0

    strategies = {}

    # 1. Die fast -- agent terminates as quickly as possible (1 step)
    strategies["die_fast"] = step_penalty * _discounted_steps(gamma, 1)

    # 2. Stand still -- agent does nothing, accumulates step penalty
    #    but never reaches goal. Survives full episode.
    disc_full = _discounted_steps(gamma, model.max_steps)
    strategies["stand_still"] = step_penalty * disc_full

    # 3. Explore (random) -- agent explores randomly
    #    Has discovery_probability chance of finding goal, pays full penalty
    avg_discovery = max(1, model.max_steps // 2)
    gamma_disc_goal = gamma ** avg_discovery if gamma < 1.0 else 1.0
    strategies["explore_random"] = (
        step_penalty * disc_full + disc_prob * goal_reward * gamma_disc_goal
    )

    # 4. Explore (full) -- agent explores exhaustively
    #    Assumes goal IS found at avg_steps/2
    disc_half = _discounted_steps(gamma, avg_discovery)
    strategies["explore_full"] = (
        step_penalty * disc_half + goal_reward * gamma_disc_goal
    )

    # 5. Optimal -- agent reaches goal efficiently (~30% of max_steps)
    optimal_steps = max(1, int(model.max_steps * 0.3))
    disc_opt = _discounted_steps(gamma, optimal_steps)
    gamma_disc_opt = gamma ** optimal_steps if gamma < 1.0 else 1.0
    strategies["optimal"] = (
        step_penalty * disc_opt + goal_reward * gamma_disc_opt
    )

    return strategies


def _classify_strategy(name: str) -> str:
    """Return color class for a strategy: 'red', 'yellow', or 'green'."""
    if name in ("die_fast", "stand_still"):
        return "red"
    elif name in ("explore_random", "explore_full"):
        return "yellow"
    else:
        return "green"


# =====================================================================
# ASCII visualization
# =====================================================================

def reward_landscape_ascii(model: EnvironmentModel,
                           config: TrainingConfig = None) -> str:
    """Generate an ASCII reward landscape chart.

    Args:
        model: The environment model to visualize.
        config: Optional training config (unused currently).

    Returns:
        A multi-line string with the ASCII chart.
    """
    strategies = _compute_strategy_evs(model)

    # Sort by EV for layout
    sorted_strats = sorted(strategies.items(), key=lambda x: x[1], reverse=True)

    # Find range for Y-axis
    all_vals = list(strategies.values())
    y_max = max(all_vals)
    y_min = min(all_vals)

    # Add some padding
    y_range = y_max - y_min if y_max != y_min else 1.0
    y_max += y_range * 0.1
    y_min -= y_range * 0.1

    # Build lines
    lines = []
    lines.append(f"Reward Landscape: {model.name}")
    lines.append(f"  goal={model.max_goal_reward}, "
                 f"penalty={model.total_step_penalty}/step, "
                 f"max_steps={model.max_steps}")
    lines.append("")

    # Label mapping for display
    labels = {
        "die_fast": "die fast",
        "stand_still": "stand still",
        "explore_random": "explore (random)",
        "explore_full": "explore (full)",
        "optimal": "optimal",
    }

    # Markers for degenerate vs intended
    markers = {
        "red": "XXXX",
        "yellow": "????",
        "green": ">>>>",
    }

    # Find the winner
    winner_name = sorted_strats[0][0]

    for name, ev in sorted_strats:
        color_class = _classify_strategy(name)
        marker = markers[color_class]
        label = labels.get(name, name)
        win_tag = " <-- WINS" if name == winner_name else ""
        lines.append(f"  {ev:+8.2f} | {marker} {label}{win_tag}")

    lines.append("")

    # Legend
    lines.append("  >>>> = intended    ???? = marginal    XXXX = degenerate")

    # Warning if degenerate wins
    winner_class = _classify_strategy(winner_name)
    if winner_class == "red":
        lines.append("")
        lines.append("  WARNING: A degenerate strategy has highest EV!")
        lines.append(f"  The agent will learn to '{labels.get(winner_name, winner_name)}'")
        lines.append("  instead of solving the task.")
    elif winner_class == "yellow":
        lines.append("")
        lines.append("  CAUTION: A marginal strategy ties or beats optimal.")
        lines.append("  The agent may settle for partial exploration.")

    return "\n".join(lines)


# =====================================================================
# Matplotlib visualization
# =====================================================================

def reward_landscape(model: EnvironmentModel,
                     config: TrainingConfig = None,
                     output: str = "landscape.png") -> str:
    """Generate a reward landscape chart as a PNG file.

    Args:
        model: The environment model to visualize.
        config: Optional training config (unused currently).
        output: Output file path for the chart.

    Returns:
        The output file path.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        raise ImportError(
            "matplotlib is required for graphical visualization.\n"
            "Install it with: pip install goodhart[viz]"
        )

    strategies = _compute_strategy_evs(model)
    sorted_strats = sorted(strategies.items(), key=lambda x: x[1], reverse=True)

    # Labels and colors
    labels_map = {
        "die_fast": "Die Fast",
        "stand_still": "Stand Still",
        "explore_random": "Explore (Random)",
        "explore_full": "Explore (Full)",
        "optimal": "Optimal",
    }
    color_map = {
        "red": "#e74c3c",
        "yellow": "#f39c12",
        "green": "#27ae60",
    }

    names = [labels_map.get(s[0], s[0]) for s in sorted_strats]
    values = [s[1] for s in sorted_strats]
    colors = [color_map[_classify_strategy(s[0])] for s in sorted_strats]

    # Find winner
    winner_idx = 0  # already sorted descending

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(range(len(names)), values, color=colors, height=0.6,
                   edgecolor="black", linewidth=0.5)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=12)
    ax.set_xlabel("Expected Value (total episode return)", fontsize=12)
    ax.set_title(f"Reward Landscape: {model.name}", fontsize=14, fontweight="bold")

    # Annotate values
    for i, (bar, val) in enumerate(zip(bars, values)):
        x_pos = val + (max(values) - min(values)) * 0.02 if val >= 0 else val - (max(values) - min(values)) * 0.02
        ha = "left" if val >= 0 else "right"
        weight = "bold" if i == winner_idx else "normal"
        ax.annotate(f"{val:+.2f}", (x_pos, i), va="center", ha=ha,
                    fontsize=10, fontweight=weight)

    # Highlight winner
    ax.get_yticklabels()[winner_idx].set_fontweight("bold")

    # Add zero line
    ax.axvline(x=0, color="gray", linewidth=0.8, linestyle="--")

    # Legend
    legend_patches = [
        mpatches.Patch(color="#27ae60", label="Intended"),
        mpatches.Patch(color="#f39c12", label="Marginal"),
        mpatches.Patch(color="#e74c3c", label="Degenerate"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=10)

    # Subtitle with config info
    subtitle = (f"goal={model.max_goal_reward}, "
                f"penalty={model.total_step_penalty}/step, "
                f"max_steps={model.max_steps}")
    ax.text(0.5, -0.12, subtitle, transform=ax.transAxes,
            ha="center", fontsize=10, color="gray")

    plt.tight_layout()
    import os
    if os.path.exists(output):
        print(f"  (overwriting existing {output})")
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output
