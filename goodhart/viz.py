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
    Accounts for all reward sources: passive per-step, active per-step,
    terminal, on-event, and shaping rewards.
    """
    gamma = model.gamma
    T = model.max_steps

    # Classify reward sources
    passive_perstep = 0.0  # earned by doing nothing (alive bonus)
    active_perstep = 0.0   # earned by acting (velocity, tracking)
    penalties = 0.0         # per-step costs (ctrl_cost, etc.)
    goal_reward = 0.0       # terminal/on-event goal rewards
    loop_reward = 0.0       # respawning/loopable rewards per step
    disc_prob = 0.0         # best goal discovery probability

    for s in model.reward_sources:
        if s.reward_type.name in ("TERMINAL", "ON_EVENT"):
            if s.value > 0:
                goal_reward += s.value
                disc_prob = max(disc_prob, s.discovery_probability)
        elif s.reward_type.name in ("PER_STEP", "SHAPING"):
            if not s.requires_action and s.value > 0:
                passive_perstep += s.value
            elif s.requires_action and s.intentional and s.value > 0:
                active_perstep += s.value
            elif s.value < 0:
                penalties += s.value  # negative

        # Loopable rewards (any type)
        if s.can_loop and s.value > 0:
            if s.reward_type.name in ("ON_EVENT", "SHAPING"):
                # Estimate per-step equivalent from loop period
                period = max(1, s.loop_period) if s.loop_period > 0 else 5
                loop_reward += s.value / period
            elif s.reward_type.name == "PER_STEP":
                loop_reward += s.value

    disc_full = _discounted_steps(gamma, T)
    avg_discovery = max(1, T // 2)
    gamma_disc_goal = gamma ** avg_discovery if gamma < 1.0 else 1.0

    strategies = {}

    # 1. Die immediately -- 1 step of everything, then done
    disc_1 = _discounted_steps(gamma, 1)
    strategies["die_fast"] = (passive_perstep + penalties) * disc_1

    # 2. Stand still -- collect passive rewards, pay no action costs,
    #    never reach goal. Full episode.
    strategies["stand_still"] = passive_perstep * disc_full + penalties * disc_full

    # 3. Random exploration -- random walk, might find goal
    strategies["explore_random"] = (
        (passive_perstep + penalties) * disc_full
        + disc_prob * goal_reward * gamma_disc_goal
    )

    # 4. Full exploration -- explore exhaustively, find goal at T/2
    disc_half = _discounted_steps(gamma, avg_discovery)
    strategies["explore_full"] = (
        (passive_perstep + active_perstep * 0.3 + penalties) * disc_half
        + goal_reward * gamma_disc_goal
    )

    # 5. Solve the task -- reach goal efficiently (~30% of T),
    #    earn active rewards along the way
    optimal_steps = max(1, int(T * 0.3))
    disc_opt = _discounted_steps(gamma, optimal_steps)
    gamma_disc_opt = gamma ** optimal_steps if gamma < 1.0 else 1.0
    strategies["optimal"] = (
        (passive_perstep + active_perstep + penalties) * disc_opt
        + goal_reward * gamma_disc_opt
    )

    # 6. Loop exploit -- if loopable rewards exist, farm them
    if loop_reward > 0:
        strategies["loop"] = (
            (passive_perstep + loop_reward + penalties) * disc_full
        )

    return strategies


def _classify_strategy(name: str) -> str:
    """Return color class for a strategy: 'red', 'yellow', or 'green'."""
    if name in ("die_fast", "stand_still", "loop"):
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
    import os, sys

    strategies = _compute_strategy_evs(model)
    sorted_strats = sorted(strategies.items(), key=lambda x: x[1], reverse=True)

    # Color support
    use_color = (hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
                 and not os.environ.get("NO_COLOR"))

    RED = "\033[31m" if use_color else ""
    GREEN = "\033[32m" if use_color else ""
    YELLOW = "\033[33m" if use_color else ""
    BOLD = "\033[1m" if use_color else ""
    DIM = "\033[2m" if use_color else ""
    RESET = "\033[0m" if use_color else ""

    colors = {"red": RED, "yellow": YELLOW, "green": GREEN}

    # Labels with explanations
    labels = {
        "die_fast": ("Die immediately", "terminate at step 1 to avoid penalties"),
        "stand_still": ("Stand still", "do nothing, collect passive rewards only"),
        "explore_random": ("Random exploration", "random walk, might find the goal"),
        "explore_full": ("Full exploration", "explore exhaustively, find the goal"),
        "optimal": ("Solve the task", "reach the goal efficiently"),
        "loop": ("Farm loop", "cycle through respawning rewards forever"),
    }

    # Bar rendering
    all_vals = list(strategies.values())
    v_max = max(all_vals)
    v_min = min(all_vals)
    v_range = v_max - v_min if v_max != v_min else 1.0
    bar_width = 30  # max bar chars

    winner_name = sorted_strats[0][0]
    winner_class = _classify_strategy(winner_name)

    lines = []
    lines.append("")
    lines.append(f"  {BOLD}What will your agent learn to do?{RESET}")
    lines.append(f"  {DIM}{model.name} — goal={model.max_goal_reward}, "
                 f"penalty={model.total_step_penalty}/step, "
                 f"T={model.max_steps}{RESET}")
    lines.append("")

    for name, ev in sorted_strats:
        color_class = _classify_strategy(name)
        color = colors[color_class]
        label, explanation = labels.get(name, (name, ""))
        is_winner = (name == winner_name)

        # Normalized bar length (handle negative EVs)
        if v_range > 0:
            bar_frac = (ev - v_min) / v_range
        else:
            bar_frac = 0.5
        bar_len = max(1, int(bar_frac * bar_width))

        bar = "█" * bar_len
        tag = f" ◀ agent learns this" if is_winner else ""
        bold = BOLD if is_winner else ""

        lines.append(f"  {bold}{color}{ev:+9.2f}{RESET}  "
                     f"{color}{bar}{RESET}  "
                     f"{bold}{label}{RESET}"
                     f"{color}{tag}{RESET}")
        lines.append(f"  {DIM}{'':9s}  {'':>{bar_width}s}  {explanation}{RESET}")

    lines.append("")

    # Verdict
    all_equal = (v_max - v_min) < 1e-10
    if all_equal:
        lines.append(f"  {DIM}All strategies score the same. Add a goal reward "
                     f"to differentiate intended from degenerate behavior.{RESET}")
    elif winner_class == "red":
        winner_label = labels.get(winner_name, (winner_name, ""))[0].lower()
        optimal_ev = strategies.get("optimal", 0)
        winner_ev = strategies[winner_name]
        lines.append(f"  {RED}{BOLD}Problem:{RESET} {RED}The agent will learn to "
                     f"{winner_label}{RESET}")
        lines.append(f"  {RED}because it scores {winner_ev:+.2f} vs "
                     f"{optimal_ev:+.2f} for solving the task.{RESET}")
        if winner_name == "die_fast":
            lines.append(f"  {DIM}Fix: reduce step penalty or add a survival bonus{RESET}")
        elif winner_name == "stand_still":
            lines.append(f"  {DIM}Fix: reduce passive rewards or increase active reward{RESET}")
        elif winner_name == "loop":
            lines.append(f"  {DIM}Fix: make loopable rewards non-cyclable or add a terminal goal that dominates{RESET}")
    elif winner_class == "yellow":
        lines.append(f"  {YELLOW}{BOLD}Caution:{RESET} {YELLOW}Exploration may "
                     f"outperform the intended strategy.{RESET}")
        lines.append(f"  {DIM}The agent may settle for partial progress "
                     f"instead of solving the task.{RESET}")
    else:
        lines.append(f"  {GREEN}{BOLD}Good:{RESET} {GREEN}Solving the task has "
                     f"the highest expected value.{RESET}")

    lines.append("")
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
