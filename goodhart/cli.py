"""goodhart -- catch reward traps before training.

  "When a measure becomes a target, it ceases to be a good measure."
  -- Charles Goodhart (1975), generalized by Marilyn Strathern (1997)

Your reward function is a measure of what you want. Your agent will
target it directly. This tool catches the moment where the measure
breaks -- degenerate equilibria, perverse incentives, and exploitable
reward structures -- before you spend compute discovering them.

Usage (interactive):
    goodhart

Usage (quick check):
    goodhart --goal 1.0 --penalty -0.01 --steps 500

Usage (in your training script):
    from goodhart import check
    check(goal=1.0, penalty=-0.01, max_steps=500)
"""

import argparse
import json
import sys
from goodhart.engine import TrainingAnalysisEngine
from goodhart.rules import RULE_COUNT
from goodhart.builders import (
    build_model_and_config,
    load_config_file,
    build_from_config_dict,
)


def _get_version():
    """Get version without circular import."""
    import importlib.metadata

    try:
        return importlib.metadata.version("goodhart")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"  # fallback for dev installs


_build_model_and_config = build_model_and_config  # backward compat alias


def preflight_check(
    goal: float = 0.0,
    penalty: float = 0.0,
    max_steps: int = 500,
    discovery_prob: float = 0.05,
    n_actors: int = 64,
    total_steps: int = 20_000_000,
    lr: float = 3e-4,
    critic_lr: float = None,
    entropy: float = 0.01,
    n_specialists: int = 1,
    routing_floor: float = 0.0,
    n_states: int = 1000,
    gamma: float = 0.99,
    name: str = "experiment",
    exit_on_critical: bool = False,
    quiet: bool = False,
    json_output: bool = False,
) -> bool:
    """Run pre-flight check. Returns True if no criticals found.

    Drop this into your training script:
        from goodhart import check
        if not check(goal=1.0, penalty=-0.01, max_steps=500):
            print("Fix reward structure before training!")
            sys.exit(1)

    Args:
        quiet: Suppress all output, just return the bool.
        json_output: Output results as JSON instead of text.
    """
    model, config = _build_model_and_config(
        goal=goal,
        penalty=penalty,
        max_steps=max_steps,
        discovery_prob=discovery_prob,
        n_actors=n_actors,
        total_steps=total_steps,
        lr=lr,
        critic_lr=critic_lr,
        entropy=entropy,
        n_specialists=n_specialists,
        routing_floor=routing_floor,
        n_states=n_states,
        gamma=gamma,
        name=name,
    )

    engine = TrainingAnalysisEngine().add_all_rules()
    result = engine.analyze(model, config)

    if json_output:
        print(json.dumps(result.to_dict(), indent=2))
    elif not quiet:
        from goodhart.fmt import (
            header,
            section,
            verdict as fmt_verdict,
            summary,
            passed_banner,
        )

        header(
            f"Pre-flight Check: {name}",
            f"goal={goal}, penalty={penalty}, steps={max_steps}, "
            f"p(goal)={discovery_prob}, actors={n_actors}",
        )

        if result.criticals:
            section("CRITICAL", len(result.criticals))
            for v in result.criticals:
                fmt_verdict(v)

        if result.warnings:
            section("WARNINGS", len(result.warnings))
            for v in result.warnings:
                fmt_verdict(v)

        if result.infos:
            section("INFO", len(result.infos))
            for v in result.infos:
                fmt_verdict(v)

        if not result.verdicts:
            passed_banner()

        summary(len(result.criticals), len(result.warnings), len(result.infos))

    if not result.passed and exit_on_critical:
        sys.exit(1)

    return result.passed


_load_config_file = load_config_file  # backward compat alias
_build_from_config_dict = build_from_config_dict  # backward compat alias


def _output_analysis(model, config, args):
    """Run analysis and output results based on --json/--quiet/default flags.

    This is the shared output dispatch used by --config and
    --goal/--penalty/--steps paths. Eliminates duplication of the
    json/quiet/print logic.
    """
    engine = TrainingAnalysisEngine().add_all_rules()

    # Apply --ignore: remove suppressed rules before analysis
    ignored = set()
    if getattr(args, "ignore", None):
        ignored = {r.strip() for r in args.ignore.split(",")}
        engine.rules = [r for r in engine.rules if r.name not in ignored]

    output_format = getattr(args, "format", "default")

    if args.json:
        result = engine.analyze(model, config)
        verbose = getattr(args, "verbose", False)
        print(json.dumps(result.to_dict(verbose=verbose), indent=2))
    elif args.quiet:
        result = engine.analyze(model, config)
    elif output_format == "compact":
        result = engine.analyze(model, config)
        for v in result.verdicts:
            severity = v.severity.value.upper()
            print(f"{severity}:{v.rule_name}: {v.message}")
    else:
        verbose = getattr(args, "verbose", False)
        result = engine.print_report(model, config, verbose=verbose)

    # Determine exit code
    strict = getattr(args, "strict", False)
    exit_on_critical = getattr(args, "exit_on_critical", False)

    if strict and (result.criticals or result.warnings):
        sys.exit(1)
    elif (args.quiet or exit_on_critical) and not result.passed:
        sys.exit(1)

    return result


def _run_doctor(args):
    """Run doctor mode: diagnose issues and suggest fixes."""
    # Build model/config from args or config file
    if args.config:
        cfg = _load_config_file(args.config)
        model, config = _build_from_config_dict(cfg, fallback_name=args.config)
    else:
        model, config = _build_model_and_config(
            goal=args.goal or 0.0,
            penalty=args.penalty or 0.0,
            max_steps=args.steps or 500,
            discovery_prob=args.discovery,
            n_actors=args.actors,
            total_steps=args.budget,
            lr=args.lr,
            critic_lr=args.critic_lr,
            entropy=args.entropy,
            n_specialists=args.specialists,
            routing_floor=args.floor,
            n_states=args.states,
            gamma=args.gamma,
            name=args.name,
        )

    engine = TrainingAnalysisEngine().add_all_rules()
    result = engine.analyze(model, config)

    from goodhart.rules.reward import _discounted_steps
    import math

    issues = result.criticals + result.warnings

    # Collect suggested fixes (shared by both text and JSON output)
    fixes = {}  # param_name -> (new_value, old_value, comment)
    issue_dicts = []  # for JSON output

    for v in issues:
        issue_entry = {"rule": v.rule_name, "message": v.message}
        if v.recommendation:
            issue_entry["fix"] = v.recommendation

        # Derive parameter fixes from known rule patterns
        if v.rule_name == "penalty_dominates_goal":
            disc_steps = _discounted_steps(model.gamma, model.max_steps)
            safe_penalty = model.max_goal_reward / disc_steps / 2
            fixes["penalty"] = (
                -safe_penalty,
                model.total_step_penalty,
                f"safe threshold: {-safe_penalty:.6f}",
            )
        elif v.rule_name == "death_beats_survival":
            fixes["penalty"] = (
                0.0,
                model.total_step_penalty,
                "remove step penalty entirely",
            )
        elif v.rule_name == "idle_exploit":
            fixes["penalty"] = (
                0.0,
                model.total_step_penalty,
                "or add action_reward > |penalty|",
            )
        elif v.rule_name == "exploration_threshold":
            # Match BudgetSufficiency's formula for consistency
            if model.death_probability > 0:
                avg_ep_len = min(1.0 / model.death_probability, model.max_steps)
            else:
                avg_ep_len = float(model.max_steps)
            avg_ep_len = max(1.0, avg_ep_len)
            best_p = (
                max(s.discovery_probability for s in model.goal_sources)
                if model.goal_sources
                else 0.0
            )
            if best_p > 0 and avg_ep_len > 0:
                # Need: total_episodes * best_p >= 10
                # total_episodes = total_steps / avg_ep_len
                min_steps = math.ceil(10 / best_p * avg_ep_len)
                fixes["budget"] = (
                    min_steps,
                    config.total_steps,
                    "minimum for ~10 goal discoveries",
                )
            else:
                fixes["budget"] = (
                    100_000_000,
                    config.total_steps,
                    "increase substantially or add intrinsic motivation",
                )
        elif v.rule_name == "expert_collapse":
            if config.num_specialists > 1:
                floor = round(1.0 / (3 * config.num_specialists), 2)
                fixes["routing_floor"] = (
                    floor,
                    config.routing_floor,
                    "minimum to prevent collapse",
                )
        elif v.rule_name == "critic_lr_ratio":
            new_clr = config.lr * 0.1
            fixes["critic_lr"] = (
                new_clr,
                config.critic_lr or config.lr,
                "10x lower than actor lr",
            )

        issue_dicts.append(issue_entry)

    # Build suggested config dict
    suggested_config = {}
    if fixes:
        params = {
            "goal": model.max_goal_reward,
            "penalty": model.total_step_penalty,
            "steps": model.max_steps,
            "lr": config.lr,
            "critic_lr": config.critic_lr or config.lr,
            "entropy": config.entropy_coeff,
            "specialists": config.num_specialists,
            "routing_floor": config.routing_floor,
            "actors": config.n_actors,
        }
        for key, val in params.items():
            if key in fixes:
                new_val, _, _ = fixes[key]
                suggested_config[key] = new_val
            else:
                suggested_config[key] = val

    # JSON output mode
    if getattr(args, "json", False):
        diagnosis = "no issues found" if not issues else f"{len(issues)} issue(s) found"
        output = {"diagnosis": diagnosis, "issues": issue_dicts}
        if suggested_config:
            output["suggested_config"] = suggested_config
        print(json.dumps(output, indent=2))
        return

    # Text output mode
    print(f"goodhart doctor: {model.name}")
    print()

    if not issues:
        print("Diagnosis: no issues found. Configuration looks good.")
        return

    print(f"Diagnosis: {len(issues)} issue(s) found")
    print()

    for i, v in enumerate(issues, 1):
        print(f"  {i}. {v.message}")
        if v.recommendation:
            print(f"     Fix: {v.recommendation}")
        print()

    # Print suggested config
    if fixes:
        print("Suggested config:")

        def _fmt(v):
            """Format values for human readability."""
            if isinstance(v, float):
                if abs(v) < 0.001 and v != 0:
                    return f"{v:.1e}"
                return f"{v:g}"
            return str(v)

        params = {
            "goal": model.max_goal_reward,
            "penalty": model.total_step_penalty,
            "steps": model.max_steps,
            "lr": config.lr,
            "critic_lr": config.critic_lr or config.lr,
            "entropy": config.entropy_coeff,
            "specialists": config.num_specialists,
            "routing_floor": config.routing_floor,
            "actors": config.n_actors,
        }
        for key, val in params.items():
            if key in fixes:
                new_val, old_val, comment = fixes[key]
                print(f"  {key}: {_fmt(new_val)}  # was: {_fmt(old_val)} ({comment})")
            else:
                print(f"  {key}: {_fmt(val)}")


def interactive():
    """Interactive mode -- ask questions, build the analysis."""
    print("goodhart -- catch reward traps before training")
    print("=" * 48)
    print()
    print("Answer a few questions about your setup.")
    print("Press Enter for defaults shown in [brackets].")
    print("(Run 'goodhart --about' to learn about Goodhart's Law)")
    print()

    def ask(prompt, default, type_fn=str):
        try:
            raw = input(f"  {prompt} [{default}]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)
        if not raw:
            return default
        return type_fn(raw)

    name = ask("Experiment name", "my_experiment")
    goal = ask("Goal/terminal reward (0 if none)", 1.0, float)
    penalty = ask("Step penalty (0 if none, negative)", 0.0, float)
    max_steps = ask("Max steps per episode", 500, int)

    if goal > 0:
        discovery = ask("P(finding goal per episode)", 0.05, float)
    else:
        discovery = 0.0

    n_actors = ask("Number of parallel actors/envs", 16, int)
    total_steps = ask("Total training steps", 10_000_000, int)
    lr = ask("Learning rate", 3e-4, float)

    critic_lr_raw = ask("Critic learning rate (Enter = same as lr)", "same", str)
    critic_lr = None if critic_lr_raw == "same" else float(critic_lr_raw)

    entropy = ask("Entropy coefficient", 0.01, float)

    n_specialists = ask("Number of specialists/networks (1=monolithic)", 1, int)

    routing_floor = 0.0
    if n_specialists > 1:
        routing_floor = ask("Routing floor (min weight per specialist)", 0.0, float)

    print()
    preflight_check(
        goal=goal,
        penalty=penalty,
        max_steps=max_steps,
        discovery_prob=discovery,
        n_actors=n_actors,
        total_steps=total_steps,
        lr=lr,
        critic_lr=critic_lr,
        entropy=entropy,
        n_specialists=n_specialists,
        routing_floor=routing_floor,
        name=name,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="goodhart",
        description=(
            "goodhart -- catch reward traps before training.\n"
            "\n"
            '"When a measure becomes a target, it ceases to be a good measure."\n'
            "-- Charles Goodhart (1975), generalized by Marilyn Strathern (1997)\n"
            "\n"
            "Your reward function is a measure of what you want. Your agent\n"
            "will target it directly -- standing still, dying fast, going in\n"
            "circles, exploiting physics. This tool catches those failure\n"
            "modes from your configuration alone, before you spend compute.\n"
            "\n"
            f"{RULE_COUNT} composable rules covering reward structure, training\n"
            "hyperparameters, architecture, and blind-spot advisories.\n"
            "Validated against 212 encodings from 133 papers (1983-2025).\n"
            "\n"
            "Takes milliseconds. Catches structural reward traps before you spend compute."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode (asks questions):
    goodhart

  Quick check (sparse reward with step penalty):
    goodhart --goal 1.0 --penalty -0.01 --steps 500

  Quick check (dense reward, no penalty -- safe):
    goodhart --goal 0 --penalty 0 --steps 500

  Full config check:
    goodhart --goal 1.0 --penalty -0.001 --steps 120 \\
      --actors 256 --budget 50000000 --lr 2e-4 \\
      --specialists 4 --floor 0.08

  From a config file (YAML, JSON, or TOML):
    goodhart --config my_experiment.yaml
    goodhart --config my_experiment.json
    goodhart --config my_experiment.toml

  In a training script:
    from goodhart import check
    if not check(goal=1.0, penalty=-0.01, max_steps=500):
        sys.exit(1)  # don't waste compute

  Browse resources:
    goodhart --about       Learn about Goodhart's Law
    goodhart --rules       List all analysis rules
    goodhart --examples    Browse cookbook examples
    goodhart --example X   Run a specific example

  Diagnose and fix:
    goodhart --doctor --goal 1.0 --penalty -0.01 --steps 500

What it catches:
  - Degenerate equilibria (standing still, dying fast)
  - Respawning reward loops (CoastRunners, Q*bert)
  - Death-as-reset exploits (Road Runner level replay)
  - Shaping reward cycles (bicycle orbiting)
  - Step penalty traps (penalty > goal)
  - Expert collapse (multi-specialist without floor)
  - Critic saturation (same lr for actor and critic)
  - Entropy collapse or explosion
  - Insufficient training budget for sparse rewards

What it can't catch:
  - Physics engine exploits (box surfing, leg hooking)
  - Adversarial action space attacks (tic-tac-toe crash)
  - Semantic specification errors (sorting by truncation)
  - Emergent multi-agent strategies

Tab completion (bash):
  eval "$(register-python-argcomplete goodhart)"
        """,
    )

    parser.add_argument(
        "--version", action="version", version=f"goodhart {_get_version()}"
    )
    parser.add_argument(
        "--about", action="store_true", help="Learn about Goodhart's Law and this tool"
    )
    parser.add_argument(
        "--fields",
        action="store_true",
        help="List all RewardSource and EnvironmentModel fields",
    )
    parser.add_argument(
        "--field",
        type=str,
        metavar="NAME",
        help="Explain a specific field (e.g. --field intentional), or --field all",
    )
    parser.add_argument("--rules", action="store_true", help="List all analysis rules")
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Browse cookbook examples from published papers",
    )
    parser.add_argument(
        "--example",
        type=str,
        metavar="NAME",
        help="Run a specific example (e.g. --example coast_runners)",
    )
    parser.add_argument(
        "--explain",
        type=str,
        metavar="RULE",
        help="Show detailed explanation for a rule (e.g. --explain idle_exploit)",
    )
    parser.add_argument(
        "--check",
        type=str,
        metavar="MODULE:FUNC",
        help="Analyze a @reward_function-decorated Python function "
        "(e.g. --check my_env:compute_reward)",
    )
    parser.add_argument(
        "--config",
        type=str,
        metavar="FILE",
        help="Load config from file (YAML, JSON, or TOML)",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Diagnose issues and suggest a fixed configuration",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress output, just set exit code (0=pass, 1=critical)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show extended explanations for each finding",
    )
    parser.add_argument(
        "--json", "-j", action="store_true", help="Output results as JSON"
    )

    parser.add_argument(
        "--goal", type=float, default=None, help="Goal/terminal reward value"
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=None,
        help="Step penalty (negative number, or 0)",
    )
    parser.add_argument("--steps", type=int, default=None, help="Max steps per episode")
    parser.add_argument(
        "--discovery",
        type=float,
        default=0.05,
        help="P(finding goal per episode) (default: 0.05)",
    )
    parser.add_argument(
        "--actors", type=int, default=64, help="Number of parallel actors (default: 64)"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=20_000_000,
        help="Total training steps (default: 20M)",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor (default: 0.99)"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--critic-lr",
        type=float,
        default=None,
        help="Critic learning rate (default: same as --lr)",
    )
    parser.add_argument(
        "--entropy", type=float, default=0.01, help="Entropy coefficient"
    )
    parser.add_argument(
        "--specialists", type=int, default=1, help="Number of specialist networks"
    )
    parser.add_argument(
        "--floor", type=float, default=0.0, help="Routing floor per specialist"
    )
    parser.add_argument(
        "--states", type=int, default=1000, help="Approximate state space size"
    )
    parser.add_argument(
        "--name", type=str, default="experiment", help="Experiment name"
    )
    parser.add_argument(
        "--exit-on-critical",
        action="store_true",
        help="Exit with code 1 if criticals found",
    )
    parser.add_argument(
        "--strict",
        "-s",
        action="store_true",
        help="Treat warnings as errors (exit code 1 on warnings too)",
    )
    parser.add_argument(
        "--ignore",
        type=str,
        metavar="RULES",
        help="Comma-separated rules to suppress (e.g. --ignore idle_exploit,reward_dominance_imbalance)",
    )
    parser.add_argument(
        "--format",
        choices=["default", "compact"],
        default="default",
        help="Output format: 'compact' for one-line-per-finding (grep-friendly)",
    )
    parser.add_argument(
        "--viz", action="store_true", help="Generate a reward landscape visualization"
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Use ASCII art instead of matplotlib for --viz",
    )
    parser.add_argument(
        "--detect",
        type=str,
        metavar="ENV_ID",
        help="Auto-detect reward structure from a Gymnasium env",
    )

    args = parser.parse_args()

    # Field reference data (shared by --fields and --field)
    _FIELD_REF = {
        # RewardSource fields
        "name": (
            "RewardSource",
            "str",
            "required",
            "A human-readable label for this reward component.",
            "Use something descriptive: 'forward_velocity', 'alive_bonus', 'collision_penalty'.",
        ),
        "reward_type": (
            "RewardSource",
            "RewardType",
            "required",
            "When this reward is given: TERMINAL (end of episode), PER_STEP (every step), ON_EVENT (when something happens), or SHAPING (guidance toward goal).",
            "Terminal for goal completion. Per-step for tracking rewards and penalties. On-event for collectibles and triggers. Shaping for distance-decrease and checkpoint guidance.",
        ),
        "value": (
            "RewardSource",
            "float",
            "required",
            "The magnitude of this reward. Positive = good, negative = penalty.",
            "Use the actual number from your reward function. The tool compares magnitudes across components to find dominance and imbalance.",
        ),
        "intentional": (
            "RewardSource",
            "bool",
            "False",
            "Is this the actual goal, or is it there to help?",
            "Set True for the thing you want the agent to accomplish (forward velocity in locomotion, goal reaching in navigation). Set False for shaping, penalties, alive bonuses, and auxiliary signals. This one flag changes which rules fire: a passive +5/step marked intentional is a survival task. Marked non-intentional, it's an idle exploit.",
        ),
        "requires_action": (
            "RewardSource",
            "bool",
            "True",
            "Does the agent need to DO something to earn this reward?",
            "Set False for alive bonuses, passive tracking, and anything earned by existing. Set True for velocity rewards, goal reaching, and anything requiring deliberate behavior. This determines the idle floor: passive rewards accumulate even when the agent does nothing.",
        ),
        "can_loop": (
            "RewardSource",
            "bool",
            "False",
            "Can the agent harvest this reward repeatedly by cycling through states?",
            "Set True for shaping rewards where the agent can move toward a target then away then toward again (distance decrease, checkpoint crossing). Set False for terminal rewards, one-time events, and potential-based shaping (which nets to zero over cycles). Triggers shaping_loop_exploit.",
        ),
        "respawn": (
            "RewardSource",
            "RespawnBehavior",
            "NONE",
            "What happens to this reward after it's collected?",
            "NONE = gone forever. TIMED = reappears after respawn_time steps. ON_DEATH = resets when the agent dies. ON_EPISODE = resets each episode. INFINITE = always available (per-step rewards). Respawning rewards can be farmed; the tool checks whether farming beats completing the task.",
        ),
        "discovery_probability": (
            "RewardSource",
            "float",
            "1.0",
            "How likely is a random agent to encounter this reward?",
            "Set 1.0 for per-step rewards the agent always sees. Set low (0.001-0.05) for sparse goals that require deliberate exploration. Drives exploration threshold analysis.",
        ),
        "state_dependent": (
            "RewardSource",
            "bool",
            "False",
            "Does the reward value change based on environment state?",
            "Set True for tracking rewards (-||error||^2), velocity rewards, and anything that varies with performance. Set False for fixed bonuses and constant penalties. Affects negative_only_reward severity.",
        ),
        "explore_fraction": (
            "RewardSource",
            "float",
            "0.0",
            "What fraction of this reward does a random agent earn?",
            "Set 0.0 if random actions produce zero reward (precise tracking). Set 0.5 if random actions earn about half (stumbling forward). Used by idle_exploit to estimate whether exploration is net-positive.",
        ),
        "respawn_time": (
            "RewardSource",
            "int",
            "0",
            "Steps until a TIMED reward respawns.",
            "Only used when respawn=TIMED. A respawn_time of 10 means the reward reappears 10 steps after collection.",
        ),
        "max_occurrences": (
            "RewardSource",
            "int",
            "1",
            "Maximum times this reward can fire per episode (0 = unlimited).",
            "Set to the number of collectibles, enemies, or events. Set 0 for unlimited per-step rewards.",
        ),
        "loop_period": (
            "RewardSource",
            "int",
            "0",
            "Steps per cycle when can_loop=True.",
            "How many steps it takes the agent to complete one loop (e.g., 2 for back-and-forth on an arrow tile).",
        ),
        # EnvironmentModel fields
        "max_steps": (
            "EnvironmentModel",
            "int",
            "500",
            "Maximum episode length in steps.",
            "Affects discount horizon analysis and penalty accumulation. Longer episodes amplify per-step rewards relative to terminal rewards.",
        ),
        "gamma": (
            "EnvironmentModel",
            "float",
            "0.99",
            "Discount factor.",
            "Lower values make the agent more myopic. At gamma=0.9, rewards 20 steps away are worth 12%. At gamma=0.99, they're worth 82%. At gamma=1.0, no discounting.",
        ),
        "n_states": (
            "EnvironmentModel",
            "int",
            "1000",
            "Approximate state space size.",
            "Affects exploration analysis and capacity checks. Atari ~100K, MuJoCo ~100K, gridworld ~100-1000.",
        ),
        "n_actions": (
            "EnvironmentModel",
            "int",
            "8",
            "Number of actions available.",
            "Atari: 18, continuous control: 2-30, gridworld: 4-8. Affects entropy and exploration analysis.",
        ),
        "death_probability": (
            "EnvironmentModel",
            "float",
            "0.01",
            "Probability of episode termination per step from agent failure.",
            "High values make death-beats-survival traps more likely. Set 0.0 for environments where the agent cannot die.",
        ),
    }

    if getattr(args, "fields", False):
        from goodhart.fmt import (
            header,
            rule_list_item,
            category_header,
            DIM_COLOR,
            RESET,
        )

        header("Fields Reference")
        print()
        for section_name in ["RewardSource", "EnvironmentModel"]:
            category_header(section_name)
            for fname, (owner, ftype, default, short, _) in _FIELD_REF.items():
                if owner == section_name:
                    rule_list_item(
                        fname, f"{short} {DIM_COLOR}[{ftype}, default={default}]{RESET}"
                    )
            print()
        print(
            f"  {DIM_COLOR}Use --field <name> for details (e.g., --field intentional){RESET}"
        )
        print(f"  {DIM_COLOR}Use --field all for the complete reference{RESET}")
        print()
        return

    if getattr(args, "field", None):
        from goodhart.fmt import (
            header,
            explain_header,
            explain_section,
            DIM_COLOR,
            RESET,
            HEADER_COLOR,
        )

        field_name = args.field

        if field_name == "all":
            header("Complete Field Reference")
            for section_name in ["RewardSource", "EnvironmentModel"]:
                print()
                print(f"  {HEADER_COLOR}{section_name}{RESET}")
                print()
                for fname, (owner, ftype, default, short, detail) in _FIELD_REF.items():
                    if owner == section_name:
                        explain_header(fname, short)
                        explain_section("Type", f"{ftype}, default={default}")
                        explain_section("Details", detail)
                        print()
            return

        if field_name not in _FIELD_REF:
            print(f"  Unknown field: {field_name}")
            print()
            print("  Available fields:")
            for fname, (owner, _, _, short, _) in _FIELD_REF.items():
                print(f"    {fname:25s} ({owner})")
            return

        owner, ftype, default, short, detail = _FIELD_REF[field_name]
        explain_header(field_name, short)
        explain_section("Type", f"{ftype}, default={default}")
        explain_section("Owner", owner)
        explain_section("Details", detail)
        print()
        return

    if args.rules:
        from goodhart.fmt import (
            header,
            category_header,
            rule_list_item,
            DIM_COLOR,
            RESET,
        )

        header("Analysis Rules")

        engine = TrainingAnalysisEngine().add_all_rules()

        categories = {
            "Reward Structure": [
                "penalty_dominates_goal",
                "death_beats_survival",
                "idle_exploit",
                "exploration_threshold",
                "respawning_exploit",
                "death_reset_exploit",
                "shaping_loop_exploit",
                "shaping_not_potential_based",
                "proxy_reward_hackability",
                "intrinsic_sufficiency",
                "budget_sufficiency",
                "compound_trap",
                "staged_reward_plateau",
                "reward_dominance_imbalance",
                "exponential_saturation",
                "intrinsic_dominance",
                "discount_horizon_mismatch",
                "negative_only_reward",
                "reward_delay_horizon",
            ],
            "Training Hyperparameters": [
                "lr_regime",
                "critic_lr_ratio",
                "entropy_regime",
                "clip_fraction_risk",
                "expert_collapse",
                "batch_size_interaction",
                "parallelism_effect",
                "memory_capacity",
                "replay_buffer_ratio",
                "target_network_update",
                "epsilon_schedule",
                "soft_update_rate",
                "sac_alpha",
            ],
            "Architecture (precedent-based)": [
                "embed_dim_capacity",
                "routing_floor_necessity",
                "recurrence_type",
                "actor_count_effect",
            ],
            "Blind-Spot Advisories (cannot check, can hint)": [
                "advisory_physics_exploit",
                "advisory_goal_misgeneralization",
                "advisory_credit_assignment",
                "advisory_constrained_rl",
                "advisory_nonstationarity",
                "advisory_learned_reward",
                "advisory_missing_constraint",
                "advisory_aggregation_trap",
            ],
        }

        rule_map = {r.name: r for r in engine.rules}
        total = 0

        for category, names in categories.items():
            category_header(category)
            for name in names:
                rule = rule_map.get(name)
                if rule:
                    rule_list_item(name, rule.description)
                    total += 1

        print()
        print(f"  {DIM_COLOR}Total: {total} rules{RESET}")
        print(
            f"  Use {RESET}--explain <rule>{DIM_COLOR} for deep explanation of any rule.{RESET}"
        )
        print()
        return

    if args.explain:
        from goodhart.rules.explanations import get_explanation

        engine = TrainingAnalysisEngine().add_all_rules()
        rule_map = {r.name: r for r in engine.rules}

        entry = get_explanation(args.explain)
        rule = rule_map.get(args.explain)

        if not rule:
            print(f"Unknown rule: {args.explain}")
            print("Use --rules to see all available rules.")
            return

        from goodhart.fmt import (
            explain_header,
            explain_section,
            RULE_COLOR,
            DIM_COLOR,
            REC_COLOR,
            HEADER_COLOR,
            RESET,
        )

        explain_header(rule.name, rule.description)

        if entry:
            if entry.get("learn_more"):
                explain_section("What this means", entry["learn_more"])

            if entry.get("examples"):
                names = ", ".join(f"{RULE_COLOR}{e}{RESET}" for e in entry["examples"])
                print(f"  {HEADER_COLOR}Examples:{RESET}  {names}")
                print(f"  {DIM_COLOR}Run: goodhart --example <name>{RESET}")
                print()

            if entry.get("papers"):
                explain_section("References", entry["papers"])

            if entry.get("see_also"):
                names = ", ".join(f"{RULE_COLOR}{r}{RESET}" for r in entry["see_also"])
                print(f"  {DIM_COLOR}See also:{RESET} {names}")
                print()

            if entry.get("proof"):
                print(f"  {REC_COLOR}Formal basis:{RESET} {entry['proof']}")
                print()
        else:
            print(f"  {DIM_COLOR}(No extended explanation available yet.){RESET}")
            print()

        if hasattr(rule, "proof") and rule.proof:
            proof = rule.proof
            if proof.statement:
                print(f"  {DIM_COLOR}Theorem: {proof.statement}{RESET}")
                print(f"  {DIM_COLOR}Strength: {proof.strength}{RESET}")
                print()

        return

    if args.examples:
        import pkgutil
        import goodhart.examples
        from goodhart.fmt import header, DIM_COLOR, RULE_COLOR, RESET, rule_list_item

        header("Cookbook Examples")
        print(f"  Run any example: {RULE_COLOR}goodhart --example <name>{RESET}")
        print()

        examples = sorted(
            m.name
            for m in pkgutil.iter_modules(goodhart.examples.__path__)
            if m.name != "__init__" and not m.name.startswith("sample")
        )
        for name in examples:
            try:
                mod = __import__(f"goodhart.examples.{name}", fromlist=[name])
                doc = (mod.__doc__ or "").strip().split("\n")[0]
                doc = doc.replace("Example: ", "").replace('"""', "")
            except Exception:
                doc = ""
            rule_list_item(name, doc, width=35)

        print()
        print(
            f"  {DIM_COLOR}{len(examples)} examples from published papers (1983-2025){RESET}"
        )
        print(
            f"  {DIM_COLOR}Includes failures, positive patterns, and limitation cases.{RESET}"
        )
        print()
        return

    if args.example:
        import importlib

        try:
            mod = importlib.import_module(f"goodhart.examples.{args.example}")
            mod.run_example()
        except ModuleNotFoundError:
            import pkgutil
            import goodhart.examples

            available = sorted(
                m.name
                for m in pkgutil.iter_modules(goodhart.examples.__path__)
                if m.name != "__init__" and not m.name.startswith("sample")
            )
            print(f"Unknown example: {args.example}")
            print()
            print("Available examples:")
            for ex in available:
                print(f"  {ex}")
            print()
            print("Run 'goodhart --examples' for descriptions.")
            sys.exit(1)
        return

    if args.detect:
        try:
            from goodhart.detect import detect_env
        except ImportError:
            print("Error: gymnasium is required for --detect.")
            print("Install it with: pip install gymnasium")
            sys.exit(1)

        try:
            model, stats = detect_env(args.detect)
        except ImportError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except RuntimeError as e:
            print(f"Error: {e}")
            sys.exit(1)

        print(f"goodhart detect: {args.detect}")
        print(f"  max_episode_steps: {stats['max_episode_steps']}")
        print(f"  n_actions:         {stats['n_actions']}")
        print(f"  n_episodes:        {stats['n_episodes']}")
        print()
        print(f"  mean_episode_length:  {stats['mean_episode_length']:.1f}")
        print(f"  mean_reward:          {stats['mean_reward']:.2f}")
        print(f"  max_reward:           {stats['max_reward']:.2f}")
        print(f"  min_reward:           {stats['min_reward']:.2f}")
        print(f"  discovery_probability:{stats['discovery_probability']:.3f}")
        print(f"  death_probability:    {stats['death_probability']:.3f}")
        print()

        # Run analysis on detected model
        engine = TrainingAnalysisEngine().add_all_rules()
        engine.print_report(model)
        return

    if args.about:
        from goodhart.fmt import (
            header,
            DIM_COLOR,
            HEADER_COLOR,
            RESET,
            RULE_COLOR,
            REC_COLOR,
            WARNING_COLOR,
        )

        header("Reward Structure Analysis for Reinforcement Learning")
        print(f'  {DIM_COLOR}"When a measure becomes a target,')
        print(f'   it ceases to be a good measure."{RESET}')
        print(f"  {DIM_COLOR}-- Charles Goodhart, 1975{RESET}")
        print()
        print("  In reinforcement learning, the reward function IS that measure.")
        print("  Your agent targets it directly. It finds ways to maximize the")
        print("  number without doing the task:")
        print()
        print(
            f"    {WARNING_COLOR}-{RESET} Standing still {DIM_COLOR}(avoids step penalties){RESET}"
        )
        print(
            f"    {WARNING_COLOR}-{RESET} Dying immediately {DIM_COLOR}(stops accumulating costs){RESET}"
        )
        print(
            f"    {WARNING_COLOR}-{RESET} Going in circles {DIM_COLOR}(harvests respawning rewards){RESET}"
        )
        print(
            f"    {WARNING_COLOR}-{RESET} Exploiting physics {DIM_COLOR}(maximizes reward through simulator bugs){RESET}"
        )
        print()
        print("  This tool catches the mathematical signatures of these failures")
        print("  from your configuration alone — before you spend compute.")
        print()
        # Dataset counts updated at release (too slow to compute from JSONL)
        proved = sum(
            1
            for r in TrainingAnalysisEngine().add_all_rules().rules
            if hasattr(r, "proof") and r.proof and r.proof.strength
        )
        print(
            f"  {HEADER_COLOR}{RULE_COUNT} rules{RESET} tested against 212 encodings from 133 papers (1983-2025)"
        )
        print(
            f"  {HEADER_COLOR}{proved} rules{RESET} linked to LEAN 4 proofs (105 theorems, zero sorry)"
        )
        print()
        print(f"  {DIM_COLOR}Cannot catch: physics exploits, goal misgeneralization,")
        print("  learned-reward gaming, missing reward terms. When config patterns")
        print(f"  match these blind spots, advisory hints are emitted.{RESET}")
        print()
        print(f"  {DIM_COLOR}Sources:{RESET}")
        print(
            f'    Goodhart, C.A.E. (1975) {DIM_COLOR}"Problems of Monetary Management"{RESET}'
        )
        print(
            f'    Ng et al. (1999) {DIM_COLOR}"Policy invariance under reward transformations"{RESET}'
        )
        print(
            f'    Skalse et al. (2022) {DIM_COLOR}"Defining and Characterizing Reward Hacking"{RESET}'
        )
        print(f"    Krakovna et al. {DIM_COLOR}Specification Gaming Master List{RESET}")
        print()
        print(f"  {REC_COLOR}https://github.com/audieleon/goodhart{RESET}")
        print()
        return

    if args.doctor:
        _run_doctor(args)
        return

    if getattr(args, "check", None):
        from goodhart.annotate import load_annotated_function

        fn = load_annotated_function(args.check)
        _output_analysis(fn.goodhart_model, fn.goodhart_config, args)
        return

    if args.config:
        try:
            cfg = _load_config_file(args.config)
        except FileNotFoundError:
            print(f"Error: config file not found: {args.config}", file=sys.stderr)
            sys.exit(1)
        except (ValueError, Exception) as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            sys.exit(1)
        if not cfg:
            print("Error: config file is empty", file=sys.stderr)
            sys.exit(1)
        model, config = _build_from_config_dict(cfg, fallback_name=args.config)
        _output_analysis(model, config, args)
        return

    # Interactive mode if no reward args given
    if args.goal is None and args.penalty is None and args.steps is None:
        if not args.quiet and not args.json and not args.viz:
            interactive()
        # quiet/json with no args: nothing to do
    else:
        model, config = _build_model_and_config(
            goal=args.goal or 0.0,
            penalty=args.penalty or 0.0,
            max_steps=args.steps or 500,
            discovery_prob=args.discovery,
            n_actors=args.actors,
            total_steps=args.budget,
            lr=args.lr,
            critic_lr=args.critic_lr,
            entropy=args.entropy,
            n_specialists=args.specialists,
            routing_floor=args.floor,
            n_states=args.states,
            gamma=args.gamma,
            name=args.name,
        )
        result = _output_analysis(model, config, args)

        if args.viz or args.ascii:
            if args.ascii:
                from goodhart.viz import reward_landscape_ascii

                print()
                print(reward_landscape_ascii(model, config, result=result))
            else:
                try:
                    from goodhart.viz import reward_landscape

                    path = reward_landscape(model, config)
                    print(f"\nReward landscape saved to: {path}")
                except ImportError as e:
                    print(f"\n{e}")
                    sys.exit(1)

        # Exit code handled by _output_analysis


if __name__ == "__main__":
    main()
