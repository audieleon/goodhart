"""Reward structure analysis rules.

Rules that analyze the MDP's reward dynamics to detect
degenerate equilibria and perverse incentives.
"""

import math

from goodhart.models import (
    EnvironmentModel, FormalBasis, ProofStrength, RewardType, RespawnBehavior,
    Severity, Verdict,
)
from goodhart.engine import Rule


def _worst_case_value(source) -> float:
    """Get the worst-case (most negative or least positive) value for a source.
    Uses value_range min if available, otherwise the point estimate."""
    if source.value_range:
        return min(source.value_range)
    return source.value


def _best_case_value(source) -> float:
    """Get the best-case (most positive) value for a source.
    Uses value_range max if available, otherwise the point estimate."""
    if source.value_range:
        return max(source.value_range)
    return source.value


def _discounted_steps(gamma: float, n: int) -> float:
    """Sum of discounted steps: sum_{t=0}^{n-1} gamma^t.

    For gamma < 1 this is the geometric series (1 - gamma^n) / (1 - gamma).
    For gamma == 1 (or very close), returns n to avoid division by zero.
    """
    if gamma >= 1.0 or abs(1.0 - gamma) < 1e-12:
        return float(n)
    return (1.0 - gamma ** n) / (1.0 - gamma)


class PenaltyDominatesGoal(Rule):
    """Check if cumulative step penalty exceeds goal reward."""

    @property
    def name(self): return "penalty_dominates_goal"

    @property
    def description(self):
        return ("Total step penalty over max episode exceeds goal reward, "
                "making even optimal play potentially negative")

    @property
    def proof(self):
        return FormalBasis(
            proof_name="penalty_breakeven_discounted",
            strength=ProofStrength.VERIFIED,
            statement="∀ g > 0, p < 0, D > 0, g < -p * D → g + p * D < 0 (D = discounted steps)",
            parameters={"goal": "g", "step_penalty": "p", "steps": "n"},
        )

    def applies_to(self, model):
        return model.total_step_penalty < 0 and model.max_goal_reward > 0

    def check(self, model, config=None):
        verdicts = []
        # Use worst-case values when ranges are available
        penalty_sources = [s for s in model.reward_sources
                           if s.reward_type == RewardType.PER_STEP and s.value < 0]
        penalty_per_step = sum(abs(_worst_case_value(s)) for s in penalty_sources) \
            if penalty_sources else abs(model.total_step_penalty)
        total_penalty = penalty_per_step * _discounted_steps(model.gamma, model.max_steps)
        goal = model.max_goal_reward

        if total_penalty > goal:
            verdicts.append(Verdict(
                rule_name=self.name,
                severity=Severity.CRITICAL,
                message=(f"Total discounted penalty ({total_penalty:.2f}) > goal reward "
                         f"({goal:.2f}). Even optimal play yields negative "
                         f"return if it takes >{int(goal / penalty_per_step)} steps."),
                details={"total_penalty": total_penalty, "goal_reward": goal,
                         "breakeven_steps": goal / penalty_per_step},
                recommendation=(f"Reduce step penalty to "
                                f"{goal / _discounted_steps(model.gamma, model.max_steps) / 2:.6f} or less"),
                learn_more=(
                    f"The agent sees: every step costs {penalty_per_step:.4f}, reaching the "
                    f"goal pays {goal:.2f}. If the optimal path is >{int(goal / penalty_per_step)} "
                    f"steps, the agent LOSES reward by succeeding. The rational response "
                    f"is to minimize steps — which often means doing nothing or dying early.\n"
                    f"Classic example: Mountain Car with -1/step and +1 goal over 200 steps. "
                    f"Every trajectory scores between -200 (timeout) and -199 (instant goal). "
                    f"All are negative, so the agent has no incentive to learn.\n"
                    f"Fix: either reduce the penalty so the goal is worth reaching, increase "
                    f"the goal reward, or add potential-based shaping (Ng 1999) to create "
                    f"gradient signal without changing the optimal policy.\n"
                    f"Formally verified: LEAN proof penalty_breakeven_discounted."
                ),
            ))
        elif total_penalty > goal * 0.5:
            verdicts.append(Verdict(
                rule_name=self.name,
                severity=Severity.WARNING,
                message=(f"Total discounted penalty ({total_penalty:.2f}) is >{total_penalty/goal*100:.0f}% "
                         f"of goal reward ({goal:.2f}). Tight margin."),
                details={"ratio": total_penalty / goal},
                recommendation=(f"The margin between penalty and goal is thin. An agent that "
                                f"takes a slightly suboptimal path may earn negative return. "
                                f"Consider reducing the penalty or increasing the goal reward."),
            ))
        return verdicts


class DeathBeatsSurvival(Rule):
    """Check if dying is more rewarding than surviving."""

    @property
    def name(self): return "death_beats_survival"

    @property
    def description(self):
        return "Dying early has higher EV than surviving, creating suicidal incentive"

    @property
    def proof(self):
        return FormalBasis(
            proof_name="death_beats_survival_discounted",
            strength=ProofStrength.VERIFIED,
            statement="∀ p < 0, ∀ fm < fn, p * fn < p * fm (fm, fn = discounted steps)",
            parameters={"step_penalty": "p", "early_step": "m", "late_step": "n"},
        )

    def applies_to(self, model):
        # Only applies when death is possible AND there's a passive step penalty.
        # Without death, the comparison is meaningless (reward desert, not trap).
        # Without passive penalty, a dead/idle agent pays nothing.
        passive_penalty = sum(s.value for s in model.reward_sources
                              if s.reward_type == RewardType.PER_STEP
                              and s.value < 0
                              and not s.requires_action)
        return passive_penalty < 0 and model.death_probability > 0

    def check(self, model, config=None):
        verdicts = []
        # Net per-step reward for a PASSIVE agent (not taking actions).
        # Only count rewards that don't require action — a dead/idle agent
        # doesn't receive action-dependent bonuses like movement rewards.
        passive_positive = sum(s.value for s in model.reward_sources
                               if s.reward_type == RewardType.PER_STEP
                               and s.value > 0
                               and not s.requires_action)
        net_per_step = model.total_step_penalty + passive_positive
        # If net per-step is positive even passively, surviving beats dying
        if net_per_step >= 0:
            return verdicts
        # Compare dying at step 1 vs surviving a short horizon.
        # The 10-step horizon is a heuristic: long enough to show the
        # penalty accumulation pattern, short enough to be relevant
        # even for short episodes. Not formally justified.
        survive_steps = min(10, model.max_steps)
        ev_die = net_per_step * _discounted_steps(model.gamma, 1)
        ev_survive = net_per_step * _discounted_steps(model.gamma, survive_steps)

        if ev_die == 0:
            return verdicts

        # ratio > 1 means surviving accumulates more penalty than dying.
        # Threshold 2.0 for CRITICAL: dying saves >50% of accumulated cost.
        # Threshold 1.0 for WARNING: any suicidal incentive at all.
        # These thresholds are calibrated against presets, not formally derived.
        ratio = ev_survive / ev_die

        if ratio > 2.0:
            verdicts.append(Verdict(
                rule_name=self.name,
                severity=Severity.CRITICAL,
                message=(f"Dying at step 1 ({ev_die:+.4f}) beats "
                         f"surviving {survive_steps} steps ({ev_survive:+.4f}) by "
                         f"{ratio:.1f}x. Agent will learn to die immediately."),
                details={"ev_die": ev_die, "ev_survive_10": ev_survive,
                         "ratio": ratio},
                recommendation="Remove step penalty or add survival reward",
                learn_more=(
                    "The agent discovers that dying is cheaper than living. With a "
                    "negative per-step reward and no compensating positive reward for "
                    "staying alive, every additional step of survival makes the total "
                    "return worse. The optimal policy becomes: die as fast as possible.\n"
                    "This is NOT the same as 'reward desert' (where all strategies are "
                    "equal). Here, the agent actively learns a WORSE behavior than random.\n"
                    "Classic example: CartPole with -1/step penalty and no +1 alive bonus "
                    "learns to drop the pole immediately (Sutton & Barto 2018).\n"
                    "Fix: add an alive bonus that exceeds the step penalty, or remove the "
                    "step penalty entirely. If you need time pressure, use a terminal "
                    "time bonus (reward = 1 - steps/max_steps) instead of per-step cost.\n"
                    "Formally verified: LEAN proof death_beats_survival_discounted."
                ),
            ))
        elif ratio > 1.0:
            verdicts.append(Verdict(
                rule_name=self.name,
                severity=Severity.WARNING,
                message=(f"Dying at step 1 ({ev_die:+.4f}) beats "
                         f"surviving {survive_steps} steps ({ev_survive:+.4f}) by "
                         f"{ratio:.1f}x. Marginal suicidal incentive."),
                details={"ev_die": ev_die, "ev_survive_10": ev_survive,
                         "ratio": ratio},
                recommendation="Consider removing step penalty or adding survival reward",
            ))
        return verdicts


class IdleExploit(Rule):
    """Check if doing nothing is optimal."""

    @property
    def name(self): return "idle_exploit"

    @property
    def description(self):
        return "Standing still (no-op) has higher EV than any active strategy"

    @property
    def proof(self):
        return FormalBasis(
            proof_name="idle_dominance_with_explore",
            strength=ProofStrength.VERIFIED,
            statement=("∀ r_idle ≥ 0, f ∈ [0,1], r_idle ≥ f * r_intentional + r_nonintentional + penalty "
                       "→ r_idle * T ≥ (f * r_intentional + r_nonintentional + penalty) * T"),
            parameters={"idle_reward": "r_idle", "intentional": "r_intentional",
                        "nonintentional": "r_nonintentional", "penalty": "penalty",
                        "explore_fraction": "f"},
        )

    def applies_to(self, model):
        return any(not s.requires_action for s in model.reward_sources) or \
               model.total_step_penalty < 0

    def check(self, model, config=None):
        verdicts = []
        idle_sources = [s for s in model.reward_sources
                        if not s.requires_action
                        and s.reward_type == RewardType.PER_STEP]
        # If all positive idle sources are intentional (e.g., survival reward),
        # getting reward for existing IS the design. Not an exploit.
        # But only skip if there ARE positive idle sources — an empty list
        # means "no passive income" not "all passive income is intentional."
        positive_idle = [s for s in idle_sources if s.value > 0]
        if positive_idle and all(s.intentional for s in positive_idle):
            return verdicts
        disc = _discounted_steps(model.gamma, model.max_steps)
        # Use best-case values for idle (worst case for the designer)
        ev_idle = sum(_best_case_value(s) for s in idle_sources) * disc

        # Explore EV: non-intentional per-step rewards at full value,
        # plus intentional rewards scaled by explore_fraction (how much
        # random exploration earns). Default explore_fraction=0.0 is
        # conservative (assumes random exploration earns nothing from
        # intentional rewards like velocity tracking).
        explore_per_step = 0.0
        for s in model.reward_sources:
            if s.reward_type != RewardType.PER_STEP:
                continue
            if s.intentional:
                explore_per_step += s.value * s.explore_fraction
            else:
                explore_per_step += s.value
        ev_explore = explore_per_step * disc
        # Discount goal reward to expected discovery time (avg_steps/2)
        avg_discovery = max(1, model.max_steps // 2)
        gamma_discount = model.gamma ** avg_discovery if model.gamma < 1.0 else 1.0
        for s in model.goal_sources:
            ev_explore += s.value * s.discovery_probability * gamma_discount

        if ev_idle > ev_explore and ev_idle >= 0:
            verdicts.append(Verdict(
                rule_name=self.name,
                severity=Severity.CRITICAL,
                message=(f"Standing still (EV={ev_idle:+.4f}) beats "
                         f"exploration (EV={ev_explore:+.4f}). "
                         f"Agent will learn to do nothing."),
                details={"ev_idle": ev_idle, "ev_explore": ev_explore},
                recommendation="Add idle penalty or remove step penalty",
                learn_more=(
                    "The agent compares two strategies: (1) do nothing and collect "
                    "passive rewards, (2) explore and pay action costs. If standing "
                    "still earns more, the agent converges to the no-op policy.\n"
                    "This is the most common reward design failure in locomotion tasks. "
                    "MuJoCo Humanoid-v4 with healthy_reward=5.0 earns 5000/episode by "
                    "standing still vs ~5500 for walking — not worth the fall risk.\n"
                    "Three fixes: (1) Remove the passive reward (alive bonus) entirely. "
                    "(2) Make the passive reward smaller than the active reward. "
                    "(3) Add an idle penalty that makes standing still costly.\n"
                    "The explore_fraction field on intentional rewards controls how much "
                    "credit random exploration gets. If your locomotion reward gives "
                    "partial credit for random stumbling (BipedalWalker), set "
                    "explore_fraction=0.5 to reflect this."
                ),
            ))
        return verdicts


class ExplorationThreshold(Rule):
    """Check if exploration can bootstrap learning."""

    @property
    def name(self): return "exploration_threshold"

    @property
    def description(self):
        return ("Minimum goal discovery rate needed for exploration "
                "to beat degenerate strategies")

    @property
    def proof(self):
        return FormalBasis(
            proof_name="exploration_threshold",
            strength=ProofStrength.GROUNDED,
            statement="∀ goal > 0, penalty < 0, p * (goal + penalty * avg) + (1-p) * (penalty * avg) > 0 → p * goal + penalty * avg > 0",
            parameters={"goal": "goal", "penalty": "penalty", "discovery_prob": "p"},
        )

    def applies_to(self, model):
        return len(model.goal_sources) > 0

    def check(self, model, config=None):
        verdicts = []

        ev_die = model.total_step_penalty * _discounted_steps(model.gamma, 1)
        # Idle EV includes passive (no-action) per-step rewards
        passive_per_step = sum(s.value for s in model.reward_sources
                               if s.reward_type == RewardType.PER_STEP
                               and not s.requires_action
                               and not s.intentional)
        idle_rate = model.total_step_penalty + passive_per_step
        disc_idle = _discounted_steps(model.gamma, model.max_steps)
        ev_idle = idle_rate * disc_idle

        best_degenerate = max(ev_die, ev_idle, 0.0)

        goal = model.max_goal_reward
        avg_steps = (model.max_steps) / 2
        penalty_per_step = model.total_step_penalty

        disc_avg = _discounted_steps(model.gamma, int(avg_steps))
        disc_max = _discounted_steps(model.gamma, model.max_steps)

        denom = goal + penalty_per_step * (disc_avg - disc_max)
        if denom <= 0:
            p_min = float('inf')
        else:
            numer = best_degenerate - penalty_per_step * disc_max
            p_min = max(0, numer / denom)

        # Estimate random exploration coverage.
        # sqrt(T) is the expected unique positions for a 1D random walk
        # over T steps (well-known result). The *2 factor is a rough
        # adjustment for 2D+ environments where coverage is higher.
        # This is a conservative lower bound — real environments may
        # have much higher or lower coverage depending on connectivity,
        # dimensionality, and action space. For continuous state spaces,
        # this estimate is not meaningful (n_states should reflect the
        # effective discretization, not the true state space size).
        unique_states = min(math.sqrt(model.max_steps) * 2, model.n_states)
        p_actual = unique_states / model.n_states

        # Determine if this is a "desert" (no gradient) or "trap" (perverse incentive)
        is_desert = (model.death_probability == 0 and
                     best_degenerate <= 0 and penalty_per_step < 0)

        if p_min > 1.0:
            if is_desert:
                verdicts.append(Verdict(
                    rule_name=self.name,
                    severity=Severity.CRITICAL,
                    message="Reward desert: all non-goal strategies score equally. "
                            "No gradient signal — agent cannot distinguish good from "
                            "bad exploration. This is an exploration problem, not a "
                            "reward trap.",
                    details={"p_min": p_min, "p_actual": p_actual, "type": "desert"},
                    recommendation="Add reward shaping (potential-based per Ng 1999), "
                                   "intrinsic motivation (RND/curiosity), or curriculum",
                ))
            else:
                verdicts.append(Verdict(
                    rule_name=self.name,
                    severity=Severity.CRITICAL,
                    message="Exploration can NEVER beat degenerate strategies "
                            "under current reward structure.",
                    details={"p_min": p_min, "p_actual": p_actual, "type": "trap"},
                    recommendation="Restructure reward: add shaping or remove penalty",
                ))
        elif p_actual < p_min:
            verdicts.append(Verdict(
                rule_name=self.name,
                severity=Severity.WARNING,
                message=(f"Need p(goal)>{p_min:.4f} ({p_min*100:.1f}%) but "
                         f"random walk achieves ~{p_actual:.4f} ({p_actual*100:.1f}%). "
                         f"Agent cannot bootstrap learning."),
                details={"p_min": p_min, "p_actual": p_actual,
                         "gap_ratio": p_min / max(p_actual, 1e-10)},
                recommendation="Add intrinsic motivation or reduce step penalty",
            ))
        else:
            verdicts.append(Verdict(
                rule_name=self.name,
                severity=Severity.INFO,
                message=(f"Exploration viable: p(goal)>{p_min:.4f} needed, "
                         f"random walk achieves ~{p_actual:.4f}."),
                details={"p_min": p_min, "p_actual": p_actual},
            ))
        return verdicts


class RespawningExploit(Rule):
    """Check if respawning reward sources create loop exploits."""

    @property
    def name(self): return "respawning_exploit"

    @property
    def description(self):
        return ("Respawning reward sources can be harvested in loops, "
                "giving higher EV than reaching the goal")

    @property
    def proof(self):
        return FormalBasis(
            proof_name="loop_dominance",
            strength=ProofStrength.VERIFIED,
            statement="∀ v > 0, g > 0, t > 0, v * T > g * t → v * T / t > g",
            parameters={"reward_value": "v", "goal": "g", "loop_period": "t"},
        )

    def applies_to(self, model):
        return len(model.loopable_sources) > 0

    def check(self, model, config=None):
        verdicts = []
        goal = model.max_goal_reward

        for source in model.loopable_sources:
            if source.can_loop:
                cycles = model.max_steps / max(source.loop_period, 1)
                ev_loop = source.value * cycles
            elif source.respawn == RespawnBehavior.TIMED:
                cycles = model.max_steps / max(source.respawn_time, 1)
                ev_loop = source.value * cycles
            elif source.respawn == RespawnBehavior.INFINITE:
                cycles = model.max_steps
                ev_loop = source.value * cycles
            else:
                continue

            # Respect max_occurrences cap (0 = unlimited).
            # Only apply when the source uses timed respawn (not explicit
            # can_loop or infinite), since can_loop=True and infinite
            # sources are designed for repeated collection.
            if (source.max_occurrences > 0
                    and not source.can_loop
                    and source.respawn != RespawnBehavior.INFINITE):
                cycles = min(cycles, source.max_occurrences)
                ev_loop = source.value * cycles

            if ev_loop > goal and ev_loop > 0:
                # Small-value infinite rewards (e.g., intrinsic motivation at
                # 0.01/step) are often designed to dominate sparse goals early
                # in training. Downgrade to INFO when the per-step value is
                # tiny and other positive reward sources exist alongside it.
                other_positive = any(s.value > 0 and s is not source
                                     for s in model.reward_sources)
                is_small_value = source.value < 0.1 and other_positive
                severity = Severity.INFO if is_small_value else Severity.CRITICAL
                verdicts.append(Verdict(
                    rule_name=self.name,
                    severity=severity,
                    message=(f"Looping '{source.name}' (EV={ev_loop:+.1f}) "
                             f"beats goal reward ({goal:+.1f}). "
                             + ("This may be intentional (e.g., intrinsic "
                                "motivation designed to dominate early training)."
                                if is_small_value else
                                "Agent will loop instead of completing the task.")),
                    details={"source": source.name, "ev_loop": ev_loop,
                             "goal_reward": goal, "cycles": cycles},
                    recommendation=(
                        f"If '{source.name}' is intrinsic motivation, this is expected — "
                        f"verify it decays as the agent learns. Otherwise, cap at "
                        f"{int(goal / max(source.value, 0.001))} occurrences or make "
                        f"non-respawning."
                        if is_small_value else
                        f"Cap '{source.name}' at "
                        f"{int(goal / max(source.value, 0.001))} occurrences "
                        f"or make non-respawning"
                    ),
                ))
        return verdicts


class DeathResetExploit(Rule):
    """Check if dying resets reward sources for re-collection."""

    @property
    def name(self): return "death_reset_exploit"

    @property
    def description(self):
        return ("Dying resets collectible rewards, making deliberate death "
                "more valuable than continuing")

    @property
    def proof(self):
        return FormalBasis(
            proof_name="death_reset_dominance",
            strength=ProofStrength.VERIFIED,
            statement="∀ c > 0, p > 0, g > 0, L > 0, c * p * T > g * L → c * p * T / L > g",
            parameters={"collectible_value": "c", "collect_prob": "p", "goal": "g", "avg_life": "L"},
        )

    def applies_to(self, model):
        return len(model.resettable_sources) > 0

    def check(self, model, config=None):
        verdicts = []

        resettable_value = sum(s.value * s.discovery_probability
                               for s in model.resettable_sources if s.value > 0)

        if resettable_value <= 0:
            return verdicts

        # Derive average life length from death probability
        if model.death_probability > 0:
            avg_steps_per_life = min(1.0 / model.death_probability, model.max_steps)
        else:
            avg_steps_per_life = model.max_steps
        n_lives = model.max_steps / max(avg_steps_per_life, 1)
        ev_replay = resettable_value * n_lives

        goal = model.max_goal_reward

        if ev_replay > goal:
            verdicts.append(Verdict(
                rule_name=self.name,
                severity=Severity.CRITICAL,
                message=(f"Die-and-replay strategy (EV={ev_replay:+.1f}) "
                         f"beats goal ({goal:+.1f}). Agent will deliberately "
                         f"die to re-collect rewards."),
                details={"ev_replay": ev_replay, "n_lives": n_lives,
                         "resettable_value": resettable_value},
                recommendation="Don't reset reward sources on death",
            ))
        return verdicts


class ShapingLoopExploit(Rule):
    """Check if shaping rewards form exploitable cycles."""

    @property
    def name(self): return "shaping_loop_exploit"

    @property
    def description(self):
        return ("Shaping rewards (e.g. distance decrease) can be harvested "
                "by cycling through states without reaching the goal")

    @property
    def proof(self):
        return FormalBasis(
            proof_name="ng_shaping_preserves_optimal",
            strength=ProofStrength.MOTIVATED,
            paper="Ng, Harada & Russell 1999",
            statement="∀ M Φ s a₁ a₂, Q*_M(s,a₁) ≤ Q*_M(s,a₂) ↔ Q*_{M'}(s,a₁) ≤ Q*_{M'}(s,a₂) — potential-based shaping preserves optimal policy",
            parameters={"potential": "Φ", "mdp": "M"},
        )

    def applies_to(self, model):
        return len(model.shaping_sources) > 0

    def check(self, model, config=None):
        verdicts = []
        goal = model.max_goal_reward

        for source in model.shaping_sources:
            if source.can_loop:
                cycles = model.max_steps / max(source.loop_period, 1)
                ev_loop = source.value * cycles

                if ev_loop > goal:
                    verdicts.append(Verdict(
                        rule_name=self.name,
                        severity=Severity.CRITICAL,
                        message=(f"Cycling '{source.name}' "
                                 f"(EV={ev_loop:+.1f}) beats goal "
                                 f"({goal:+.1f}). Agent will orbit "
                                 f"instead of completing the task."),
                        details={"source": source.name, "ev_loop": ev_loop,
                                 "goal_reward": goal},
                        recommendation=(f"Use potential-based shaping for "
                                        f"'{source.name}' (Ng et al. 1999) "
                                        f"to guarantee policy invariance"),
                    ))
        return verdicts


class IntrinsicSufficiency(Rule):
    """Check if intrinsic reward overcomes step penalty."""

    @property
    def name(self): return "intrinsic_sufficiency"

    @property
    def description(self):
        return "Intrinsic motivation must exceed step penalty on novel states"

    @property
    def proof(self):
        return FormalBasis(
            proof_name="intrinsic_insufficient",
            strength=ProofStrength.VERIFIED,
            statement="∀ intrinsic > 0, penalty > 0, intrinsic < penalty → intrinsic - penalty < 0",
            parameters={"intrinsic_per_step": "intrinsic", "penalty_per_step": "penalty"},
        )

    def applies_to(self, model):
        return any(s.reward_type == RewardType.PER_STEP and s.value > 0
                   for s in model.reward_sources)

    def check(self, model, config=None):
        verdicts = []
        intrinsic_sources = [s for s in model.reward_sources
                             if s.reward_type == RewardType.PER_STEP and s.value > 0]
        intrinsic_per_step = sum(s.value for s in intrinsic_sources)
        penalty_per_step = abs(model.total_step_penalty)

        if intrinsic_per_step < penalty_per_step:
            verdicts.append(Verdict(
                rule_name=self.name,
                severity=Severity.WARNING,
                message=(f"Intrinsic reward ({intrinsic_per_step:.5f}/step) < "
                         f"step penalty ({penalty_per_step:.5f}/step). "
                         f"Exploration is still punished on net."),
                details={"intrinsic": intrinsic_per_step,
                         "penalty": penalty_per_step,
                         "ratio": intrinsic_per_step / max(penalty_per_step, 1e-10)},
                recommendation=(f"Increase intrinsic coefficient to at least "
                                f"{penalty_per_step / max(intrinsic_per_step, 1e-10):.1f}x "
                                f"current value"),
            ))
        return verdicts


class BudgetSufficiency(Rule):
    """Check if training budget allows enough goal discoveries."""

    @property
    def name(self): return "budget_sufficiency"

    @property
    def description(self):
        return "Training budget must allow enough random goal discoveries to learn"

    @property
    def proof(self):
        return FormalBasis(
            proof_name="budget_sufficiency",
            strength=ProofStrength.GROUNDED,
            statement="∀ p > 0, k > 0, n ≥ k/p → p * n ≥ k",
            parameters={"discovery_prob": "p", "min_discoveries": "k", "episodes": "n"},
        )

    def check(self, model, config=None):
        verdicts = []
        if not model.goal_sources:
            return verdicts

        n_actors = getattr(config, 'n_actors', None) if config else None
        total_steps = getattr(config, 'total_steps', None) if config else None
        if n_actors is None or total_steps is None:
            return verdicts

        # Estimate average episode length, accounting for death.
        # Geometric distribution: E[life] = 1/death_probability, capped at max_steps.
        # When death_probability=0, episodes run to max_steps.
        if model.death_probability > 0:
            avg_episode_length = min(1.0 / model.death_probability, model.max_steps)
        else:
            avg_episode_length = float(model.max_steps)
        avg_episode_length = max(1.0, avg_episode_length)
        # total_steps is the total environment steps across ALL actors
        # (standard RL convention: Sample Factory's train_for_env_steps)
        total_episodes = total_steps / avg_episode_length

        for source in model.goal_sources:
            expected_discoveries = total_episodes * source.discovery_probability

            if expected_discoveries < 10:
                verdicts.append(Verdict(
                    rule_name=self.name,
                    severity=Severity.CRITICAL,
                    message=(f"Expected only {expected_discoveries:.0f} goal "
                             f"discoveries in {total_steps/1e6:.0f}M steps "
                             f"with {n_actors} actors. Need >=10 to learn."),
                    details={"expected_discoveries": expected_discoveries,
                             "total_episodes": total_episodes,
                             "p_discovery": source.discovery_probability},
                    recommendation=(f"Increase budget to "
                                    f"{int(10 / max(source.discovery_probability, 1e-10) * avg_episode_length / 1e6) + 1}M "
                                    f"steps or add intrinsic motivation"),
                ))
            elif expected_discoveries < 100:
                verdicts.append(Verdict(
                    rule_name=self.name,
                    severity=Severity.WARNING,
                    message=(f"Only ~{expected_discoveries:.0f} goal discoveries "
                             f"expected. Learning will be slow and high-variance."),
                    details={"expected_discoveries": expected_discoveries},
                ))
        return verdicts


class CompoundTrap(Rule):
    """Detect traps from reward source interactions."""

    @property
    def name(self): return "compound_trap"

    @property
    def description(self):
        return ("Detect traps arising from combinations of reward sources "
                "that individual rules miss")

    @property
    def proof(self):
        return FormalBasis(
            proof_name="compound_trap",
            strength=ProofStrength.VERIFIED,
            statement=("∀ goal > 0, penalty < 0, D > 0, loop_ev > 0, "
                       "goal < -penalty * D → loop_ev > goal + penalty * D"),
            parameters={"goal": "goal", "penalty": "penalty",
                        "disc_steps": "D", "loop_ev": "loop_ev"},
        )

    def applies_to(self, model):
        return len(model.reward_sources) >= 2

    def check(self, model, config=None):
        verdicts = []

        # 1. Step penalty + respawning reward: looping beats exploring
        if model.total_step_penalty < 0 and model.loopable_sources:
            penalty_per_step = abs(model.total_step_penalty)
            goal_ev = model.max_goal_reward
            for source in model.loopable_sources:
                if source.can_loop and source.loop_period > 0:
                    loop_reward_per_step = source.value / source.loop_period
                elif source.respawn == RespawnBehavior.TIMED and source.respawn_time > 0:
                    loop_reward_per_step = source.value / source.respawn_time
                elif source.respawn == RespawnBehavior.INFINITE:
                    loop_reward_per_step = source.value
                else:
                    continue
                # Respect max_occurrences for timed sources (same logic as RespawningExploit)
                effective_steps = model.max_steps
                if (source.max_occurrences > 0
                        and not source.can_loop
                        and source.respawn != RespawnBehavior.INFINITE):
                    max_loop_steps = source.max_occurrences * max(source.respawn_time, 1)
                    effective_steps = min(model.max_steps, max_loop_steps)
                loop_ev = (loop_reward_per_step - penalty_per_step) * effective_steps
                if loop_ev > goal_ev and loop_ev > 0:
                    verdicts.append(Verdict(
                        rule_name=self.name,
                        severity=Severity.CRITICAL,
                        message=(f"Penalty + respawning '{source.name}' compound trap: "
                                 f"loop EV ({loop_ev:+.1f}) > goal EV ({goal_ev:+.1f}). "
                                 f"Looping is the only positive-EV strategy because "
                                 f"the penalty makes exploration costly."),
                        details={"loop_ev": loop_ev, "goal_ev": goal_ev,
                                 "loop_reward_per_step": loop_reward_per_step,
                                 "penalty_per_step": penalty_per_step},
                        recommendation=(f"Remove step penalty or cap '{source.name}' "
                                        f"occurrences"),
                    ))

        # 2. Shaping + no terminal: agent cycles shaping forever
        if model.shaping_sources and not model.goal_sources:
            shaping_names = [s.name for s in model.shaping_sources]
            verdicts.append(Verdict(
                rule_name=self.name,
                severity=Severity.CRITICAL,
                message=(f"Shaping rewards ({', '.join(shaping_names)}) with no "
                         f"terminal goal. Agent will cycle the shaping reward "
                         f"forever since there is no incentive to terminate."),
                details={"shaping_sources": shaping_names},
                recommendation="Add a terminal goal reward or remove shaping",
            ))

        # 3. Death reward + penalty: dying is doubly incentivized
        # Only fire when the resettable reward is significant relative
        # to the penalty — a 0.001 resettable with -0.00001 penalty
        # is not a meaningful compound trap.
        if model.resettable_sources and model.total_step_penalty < 0:
            resettable_value = sum(s.value * s.discovery_probability
                                  for s in model.resettable_sources if s.value > 0)
            penalty_magnitude = abs(model.total_step_penalty)
            # Only fire if resettable reward is at least 10% of penalty per step
            if resettable_value <= penalty_magnitude * 0.1:
                pass  # too small to matter
            else:
                resettable_names = [s.name for s in model.resettable_sources]
                verdicts.append(Verdict(
                    rule_name=self.name,
                    severity=Severity.CRITICAL,
                    message=(f"Death-resettable rewards ({', '.join(resettable_names)}) "
                             f"combined with step penalty ({model.total_step_penalty:+.4f}). "
                             f"Dying is doubly incentivized: stops bleeding penalty AND "
                             f"resets fresh rewards for re-collection."),
                    details={"resettable_sources": resettable_names,
                             "total_step_penalty": model.total_step_penalty},
                    recommendation="Remove step penalty or don't reset rewards on death",
                ))

        return verdicts


class ShapingNotPotentialBased(Rule):
    """Check if shaping rewards are potential-based (safe) or not.

    Ng et al. 1999 proved that potential-based shaping F(s,a,s') = γΦ(s') - Φ(s)
    is the ONLY form that guarantees policy invariance. If shaping depends on
    the action a, or doesn't decompose into a potential difference, the optimal
    policy may change.

    Machine-verified: ng_shaping_preserves_optimal (LEAN 4, zero sorry)."""

    @property
    def name(self): return "shaping_not_potential_based"

    @property
    def description(self):
        return ("Non-potential-based shaping can change the optimal policy. "
                "Only F(s,a,s') = γΦ(s') - Φ(s) guarantees invariance (Ng 1999)")

    @property
    def proof(self):
        return FormalBasis(
            proof_name="ng_vstar_shaped",
            strength=ProofStrength.MOTIVATED,
            paper="Ng, Harada & Russell 1999",
            statement="V*_{M'} = V*_M - Φ — potential-based shaping shifts value by potential without changing optimal policy",
            parameters={"potential": "Φ", "mdp": "M"},
        )

    def applies_to(self, model):
        return len(model.shaping_sources) > 0

    def check(self, model, config=None):
        verdicts = []
        for source in model.shaping_sources:
            if source.requires_action:
                verdicts.append(Verdict(
                    rule_name=self.name,
                    severity=Severity.WARNING,
                    message=(f"Shaping reward '{source.name}' depends on action. "
                             f"This is NOT potential-based (Ng 1999 requires "
                             f"F(s,a,s') = γΦ(s') - Φ(s), independent of a). "
                             f"Optimal policy may change."),
                    details={"source": source.name,
                             "action_dependent": True},
                    recommendation=(f"Redesign '{source.name}' as a state-only "
                                    f"potential difference, or verify empirically "
                                    f"that the shaping doesn't distort the policy"),
                ))
            if source.can_loop and not source.intentional:
                verdicts.append(Verdict(
                    rule_name=self.name,
                    severity=Severity.INFO,
                    message=(f"Shaping reward '{source.name}' can be cycled. "
                             f"If this is potential-based (Φ(s') - Φ(s)), the "
                             f"cycle sum is zero (telescoping). If not, the "
                             f"agent may exploit the cycle."),
                    details={"source": source.name},
                ))
        return verdicts


class ProxyRewardHackability(Rule):
    """Check if a proxy reward can be hacked relative to the true objective.

    Based on Skalse et al. 2022 "Defining and Characterizing Reward Hacking"
    (NeurIPS 2022). Two reward functions are hackable if there exist policies
    where one reward prefers π but the other prefers π' — the proxy can be
    gamed at the expense of the true objective.

    Machine-verified: skalse_existence_two, skalse_existence_general
    (LEAN 4, zero sorry)."""

    @property
    def name(self): return "proxy_reward_hackability"

    @property
    def description(self):
        return ("Proxy reward may be hackable: optimizing the proxy could "
                "degrade the true objective (Skalse et al. 2022)")

    @property
    def proof(self):
        return FormalBasis(
            proof_name="skalse_existence_general",
            strength=ProofStrength.MOTIVATED,
            paper="Skalse et al. 2022",
            statement="∀ finite Π with distinct occupancy measures, ∀ R₁, ∃ R₂ non-trivially unhackable — unhackable proxies always exist",
            parameters={"policies": "Π", "true_reward": "R₁"},
        )

    def applies_to(self, model):
        # Applies when the model has both a goal and shaping/proxy rewards
        return len(model.goal_sources) > 0 and len(model.shaping_sources) > 0

    def check(self, model, config=None):
        verdicts = []
        if not model.shaping_sources:
            return verdicts

        # Check if shaping rewards create a proxy that could diverge from goal
        shaping_names = [s.name for s in model.shaping_sources]
        goal_names = [s.name for s in model.goal_sources]

        # Non-potential-based shaping = hackable proxy
        action_dependent = [s for s in model.shaping_sources if s.requires_action]
        if action_dependent:
            names = [s.name for s in action_dependent]
            verdicts.append(Verdict(
                rule_name=self.name,
                severity=Severity.WARNING,
                message=(f"Action-dependent shaping ({', '.join(names)}) acts as "
                         f"a proxy reward that may be hackable relative to the "
                         f"true goal ({', '.join(goal_names)}). Skalse et al. "
                         f"proved that non-trivial unhackable proxies exist — "
                         f"but this configuration is not guaranteed to be one."),
                details={"shaping_sources": names,
                         "goal_sources": goal_names},
                recommendation=("Use potential-based shaping (Ng 1999) to "
                                "guarantee the proxy preserves optimal policy"),
            ))

        # Dense shaping that dominates sparse goal = effective proxy replacement
        shaping_ev = sum(abs(s.value) * model.max_steps
                         for s in model.shaping_sources)
        goal_ev = model.max_goal_reward
        if goal_ev > 0 and shaping_ev > 10 * goal_ev:
            verdicts.append(Verdict(
                rule_name=self.name,
                severity=Severity.WARNING,
                message=(f"Shaping magnitude ({shaping_ev:.0f}) is "
                         f"{shaping_ev/goal_ev:.0f}x the goal ({goal_ev:.0f}). "
                         f"The shaping effectively replaces the goal as the "
                         f"optimization target — a classic proxy hacking setup."),
                details={"shaping_ev": shaping_ev, "goal_ev": goal_ev,
                         "ratio": shaping_ev / goal_ev},
                recommendation=("Reduce shaping magnitude or increase goal "
                                "reward so the true objective dominates"),
            ))

        return verdicts


class StagedRewardPlateau(Rule):
    """Detect plateau risk in staged/prerequisite-gated rewards.

    When reward B requires reward A as a prerequisite (e.g., Robosuite:
    lift requires grasp, hover requires lift), the agent gets ZERO
    gradient signal for B until A is achieved. This creates plateaus
    where the agent appears stuck. The harder the prerequisite, the
    longer the plateau."""

    @property
    def name(self): return "staged_reward_plateau"

    @property
    def description(self):
        return ("Staged rewards with prerequisite gates create learning "
                "plateaus — no gradient signal until prerequisites are met")

    @property
    def proof(self):
        return FormalBasis(
            proof_name="staged_sparsity",
            strength=ProofStrength.VERIFIED,
            statement=("∀ p : Fin n → [0,1], ∀ j, ∏ p_i ≤ p_j "
                       "(product of stage probabilities ≤ any single stage)"),
            parameters={"stage_probs": "p", "n_stages": "n"},
        )

    def applies_to(self, model):
        return len(model.staged_sources) > 0

    def check(self, model, config=None):
        verdicts = []
        source_names = {s.name for s in model.reward_sources}

        for source in model.staged_sources:
            if source.prerequisite not in source_names:
                verdicts.append(Verdict(
                    rule_name=self.name,
                    severity=Severity.WARNING,
                    message=(f"'{source.name}' requires prerequisite "
                             f"'{source.prerequisite}' which is not a defined "
                             f"reward source."),
                    recommendation="Check prerequisite name matches a source",
                ))
                continue

            # Find the chain depth (with cycle detection)
            chain = [source.name]
            visited = {source.name}
            current = source.prerequisite
            while current and current in source_names and current not in visited:
                chain.append(current)
                visited.add(current)
                prereq_src = next((s for s in model.reward_sources
                                   if s.name == current), None)
                if prereq_src and prereq_src.prerequisite:
                    current = prereq_src.prerequisite
                else:
                    break

            if len(chain) >= 3:
                verdicts.append(Verdict(
                    rule_name=self.name,
                    severity=Severity.WARNING,
                    message=(f"Deep prerequisite chain ({len(chain)} stages): "
                             f"{' → '.join(reversed(chain))}. Agent receives "
                             f"zero gradient for later stages until all "
                             f"prerequisites are achieved. Consider adding "
                             f"intermediate shaping rewards."),
                    details={"chain": list(reversed(chain)),
                             "depth": len(chain)},
                    recommendation=("Add distance-based shaping for each stage, "
                                    "or use curriculum learning to unlock stages"),
                ))
            elif len(chain) == 2:
                verdicts.append(Verdict(
                    rule_name=self.name,
                    severity=Severity.INFO,
                    message=(f"'{source.name}' is gated by '{source.prerequisite}'. "
                             f"No gradient signal until prerequisite is achieved."),
                    details={"chain": list(reversed(chain))},
                ))

        return verdicts


class RewardDominanceImbalance(Rule):
    """Detect when one reward component overwhelms all others.

    In multi-component rewards (robotics, game AI), a single term
    with large magnitude can dominate the gradient, making all other
    terms invisible to the optimizer. This is common in legged
    locomotion where a tracking reward at weight 1.0 coexists with
    penalties at weight 0.00001."""

    @property
    def name(self): return "reward_dominance_imbalance"

    @property
    def proof(self):
        return FormalBasis(
            proof_name="dominance_negligible",
            strength=ProofStrength.GROUNDED,
            statement="b < ε(a+b) → b/(a+b) < ε — large component makes small ones negligible",
            parameters={"dominant": "a", "small": "b", "threshold": "ε"},
        )

    @property
    def description(self):
        return ("One reward component dominates all others by >100x, "
                "making other components invisible to the optimizer")

    def applies_to(self, model):
        per_step = [s for s in model.reward_sources
                    if s.reward_type == RewardType.PER_STEP]
        return len(per_step) >= 3

    def check(self, model, config=None):
        verdicts = []
        per_step = [s for s in model.reward_sources
                    if s.reward_type == RewardType.PER_STEP]
        if len(per_step) < 3:
            return verdicts

        # Use value_range max if available, else abs(value)
        magnitudes = {}
        for s in per_step:
            if s.value_range:
                mag = max(abs(s.value_range[0]), abs(s.value_range[1]))
            else:
                mag = abs(s.value)
            magnitudes[s.name] = mag

        if not magnitudes:
            return verdicts

        max_mag = max(magnitudes.values())
        min_mag = min(v for v in magnitudes.values() if v > 0) if any(
            v > 0 for v in magnitudes.values()) else 1.0

        if max_mag > 0 and min_mag > 0 and max_mag / min_mag > 100:
            dominant = [n for n, m in magnitudes.items() if m == max_mag]
            invisible = [n for n, m in magnitudes.items()
                         if m > 0 and max_mag / m > 100]
            verdicts.append(Verdict(
                rule_name=self.name,
                severity=Severity.WARNING,
                message=(f"'{dominant[0]}' (magnitude {max_mag:.4g}) dominates "
                         f"by >{max_mag/min_mag:.0f}x over {len(invisible)} "
                         f"other per-step components. Smaller terms "
                         f"({', '.join(invisible[:3])}) are effectively "
                         f"invisible to the optimizer."),
                details={"dominant": dominant[0], "ratio": max_mag / min_mag,
                         "invisible": invisible},
                recommendation=("Rescale reward components to similar magnitude, "
                                "or accept that smaller terms serve as tiebreakers"),
            ))

        return verdicts


class ExponentialSaturation(Rule):
    """Detect saturation in exponential tracking rewards.

    Rewards like exp(-error/sigma) saturate to 1.0 quickly as the
    error decreases, creating a flat gradient region near the target.
    The agent learns to be "close enough" rather than precise.
    Common in legged locomotion velocity tracking."""

    @property
    def name(self): return "exponential_saturation"

    @property
    def proof(self):
        return FormalBasis(
            proof_name="exp_neg_ge_one_sub",
            strength=ProofStrength.VERIFIED,
            statement="exp(-x) ≥ 1-x for x ≥ 0 — exponential saturates within x of 1.0",
            parameters={"error_over_sigma": "x"},
        )

    @property
    def description(self):
        return ("Exponential tracking rewards saturate near the target, "
                "creating flat gradient regions that prevent precise control")

    def applies_to(self, model):
        return any(s.value_type == "exponential" for s in model.reward_sources)

    def check(self, model, config=None):
        verdicts = []
        for source in model.reward_sources:
            if source.value_type != "exponential":
                continue
            sigma = (source.value_params or {}).get("sigma", 0.25)
            if sigma < 0.5:
                verdicts.append(Verdict(
                    rule_name=self.name,
                    severity=Severity.INFO,
                    message=(f"'{source.name}' uses exp(-error/{sigma}) tracking. "
                             f"Saturates to ~0.95 at error={sigma*3:.2f}. "
                             f"Agent may learn 'close enough' rather than precise."),
                    details={"source": source.name, "sigma": sigma,
                             "saturation_95": sigma * 3},
                    recommendation=(f"If precision matters, consider linear or "
                                    f"quadratic penalty near the target"),
                ))
        return verdicts


class IntrinsicDominance(Rule):
    """Check if accumulated per-step intrinsic reward dominates the goal.

    Non-PBRS per-step additions can change the optimal policy (Ng 1999).
    When accumulated intrinsic EV exceeds goal EV, the agent optimizes
    curiosity over task completion. Documented in Burda 2019 (noisy TV,
    skull dancing), Taiga 2021 (ChopperCommand 4.7x worse with RND),
    Mavor-Parker 2022 (MiniGrid 100->5 novel states with TV).
    """

    @property
    def name(self): return "intrinsic_dominance"

    @property
    def description(self):
        return ("Accumulated per-step intrinsic reward exceeds "
                "the discounted goal reward over the episode")

    @property
    def proof(self):
        return FormalBasis(
            proof_name="ng_vstar_shaped",
            strength=ProofStrength.GROUNDED,
            statement=("Ng 1999: non-PBRS reward additions can change the "
                       "optimal policy. When accumulated intrinsic >> goal, "
                       "the mixed-reward policy diverges from the task policy."),
            parameters={"intrinsic_ev": "sum(r_i * D)",
                        "goal_ev": "R_goal * p * gamma^t"},
        )

    def applies_to(self, model):
        # Applies when there are non-intentional per-step sources with
        # infinite respawn (intrinsic motivation like RND/ICM) alongside
        # terminal/event goals. Intentional per-step sources (alive bonus
        # in a survival task) are the goal, not competing with it.
        has_intrinsic = any(
            s.reward_type == RewardType.PER_STEP
            and s.respawn == RespawnBehavior.INFINITE
            and s.value > 0
            and not s.intentional
            for s in model.reward_sources
        )
        has_goal = any(
            s.reward_type in (RewardType.TERMINAL, RewardType.ON_EVENT)
            and s.value > 0
            for s in model.reward_sources
        )
        return has_intrinsic and has_goal

    def check(self, model, config=None):
        verdicts = []
        disc = _discounted_steps(model.gamma, model.max_steps)

        # Accumulated intrinsic EV: non-intentional per-step with
        # INFINITE respawn. Intentional sources (alive bonus, survival
        # reward) are the goal itself, not intrinsic motivation.
        intrinsic_per_step = sum(
            _best_case_value(s) for s in model.reward_sources
            if s.reward_type == RewardType.PER_STEP
            and s.respawn == RespawnBehavior.INFINITE
            and s.value > 0
            and not s.intentional
        )
        intrinsic_ev = intrinsic_per_step * disc

        # Goal value: what the agent receives when it completes the task.
        # We use the raw goal value (not discounted by discovery probability)
        # because the question is whether the intrinsic signal competes with
        # the goal AFTER the agent has learned the path, not during random
        # exploration. Discovery probability affects exploration_threshold;
        # this rule asks: once the agent can reach the goal, is the intrinsic
        # reward still more attractive than finishing?
        goal_ev = 0.0
        for s in model.reward_sources:
            if s.reward_type == RewardType.TERMINAL and s.value > 0:
                goal_ev += s.value
            elif s.reward_type == RewardType.ON_EVENT and s.value > 0:
                goal_ev += s.value

        if intrinsic_ev <= 0 or goal_ev <= 0:
            return verdicts

        ratio = intrinsic_ev / goal_ev

        if ratio >= 5:
            verdicts.append(Verdict(
                rule_name=self.name,
                severity=Severity.CRITICAL,
                message=(f"Accumulated intrinsic reward "
                         f"(EV={intrinsic_ev:+.4f}) is {ratio:.1f}x the "
                         f"goal reward ({goal_ev:+.4f}). "
                         f"Agent will optimize curiosity over task completion."),
                details={"intrinsic_ev": intrinsic_ev, "goal_ev": goal_ev,
                         "ratio": ratio},
                recommendation=(
                    "Reduce intrinsic coefficient, anneal it during training, "
                    "increase goal reward, or use separate value heads for "
                    "intrinsic and extrinsic returns (Burda 2019)"),
                learn_more=(
                    "Non-PBRS per-step reward additions can change the optimal "
                    "policy (Ng 1999). When the accumulated intrinsic signal "
                    "exceeds the goal, the agent earns more from exploring than "
                    "from finishing the task. Documented failures: Pong agent "
                    "maximizes bounces not score (Burda 2019), ChopperCommand "
                    "scores 4.7x worse with RND than epsilon-greedy (Taiga 2021), "
                    "MiniGrid agent visits 100 novel states without noise but only "
                    "5 with noisy TV (Mavor-Parker 2022).\n"
                    "Fixes: (1) anneal intrinsic coefficient during training, "
                    "(2) use separate value heads with different discount factors "
                    "(gamma_E=0.999, gamma_I=0.99 per RND paper), "
                    "(3) constrained optimization (EIPO, Hong 2022), "
                    "(4) increase goal reward magnitude."
                ),
            ))
        elif ratio >= 0.5:
            verdicts.append(Verdict(
                rule_name=self.name,
                severity=Severity.WARNING,
                message=(f"Accumulated intrinsic reward "
                         f"(EV={intrinsic_ev:+.4f}) is {ratio:.1f}x the "
                         f"goal reward ({goal_ev:+.4f}). "
                         f"Intrinsic signal may compete with task completion."),
                details={"intrinsic_ev": intrinsic_ev, "goal_ev": goal_ev,
                         "ratio": ratio},
                recommendation=(
                    "Consider reducing intrinsic coefficient or increasing "
                    "goal reward to ensure task completion dominates"),
            ))
        return verdicts


# Standard reward rule collection
REWARD_RULES = [
    PenaltyDominatesGoal(),
    DeathBeatsSurvival(),
    IdleExploit(),
    ExplorationThreshold(),
    RespawningExploit(),
    DeathResetExploit(),
    ShapingLoopExploit(),
    ShapingNotPotentialBased(),
    ProxyRewardHackability(),
    IntrinsicSufficiency(),
    BudgetSufficiency(),
    CompoundTrap(),
    StagedRewardPlateau(),
    RewardDominanceImbalance(),
    ExponentialSaturation(),
    IntrinsicDominance(),
]
