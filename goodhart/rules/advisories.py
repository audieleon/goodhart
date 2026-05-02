"""Blind-spot advisories — pattern-based hints about failure modes
that static reward analysis cannot detect.

These rules emit INFO-level verdicts when the model configuration
matches patterns associated with known classes of RL failure that
lie outside goodhart's analytical scope. They don't diagnose the
problem — they say "based on your config, you should think about X."

Each advisory documents:
  - What pattern triggered it
  - What failure mode it hints at
  - Why goodhart can't check it
  - What the user should investigate

These are NOT warnings or criticals. They are honest disclosures
of the tool's boundaries.
"""

from goodhart.models import (
    RewardType,
    Severity,
    Verdict,
)
from goodhart.engine import Rule


class PhysicsExploitAdvisory(Rule):
    """Flag environments where emergent physics exploits are likely.

    Pattern: high state/action space + low death probability + long
    episodes. Rich dynamics with permissive termination let agents
    discover unintended affordances (box surfing, ramp launching).

    Cannot check: transition function T(s,a,s') exploits.
    """

    @property
    def name(self):
        return "advisory_physics_exploit"

    @property
    def description(self):
        return "Rich environment dynamics may enable physics exploits that reward analysis cannot detect"

    def applies_to(self, model):
        # Only fires for complex environments
        return model.n_states >= 50000 and model.n_actions >= 6 and model.death_probability < 0.05

    def check(self, model, config=None):
        # Additional signal: multiple reward sources (complex reward =
        # complex environment = more surface area for exploits)
        has_many_sources = len(model.reward_sources) >= 4
        has_long_episodes = model.max_steps >= 500

        if not (has_many_sources and has_long_episodes):
            return []

        return [
            Verdict(
                rule_name=self.name,
                severity=Severity.INFO,
                message=(
                    f"Complex environment ({model.n_states} states, "
                    f"{model.n_actions} actions, {len(model.reward_sources)} "
                    f"reward components, {model.max_steps} steps) with low "
                    f"termination risk — conditions where emergent physics "
                    f"exploits are commonly discovered."
                ),
                details={
                    "n_states": model.n_states,
                    "n_actions": model.n_actions,
                    "n_sources": len(model.reward_sources),
                    "death_probability": model.death_probability,
                },
                recommendation=(
                    "goodhart analyzes reward structure, not environment "
                    "dynamics. Monitor training for unexpected behaviors: "
                    "agents moving through walls, surfing objects, or "
                    "exploiting collision geometry. See Baker et al. 2020 "
                    "(Hide-and-Seek) for examples."
                ),
            )
        ]


class GoalMisgeneralizationAdvisory(Rule):
    """Flag environments where goal misgeneralization is likely.

    Pattern: single dominant terminal reward + high discovery
    probability. When the goal is easy to reach during training,
    agents may learn spurious correlations (shortcuts) instead of
    the intended behavior.

    Cannot check: what features the agent actually learns.
    """

    @property
    def name(self):
        return "advisory_goal_misgeneralization"

    @property
    def description(self):
        return (
            "Easy terminal goal may enable misgeneralization — agent learns a shortcut instead of the intended behavior"
        )

    def applies_to(self, model):
        goals = model.goal_sources
        if not goals:
            return False
        # Single dominant terminal goal with high discovery rate
        dominant = max(goals, key=lambda s: s.value)
        if dominant.discovery_probability < 0.5:
            return False
        # Exclude competitive environments (symmetric win/lose)
        neg_terminals = [s for s in model.reward_sources if s.reward_type == RewardType.TERMINAL and s.value < 0]
        if neg_terminals:
            # Symmetric win/lose = competitive, not misgeneralization
            return False
        # Exclude very high action spaces (likely learned reward / LLM)
        if model.n_actions >= 100:
            return False
        return True

    def check(self, model, config=None):
        goals = model.goal_sources
        dominant = max(goals, key=lambda s: s.value)

        # Only flag if there are few other reward signals
        non_goal_signals = [s for s in model.reward_sources if s.reward_type != RewardType.TERMINAL and s.value > 0]

        if len(non_goal_signals) >= 3:
            # Dense reward signal — less likely to misgeneralize
            return []

        return [
            Verdict(
                rule_name=self.name,
                severity=Severity.INFO,
                message=(
                    f"Goal '{dominant.name}' has high discovery rate "
                    f"({dominant.discovery_probability:.0%}). If training "
                    f"environments share structural patterns (goal always "
                    f"in same location, same visual features), the agent "
                    f"may learn those patterns instead of the reward."
                ),
                details={
                    "goal": dominant.name,
                    "discovery_probability": dominant.discovery_probability,
                },
                recommendation=(
                    "goodhart analyzes reward structure, not learned "
                    "representations. If training on procedural levels, "
                    "verify the agent generalizes to held-out layouts. "
                    "See Langosco et al. 2022 (Goal Misgeneralization, "
                    "ICML) and Di Langosco et al. 2023."
                ),
            )
        ]


class CreditAssignmentAdvisory(Rule):
    """Flag environments where credit assignment depth is extreme.

    Pattern: very low discovery probability + long episodes + no
    intermediate shaping. This suggests the agent needs a long
    sequence of correct actions before any reward — a credit
    assignment problem, not just sparsity.

    Cannot check: temporal depth of the required action sequence.
    """

    @property
    def name(self):
        return "advisory_credit_assignment"

    @property
    def description(self):
        return "Extremely sparse reward in long episodes suggests deep credit assignment problem beyond simple sparsity"

    def applies_to(self, model):
        goals = [s for s in model.reward_sources if s.requires_exploration and s.discovery_probability < 0.01]
        return len(goals) > 0 and model.max_steps >= 500

    def check(self, model, config=None):
        # Check if there's dense intermediate signal to help
        has_shaping = len(model.shaping_sources) > 0
        has_dense_events = any(
            s.reward_type == RewardType.ON_EVENT
            and s.value > 0
            and s.discovery_probability >= 0.1  # must be reasonably discoverable
            for s in model.reward_sources
        )

        if has_shaping or has_dense_events:
            # Dense intermediate signal — less likely to be a pure
            # credit assignment problem
            return []

        sparse_goals = [s for s in model.reward_sources if s.requires_exploration and s.discovery_probability < 0.01]

        steps_per_discovery = int(model.max_steps / max(sparse_goals[0].discovery_probability, 1e-6))

        return [
            Verdict(
                rule_name=self.name,
                severity=Severity.INFO,
                message=(
                    f"Extremely sparse reward (p={sparse_goals[0].discovery_probability}) "
                    f"over {model.max_steps} steps with no intermediate "
                    f"shaping. Expected ~{steps_per_discovery} steps between "
                    f"discoveries. This may be a deep credit assignment "
                    f"problem, not just reward sparsity."
                ),
                details={
                    "min_discovery_prob": sparse_goals[0].discovery_probability,
                    "max_steps": model.max_steps,
                    "steps_per_discovery": steps_per_discovery,
                    "has_shaping": has_shaping,
                },
                recommendation=(
                    "The exploration_threshold rule flags sparsity, but "
                    "can't distinguish 'one lucky action' from 'a sequence "
                    "of 100 correct actions.' If the task requires sequential "
                    "discoveries (find key → open door → reach goal), "
                    "consider curriculum learning or hierarchical RL, not "
                    "just intrinsic motivation. See Bellemare et al. 2013 "
                    "(Montezuma's Revenge)."
                ),
            )
        ]


class ConstrainedRLAdvisory(Rule):
    """Flag configs that look like soft-penalty approximations of constraints.

    Pattern: negative ON_EVENT sources with small absolute value relative
    to positive rewards. This often indicates a safety cost modeled as a
    reward penalty, which is NOT equivalent to constrained optimization.

    Cannot check: whether the user intends hard constraints.
    """

    @property
    def name(self):
        return "advisory_constrained_rl"

    @property
    def description(self):
        return (
            "Negative event rewards may approximate safety constraints "
            "— soft penalties are not equivalent to hard constraints"
        )

    def applies_to(self, model):
        # Look for negative ON_EVENT that smell like costs
        neg_events = [
            s
            for s in model.reward_sources
            if s.reward_type == RewardType.ON_EVENT and s.value < 0 and not s.requires_action
        ]
        return len(neg_events) > 0

    def check(self, model, config=None):
        neg_events = [
            s
            for s in model.reward_sources
            if s.reward_type == RewardType.ON_EVENT and s.value < 0 and not s.requires_action
        ]

        pos_value = sum(s.value for s in model.reward_sources if s.value > 0)

        # Filter to penalties that smell like safety costs, not punishments:
        # - Small relative to positive reward (< 20% of pos value)
        # - Name doesn't suggest game-like punishment
        punishment_names = {"death", "die", "kill", "damage", "hit", "lose"}
        small_penalties = [
            s
            for s in neg_events
            if abs(s.value) < pos_value * 0.2 and not any(p in s.name.lower() for p in punishment_names)
        ]

        if not small_penalties:
            return []

        names = ", ".join(f"'{s.name}'" for s in small_penalties)
        return [
            Verdict(
                rule_name=self.name,
                severity=Severity.INFO,
                message=(
                    f"Negative event rewards ({names}) may represent safety "
                    f"constraints modeled as penalties. If these are intended "
                    f"as hard limits (max N violations per episode), a soft "
                    f"penalty allows the agent to exceed the budget when "
                    f"reward is high enough."
                ),
                details={
                    "penalty_sources": [s.name for s in small_penalties],
                    "positive_reward_sum": pos_value,
                },
                recommendation=(
                    "If these are safety constraints, consider constrained "
                    "RL (CPO, FOCOPS) which enforces hard cost budgets. "
                    "goodhart models costs as reward penalties but cannot "
                    "analyze constraint feasibility. See Achiam et al. 2017 "
                    "(CPO, ICML) and Ray et al. 2019 (Safety Gym)."
                ),
            )
        ]


class NonStationarityAdvisory(Rule):
    """Flag environments that suggest multi-agent or self-play dynamics.

    Pattern: symmetric positive/negative terminal rewards, or a name
    suggesting competition. Non-stationary opponents make the effective
    reward distribution shift during training.

    Cannot check: opponent adaptation dynamics.
    """

    @property
    def name(self):
        return "advisory_nonstationarity"

    @property
    def description(self):
        return (
            "Symmetric win/lose rewards suggest competitive dynamics where opponent adaptation creates non-stationarity"
        )

    def applies_to(self, model):
        terminals = [s for s in model.reward_sources if s.reward_type == RewardType.TERMINAL]
        if len(terminals) < 2:
            return False
        pos = [s for s in terminals if s.value > 0]
        neg = [s for s in terminals if s.value < 0]
        if not pos or not neg:
            return False
        # Check for symmetric-ish rewards (win/lose)
        max_pos = max(s.value for s in pos)
        min_neg = min(s.value for s in neg)
        if abs(max_pos + min_neg) >= max_pos * 0.5:
            return False
        # Distinguish competitive (both outcomes require action) from
        # survive/die (death is passive). Symmetric +15/-15 with
        # passive death is single-agent survival, not competition.
        pos_passive = any(not s.requires_action for s in pos)
        neg_passive = any(not s.requires_action for s in neg)
        if pos_passive or neg_passive:
            # One outcome doesn't require agent action → not competitive
            return False
        return True

    def check(self, model, config=None):
        terminals = [s for s in model.reward_sources if s.reward_type == RewardType.TERMINAL]
        pos = [s for s in terminals if s.value > 0]
        neg = [s for s in terminals if s.value < 0]

        return [
            Verdict(
                rule_name=self.name,
                severity=Severity.INFO,
                message=(
                    f"Symmetric terminal rewards ({pos[0].name}: "
                    f"{pos[0].value:+.1f}, {neg[0].name}: {neg[0].value:+.1f}) "
                    f"suggest competitive/self-play dynamics. If the opponent "
                    f"adapts during training, the reward distribution is "
                    f"non-stationary."
                ),
                details={
                    "positive_terminal": pos[0].name,
                    "negative_terminal": neg[0].name,
                },
                recommendation=(
                    "goodhart analyzes static reward structure. In self-play "
                    "or multi-agent settings, watch for: forgetting cycles "
                    "(agent loses to old strategies), strategy collapse "
                    "(both agents converge to dominated equilibrium), and "
                    "non-transitivity. See Bansal et al. 2018 (ICLR) and "
                    "Lanctot et al. 2017 (NIPS)."
                ),
            )
        ]


class LearnedRewardAdvisory(Rule):
    """Flag configurations that might use a learned reward model.

    Pattern: single terminal reward + very high state/action space +
    no shaping or intermediate rewards. This sparse structure in a
    complex environment is unusual for hand-designed rewards and may
    indicate a learned reward model wrapper.

    Cannot check: whether the reward function is learned vs hand-designed.
    """

    @property
    def name(self):
        return "advisory_learned_reward"

    @property
    def description(self):
        return (
            "Minimal reward structure in complex environment may "
            "indicate a learned reward model — goodhart cannot "
            "analyze learned reward dynamics"
        )

    def applies_to(self, model):
        return len(model.reward_sources) <= 2 and model.n_states >= 100000 and model.n_actions >= 10

    def check(self, model, config=None):
        # Very high action space with very simple reward = likely learned
        if model.n_actions < 100:
            return []

        return [
            Verdict(
                rule_name=self.name,
                severity=Severity.INFO,
                message=(
                    f"Simple reward structure ({len(model.reward_sources)} sources) "
                    f"in a very complex environment ({model.n_states} states, "
                    f"{model.n_actions} actions). If this reward comes from a "
                    f"learned model (RLHF, preference learning), goodhart "
                    f"cannot analyze it — the failure modes are in the reward "
                    f"model, not the reward structure."
                ),
                details={
                    "n_sources": len(model.reward_sources),
                    "n_states": model.n_states,
                    "n_actions": model.n_actions,
                },
                recommendation=(
                    "If using a learned reward model, watch for reward model "
                    "overoptimization (Gao et al. 2023), sycophancy, and "
                    "distributional shift between RM training data and policy "
                    "outputs. See Casper et al. 2023 ('Open Problems and "
                    "Fundamental Limitations of RLHF')."
                ),
            )
        ]


class MissingConstraintAdvisory(Rule):
    """Flag configs where the reward may be structurally incomplete.

    Pattern: rich continuous environment (many actuators, large state
    space, permissive termination) with only positive reward terms and
    few sources. This is the pattern where engineers forget to specify
    safety constraints — the agent has many degrees of freedom and
    nothing telling it what NOT to do.

    Cannot check: what reward terms SHOULD exist but don't.
    """

    @property
    def name(self):
        return "advisory_missing_constraint"

    @property
    def description(self):
        return (
            "All-positive reward in rich control environment — agent "
            "has many actuators but nothing penalizing unsafe behavior"
        )

    def applies_to(self, model):
        # Fire on continuous control with few sources and permissive
        # termination — regardless of reward sign. The missing
        # constraint risk is the same whether the reward is all-positive
        # (tokamak: tracking + no coil constraint) or all-negative
        # (nuclear: -||error||² + no temperature constraint).
        no_safety_penalties = not any(
            s.reward_type == RewardType.ON_EVENT and s.value < 0 for s in model.reward_sources
        )
        rich_state = model.n_states >= 50000
        permissive = model.death_probability < 0.05
        few_sources = len(model.reward_sources) <= 4
        return no_safety_penalties and model.is_continuous_control and rich_state and permissive and few_sources

    def check(self, model, config=None):
        return [
            Verdict(
                rule_name=self.name,
                severity=Severity.INFO,
                message=(
                    f"All {len(model.reward_sources)} reward sources are "
                    f"non-negative in an environment with {model.n_actions} "
                    f"actions and {model.n_states} states. The agent has many "
                    f"degrees of freedom but nothing penalizing unsafe or "
                    f"undesired behavior."
                ),
                details={
                    "n_sources": len(model.reward_sources),
                    "n_actions": model.n_actions,
                    "all_positive": True,
                },
                recommendation=(
                    "A clean bill from goodhart means 'no structural traps in "
                    "what you specified' — not 'your reward is complete.' In "
                    "continuous control with many actuators, consider whether "
                    "safety constraints (joint limits, force limits, collision "
                    "avoidance) should be reward penalties or hard action masks. "
                    "See Degrave et al. 2022 (tokamak plasma) for a case where "
                    "a missing coil-balance term caused real engineering damage."
                ),
            )
        ]


class AggregationTrapAdvisory(Rule):
    """Flag configs where episode-level aggregation may create traps.

    Pattern: only small positive per-step rewards, no terminal goal,
    no penalty for inaction. This suggests the real objective is a
    RATIO or other aggregation (Sharpe, success rate, average) rather
    than the sum of per-step rewards.

    Cannot check: what aggregation function is applied over episodes.
    """

    @property
    def name(self):
        return "advisory_aggregation_trap"

    @property
    def description(self):
        return (
            "Small per-step rewards with no terminal goal suggest "
            "the real objective may be an aggregation (ratio, average) "
            "that creates traps invisible to per-step analysis"
        )

    def applies_to(self, model):
        has_terminal = any(s.reward_type == RewardType.TERMINAL for s in model.reward_sources)
        if has_terminal:
            return False
        # All per-step rewards are small and positive
        per_step = [s for s in model.reward_sources if s.reward_type == RewardType.PER_STEP]
        if not per_step:
            return False
        all_small = all(abs(s.value) < 0.1 for s in per_step)
        all_positive = all(s.value >= 0 for s in per_step)
        return all_small and all_positive

    def check(self, model, config=None):
        per_step = [s for s in model.reward_sources if s.reward_type == RewardType.PER_STEP]
        max_val = max(s.value for s in per_step)

        return [
            Verdict(
                rule_name=self.name,
                severity=Severity.INFO,
                message=(
                    f"All rewards are small per-step values (max {max_val}) "
                    f"with no terminal goal. If the real objective is a ratio "
                    f"(Sharpe, success rate) or other non-sum aggregation, "
                    f"the per-step structure may not reveal traps in the "
                    f"aggregation."
                ),
                details={
                    "max_per_step_value": max_val,
                    "n_per_step": len(per_step),
                },
                recommendation=(
                    "goodhart analyzes sum-of-rewards structure. If your real "
                    "objective is a ratio (e.g., Sharpe = mean/std) or rate "
                    "(e.g., win%), doing nothing may be optimal under that "
                    "aggregation even though the per-step reward looks correct. "
                    "Check: does inaction produce a degenerate ratio? See "
                    "financial RL literature on Sharpe ratio idle exploits."
                ),
            )
        ]


ADVISORY_RULES = [
    PhysicsExploitAdvisory(),
    GoalMisgeneralizationAdvisory(),
    CreditAssignmentAdvisory(),
    ConstrainedRLAdvisory(),
    NonStationarityAdvisory(),
    LearnedRewardAdvisory(),
    MissingConstraintAdvisory(),
    AggregationTrapAdvisory(),
]
