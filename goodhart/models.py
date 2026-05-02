"""Formal environment and training models.

Dataclasses that describe the MDP's reward dynamics, state structure,
transition properties, and training hyperparameters. These are the
shared "space" that all analysis rules operate on.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# =====================================================================
# Reward source types and behaviors
# =====================================================================

class RewardType(Enum):
    """How a reward source behaves over time."""
    TERMINAL = "terminal"          # given once at episode end (goal)
    PER_STEP = "per_step"          # given every step (penalty/bonus)
    ON_EVENT = "on_event"          # given when event occurs (collect item)
    SHAPING = "shaping"            # function of state transition (distance decrease)


class RespawnBehavior(Enum):
    """How a reward source regenerates."""
    NONE = "none"                  # collected once, gone
    TIMED = "timed"                # respawns after N steps
    ON_DEATH = "on_death"          # resets when agent dies
    ON_EPISODE = "on_episode"      # resets each episode
    INFINITE = "infinite"          # always available (per-step)


# =====================================================================
# Reward source
# =====================================================================

@dataclass
class RewardSource:
    """A single source of reward in the environment.

    Quick reference for the flags that matter most:

    intentional (bool, default False):
        Is this reward component the actual goal? Set True for the thing
        you want the agent to accomplish (forward velocity in locomotion,
        goal reaching in navigation). Set False for shaping, penalties,
        alive bonuses, and auxiliary signals. This flag changes which
        rules fire: a passive +5/step marked intentional is a survival
        task; marked non-intentional, it's an idle exploit.

    requires_action (bool, default True):
        Does the agent need to DO something to earn this reward? Set False
        for alive bonuses, passive tracking, and anything earned by
        existing. Set True for velocity rewards, goal reaching, and
        anything requiring deliberate behavior. This flag determines the
        idle floor: passive rewards accumulate even when the agent does
        nothing.

    can_loop (bool, default False):
        Can the agent harvest this reward repeatedly by cycling through
        states? Set True for shaping rewards where the agent can move
        toward a target then away then toward again (distance decrease,
        checkpoint crossing). Set False for terminal rewards, one-time
        events, and potential-based shaping (which nets to zero over
        cycles). This flag triggers shaping_loop_exploit.

    respawn (RespawnBehavior, default NONE):
        What happens to this reward after it's collected? NONE = gone
        forever. TIMED = reappears after respawn_time steps. ON_DEATH =
        resets when the agent dies. ON_EPISODE = resets each episode.
        INFINITE = always available (per-step rewards). Respawning
        rewards can be farmed; the tool checks whether farming them
        beats completing the actual task.

    discovery_probability (float, default 1.0):
        How likely is a random agent to encounter this reward in an
        episode? Set 1.0 for per-step rewards the agent always sees.
        Set low (0.001-0.05) for sparse goals that require deliberate
        exploration to find. Drives exploration threshold analysis.

    state_dependent (bool, default False):
        Does the reward value change based on the environment state?
        Set True for tracking rewards (-||error||^2), velocity rewards,
        and anything that varies with how well the agent is doing.
        Set False for fixed bonuses (+1/step alive) and constant
        penalties. Affects negative_only_reward severity: constant
        negative is worse than state-dependent negative.

    explore_fraction (float, default 0.0):
        What fraction of this reward does a random agent earn? Set 0.0
        if random actions produce zero reward (e.g., precise velocity
        tracking). Set 0.5 if random actions earn about half (e.g.,
        a walking agent that sometimes stumbles forward). Used by
        idle_exploit to estimate whether exploration is net-positive.

    Other fields (value_range, value_type, value_params, scales_with,
    prerequisite, modifies, modifier_type) model advanced reward
    structures. See inline comments below for details.
    """
    name: str
    reward_type: RewardType
    value: float                           # reward per occurrence
    respawn: RespawnBehavior = RespawnBehavior.NONE
    respawn_time: int = 0                  # steps until respawn (if TIMED)
    max_occurrences: int = 1               # per episode (0 = unlimited)
    requires_action: bool = True           # does agent need to act to get it?
    requires_exploration: bool = False     # does agent need to find it first?
    discovery_probability: float = 1.0     # p(finding it) per episode
                                          # For continuous rewards (velocity, distance),
                                          # set to 1.0 — the agent receives the signal
                                          # every episode. For sparse goals (reach
                                          # a staircase, collect an item), set to the
                                          # fraction of random episodes that succeed.
    can_loop: bool = False                 # can agent harvest in a cycle?
    loop_period: int = 0                   # steps per loop cycle
    intentional: bool = False             # True = this reward IS the goal
                                          # (e.g., alive bonus in survival tasks)
                                          # Intentional rewards are not flagged
                                          # as respawning exploits.
    # --- Expressiveness extensions ---

    # State-dependent value modeling
    value_range: Optional[tuple] = None   # (min, max) actual range if known.
                                          # Rules use range bounds for
                                          # worst/best case analysis.
    value_type: str = "constant"          # How value varies with state:
                                          # "constant" — fixed scalar (default)
                                          # "proportional" — scales with a state var
                                          # "exponential" — exp(-error/sigma) tracking
                                          # "inverse" — 1/(distance + eps)
    value_params: Optional[Dict] = None   # Parameters for the value function:
                                          # proportional: {"scale": 0.1}
                                          # exponential: {"sigma": 0.25}
                                          # inverse: {"eps": 0.1}
    state_dependent: bool = False         # True if reward varies with state
    scales_with: Optional[str] = None     # What it tracks ("velocity", "distance")

    # Prerequisite gating (staged rewards)
    prerequisite: Optional[str] = None    # Name of another RewardSource that
                                          # must have fired before this one
                                          # activates. For staged rewards like
                                          # Robosuite pick-and-place:
                                          # grasp → lift → hover → place.
                                          # Rules detect plateau risk when
                                          # prerequisites create hard gates.

    # Reward modifiers (one source scales another)
    modifies: Optional[str] = None        # Name of another RewardSource this
                                          # one modifies (e.g., MetaDrive's
                                          # lateral_factor modifies driving_reward)
    modifier_type: str = "none"           # "none", "multiplicative", "additive"

    explore_fraction: float = 0.0         # Fraction of this reward earned by
                                          # random exploration (0.0 = none,
                                          # 1.0 = full value). Used by idle_exploit
                                          # to estimate explore EV for intentional
                                          # rewards. E.g., BipedalWalker forward
                                          # progress: ~0.5 (random falling earns
                                          # some velocity). ANYmal velocity
                                          # tracking: 0.0 (random actions don't
                                          # track velocity at all).

    def __post_init__(self):
        import math
        if math.isnan(self.value) or math.isinf(self.value):
            raise ValueError(
                f"value must be finite, got {self.value}")
        if not (0.0 <= self.discovery_probability <= 1.0):
            raise ValueError(
                f"discovery_probability must be in [0, 1], "
                f"got {self.discovery_probability}")
        if self.max_occurrences < 0:
            raise ValueError(
                f"max_occurrences must be >= 0, "
                f"got {self.max_occurrences}")
        if self.loop_period < 0:
            raise ValueError(
                f"loop_period must be >= 0, got {self.loop_period}")
        if not (0.0 <= self.explore_fraction <= 1.0):
            raise ValueError(
                f"explore_fraction must be in [0, 1], "
                f"got {self.explore_fraction}")


# =====================================================================
# Environment model
# =====================================================================

@dataclass
class EnvironmentModel:
    """Formal description of an MDP's reward dynamics.

    You don't need to model the full MDP — just the reward structure
    and enough environment context for the rules to reason about it.

    name (str): A human-readable label for the environment.

    max_steps (int, default 500): Maximum episode length in steps.
        Affects discount horizon analysis and penalty accumulation.

    gamma (float, default 0.99): Discount factor. Lower values make
        the agent more myopic. At gamma=0.9, rewards 20 steps away
        are worth 12% of immediate rewards. At gamma=0.99, they're
        worth 82%.

    n_states (int, default 1000): Approximate state space size.
        Affects exploration analysis and capacity checks.

    n_actions (int, default 8): Number of actions available.
        Continuous control typically has fewer actions but each is
        a vector. Affects entropy and exploration analysis.

    action_type (str, default "auto"): "discrete" for Atari/gridworld,
        "continuous" for robotics/control, "auto" to infer from context.

    death_probability (float, default 0.01): Probability of episode
        termination per step from agent failure. High values make
        death-beats-survival traps more likely.

    wall_probability (float, default 0.3): Probability that an action
        has no effect (hitting a wall). Affects exploration analysis.
    """
    name: str

    # Episode structure
    max_steps: int = 500
    gamma: float = 0.99

    # Reward sources (the core of the model)
    reward_sources: List[RewardSource] = field(default_factory=list)

    # State space
    n_states: int = 1000
    n_actions: int = 8
    action_type: str = "auto"              # "discrete", "continuous", or "auto"
                                           # "discrete" = K choices (Atari, gridworld)
                                           # "continuous" = R^n vector (robotics, control)
                                           # "auto" = inferred from context
    death_probability: float = 0.01        # p(die) per step
    wall_probability: float = 0.3          # p(wasted action) per step

    def __post_init__(self):
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError(
                f"gamma must be in [0, 1], got {self.gamma}")
        if self.max_steps <= 0:
            raise ValueError(
                f"max_steps must be > 0, got {self.max_steps}")
        if self.n_states <= 0:
            raise ValueError(
                f"n_states must be > 0, got {self.n_states}")
        if self.n_actions <= 0:
            raise ValueError(
                f"n_actions must be > 0, got {self.n_actions}")
        if not (0.0 <= self.death_probability <= 1.0):
            raise ValueError(
                f"death_probability must be in [0, 1], got {self.death_probability}")
        if not (0.0 <= self.wall_probability <= 1.0):
            raise ValueError(
                f"wall_probability must be in [0, 1], got {self.wall_probability}")

    def add_reward_source(self, source: RewardSource):
        self.reward_sources.append(source)

    @property
    def is_continuous_control(self) -> bool:
        """Whether this environment has continuous action space.

        Uses action_type if explicitly set; otherwise infers from
        the presence of state-dependent per-step rewards that scale
        with physical quantities (velocity, torque, distance) — a
        strong signal for continuous control environments.
        """
        if self.action_type == "continuous":
            return True
        if self.action_type == "discrete":
            return False
        # Auto: infer from reward structure
        physical_signals = any(
            s.scales_with is not None
            for s in self.reward_sources
        )
        return physical_signals

    @property
    def goal_sources(self) -> List[RewardSource]:
        return [s for s in self.reward_sources
                if s.reward_type == RewardType.TERMINAL and s.value > 0]

    @property
    def penalty_sources(self) -> List[RewardSource]:
        return [s for s in self.reward_sources
                if s.value < 0]

    @property
    def shaping_sources(self) -> List[RewardSource]:
        return [s for s in self.reward_sources
                if s.reward_type == RewardType.SHAPING]

    @property
    def loopable_sources(self) -> List[RewardSource]:
        return [s for s in self.reward_sources
                if not s.intentional  # intentional rewards are not exploits
                and (s.can_loop or s.respawn in (RespawnBehavior.TIMED,
                                                  RespawnBehavior.INFINITE))]

    @property
    def resettable_sources(self) -> List[RewardSource]:
        return [s for s in self.reward_sources
                if s.respawn == RespawnBehavior.ON_DEATH]

    @property
    def staged_sources(self) -> List[RewardSource]:
        """Sources with prerequisite gates (staged rewards)."""
        return [s for s in self.reward_sources if s.prerequisite is not None]

    @property
    def modifier_sources(self) -> List[RewardSource]:
        """Sources that modify other sources."""
        return [s for s in self.reward_sources
                if s.modifies is not None and s.modifier_type != "none"]

    @property
    def state_dependent_sources(self) -> List[RewardSource]:
        """Sources whose value varies with state."""
        return [s for s in self.reward_sources if s.state_dependent]

    @property
    def total_step_penalty(self) -> float:
        """Sum of negative per-step rewards, excluding multiplicative modifiers.

        Multiplicative modifiers (modifier_type != "none") scale another
        source's value rather than adding independently. Including them
        in the additive sum double-counts their effect and produces
        incorrect EV calculations (e.g., treating a 0.5x factor as -0.5
        penalty).
        """
        return sum(s.value for s in self.reward_sources
                   if s.reward_type == RewardType.PER_STEP
                   and s.value < 0
                   and s.modifier_type == "none")

    @property
    def independent_sources(self) -> list:
        """Sources that contribute independently (not modifiers of others)."""
        return [s for s in self.reward_sources
                if s.modifier_type == "none"]

    def effective_value(self, source: 'RewardSource') -> float:
        """Compute the effective per-step value of a source after modifiers.

        For a base source with multiplicative modifiers, the effective
        value is: base_value * prod(1 + modifier_value) for each modifier.
        For sources with no modifiers, returns the raw value.
        """
        if source.modifier_type != "none":
            return 0.0  # Modifiers don't have independent value
        modifiers = [s for s in self.reward_sources
                     if s.modifies == source.name
                     and s.modifier_type == "multiplicative"]
        if not modifiers:
            return source.value
        # Multiplicative modifiers are factors in [0,1] applied to base
        # Worst case for designer: all modifiers at their penalty value
        factor = 1.0
        for m in modifiers:
            # modifier value is the penalty (e.g., -0.5 means p=0.5)
            factor *= max(0.0, 1.0 + m.value)
        return source.value * factor

    @property
    def max_goal_reward(self) -> float:
        return sum(s.value for s in self.goal_sources)


# =====================================================================
# Verdict -- what a rule produces
# =====================================================================

class Severity(Enum):
    """Verdict severity level.

    INFO: supplementary information, no action required.
    WARNING: potential issue depending on environment dynamics.
    CRITICAL: mathematically provable failure mode — fix before training.
    """
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Verdict:
    """Output of a rule analysis."""
    rule_name: str
    severity: Severity
    message: str
    details: dict = field(default_factory=dict)
    recommendation: Optional[str] = None
    learn_more: Optional[str] = None   # deeper explanation for --verbose

    def __str__(self):
        icon = {"info": "i", "warning": "!", "critical": "X"}[self.severity.value]
        s = f"[{icon}] [{self.rule_name}] {self.message}"
        if self.recommendation:
            s += f"\n  -> {self.recommendation}"
        return s

    def verbose_str(self):
        """Extended output including learn_more context."""
        s = str(self)
        if self.learn_more:
            # Indent each line of learn_more
            for line in self.learn_more.split("\n"):
                s += f"\n     {line}"
        return s


# =====================================================================
# Result -- typed return from analysis
# =====================================================================

@dataclass
class Result:
    """Typed return from analysis."""
    verdicts: List[Verdict]
    passed: bool  # True if no criticals

    @property
    def criticals(self) -> List[Verdict]:
        return [v for v in self.verdicts if v.severity == Severity.CRITICAL]

    @property
    def warnings(self) -> List[Verdict]:
        return [v for v in self.verdicts if v.severity == Severity.WARNING]

    @property
    def infos(self) -> List[Verdict]:
        return [v for v in self.verdicts if v.severity == Severity.INFO]

    @property
    def has_criticals(self) -> bool:
        return len(self.criticals) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def to_dict(self, verbose=False) -> dict:
        def _verdict_dict(v, include_rec=True):
            d = {"rule": v.rule_name, "message": v.message}
            if include_rec and v.recommendation:
                d["recommendation"] = v.recommendation
            if verbose and v.learn_more:
                d["learn_more"] = v.learn_more
            return d
        return {
            "passed": self.passed,
            "criticals": [_verdict_dict(v) for v in self.criticals],
            "warnings": [_verdict_dict(v) for v in self.warnings],
            "infos": [_verdict_dict(v, include_rec=True) for v in self.infos],
        }


# =====================================================================
# Training configuration
# =====================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters -- extends EnvironmentModel for
    optimizer-aware analysis."""

    # Algorithm
    algorithm: str = "PPO"           # PPO, APPO, IMPALA, DQN, SAC, DDPG, TD3, A2C

    # Learning rates
    lr: float = 3e-4
    critic_lr: Optional[float] = None  # None = same as lr
    warmup_updates: int = 50

    # PPO/APPO-specific
    clip_epsilon: float = 0.2
    target_kl: Optional[float] = None  # None = no KL early stopping
    num_epochs: int = 4              # PPO reuse epochs
    minibatch_size: int = 512
    rollout_length: int = 128

    # Entropy
    entropy_coeff: float = 0.01
    entropy_coeff_final: float = 0.001

    # Gradient
    max_grad_norm: float = 0.5

    # Value function
    value_coef: float = 0.5
    gae_lambda: float = 0.95

    # Off-policy (DQN, SAC, DDPG, TD3)
    replay_buffer_size: int = 0      # 0 = on-policy (no buffer)
    target_update_freq: int = 0      # 0 = no target network
    tau: float = 0.005               # soft update coefficient
    epsilon_start: float = 1.0       # DQN exploration
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 0     # 0 = no epsilon schedule

    # SAC-specific
    alpha: float = 0.2               # entropy temperature
    auto_alpha: bool = False         # learn alpha automatically

    # TD3-specific
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2            # update policy every N critic updates

    # Architecture
    model_params: int = 1_000_000    # total model parameters
    embed_dim: int = 256
    num_specialists: int = 1         # 1 = monolithic, 3+ = multi-network
    routing_floor: float = 0.0       # 0 = no floor constraint
    balance_coef: float = 0.0        # 0 = no load balancing

    # Parallelism
    num_envs: int = 16
    num_workers: int = 1             # for APPO/IMPALA

    # Training budget
    n_actors: int = 64
    total_steps: int = 20_000_000

    # Recurrence
    use_rnn: bool = False
    rnn_type: str = "lstm"           # lstm or gru
    rnn_size: int = 256

    def __post_init__(self):
        if self.lr < 0:
            raise ValueError(f"lr must be >= 0, got {self.lr}")
        if self.total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {self.total_steps}")
        if self.n_actors <= 0:
            raise ValueError(f"n_actors must be > 0, got {self.n_actors}")
        if self.num_envs <= 0:
            raise ValueError(f"num_envs must be > 0, got {self.num_envs}")
        if self.minibatch_size <= 0:
            raise ValueError(f"minibatch_size must be > 0, got {self.minibatch_size}")
        if self.clip_epsilon <= 0:
            raise ValueError(f"clip_epsilon must be > 0, got {self.clip_epsilon}")


# =====================================================================
# Formal basis -- links rules to LEAN proofs
# =====================================================================

class ProofStrength:
    """How strongly the LEAN theorem validates the Python rule.

    VERIFIED: The Python check() is a direct instance of the theorem.
      The theorem proves exactly the property the rule checks.
      (e.g., intrinsic_sufficiency: intrinsic < penalty → net < 0)

    GROUNDED: The theorem proves the undiscounted/simplified core.
      The Python extends with discounting, thresholds, or aggregation
      that go beyond what's formally verified.
      (e.g., penalty_dominates_goal: theorem is undiscounted,
      Python adds gamma-discounting and severity thresholds)

    MOTIVATED: The theorem proves WHY the issue matters, but the
      Python rule checks something structurally different (config
      flags, heuristic thresholds, existence results).
      (e.g., expert_collapse: theorem proves softmax concentrates,
      Python checks if routing_floor == 0)
    """
    VERIFIED = "verified"
    GROUNDED = "grounded"
    MOTIVATED = "motivated"


@dataclass
class FormalBasis:
    """Links a Python rule to a LEAN theorem.

    The `strength` field indicates how directly the theorem validates
    the rule's runtime logic. See ProofStrength for definitions.

    The test suite verifies that every declared proof_name exists
    in the LEAN source files.
    """
    proof_name: str                          # LEAN theorem name
    strength: str = ProofStrength.GROUNDED   # verified/grounded/motivated
    proof_file: str = "GoodhartProofs/Basic.lean"  # relative to proofs/
    paper: Optional[str] = None              # source paper if verifying published work
    statement: str = ""                      # human-readable theorem statement
    parameters: Dict[str, str] = field(default_factory=dict)  # maps Python param → LEAN param
