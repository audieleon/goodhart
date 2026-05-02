"""Training and optimizer analysis rules.

8 rules that operate on training hyperparameters to detect
known failure modes BEFORE training starts.

Based on documented failure modes from:
- "37 Implementation Details of PPO" (ICLR Blog 2022)
- "When Learning Rates Go Wrong" (arXiv 2603.09950)
- "On Pathologies in KL-Regularized RL" (NeurIPS 2021)
- Andy Jones' "Debugging RL Systems"
"""

from goodhart.models import (
    FormalBasis,
    ProofStrength,
    Severity,
    Verdict,
    TrainingConfig,
)
from goodhart.engine import Rule


class LearningRateRegime(Rule):
    """Check if learning rate is in a known-bad regime."""

    @property
    def name(self):
        return "lr_regime"

    @property
    def description(self):
        return "Learning rate in known-bad regime (too high or too low)"

    def check(self, model, config: TrainingConfig = None):
        if config is None:
            return []
        verdicts = []

        # Algorithm-specific LR ranges (empirical, from published baselines)
        lr_ranges = {
            "PPO": (1e-5, 1e-3, "3e-4"),
            "APPO": (1e-5, 1e-3, "3e-4"),
            "A2C": (1e-5, 1e-3, "7e-4"),
            "IMPALA": (1e-5, 1e-3, "3e-4"),
            "DQN": (1e-5, 5e-3, "1e-3"),
            "SAC": (1e-5, 1e-2, "3e-4"),
            "DDPG": (1e-5, 1e-2, "1e-3"),
            "TD3": (1e-5, 1e-2, "3e-4"),
        }
        algo = config.algorithm
        if algo in lr_ranges:
            lo, hi, default = lr_ranges[algo]
            if config.lr > hi:
                verdicts.append(
                    Verdict(
                        rule_name=self.name,
                        severity=Severity.WARNING,
                        message=(f"lr={config.lr:.0e} is high for {algo}. Standard range: {lo:.0e} to {hi:.0e}."),
                        recommendation=f"Try lr={default} as starting point",
                    )
                )
            elif config.lr < lo:
                verdicts.append(
                    Verdict(
                        rule_name=self.name,
                        severity=Severity.WARNING,
                        message=(f"lr={config.lr:.0e} is very low for {algo}. Training will be slow."),
                        recommendation=f"Try lr={default} unless deliberately fine-tuning",
                    )
                )

        return verdicts


class CriticLearningRate(Rule):
    """Check critic vs actor learning rate ratio.

    Konda & Tsitsiklis 2003 proved that two-timescale actor-critic
    convergence requires the critic to update on a faster timescale
    than the actor. When critic_lr >= actor_lr, the critic may
    saturate before the actor learns, killing the advantage signal.
    """

    @property
    def name(self):
        return "critic_lr_ratio"

    @property
    def description(self):
        return "Critic learning rate vs actor learning rate ratio"

    @property
    def proof(self):
        return FormalBasis(
            proof_name="two_timescale_convergence",
            strength=ProofStrength.MOTIVATED,
            paper="Konda & Tsitsiklis 2003",
            statement=(
                "Two-timescale stochastic approximation converges when "
                "the critic step size dominates the actor step size "
                "(α_critic/α_actor → ∞). Violating this breaks the "
                "separation of timescales required for convergence"
            ),
            parameters={"critic_lr": "α_critic", "actor_lr": "α_actor"},
        )

    def check(self, model, config: TrainingConfig = None):
        if config is None:
            return []
        verdicts = []

        if config.lr <= 0:
            return verdicts
        # Only flag if critic_lr was explicitly set (not defaulting to lr)
        if config.critic_lr is None:
            return verdicts
        critic_lr = config.critic_lr
        ratio = critic_lr / config.lr

        if ratio >= 1.0 and config.lr > 1e-5:
            verdicts.append(
                Verdict(
                    rule_name=self.name,
                    severity=Severity.WARNING,
                    message=(
                        f"Critic lr ({critic_lr:.0e}) >= actor lr ({config.lr:.0e}). "
                        f"Critic may saturate before actor learns, "
                        f"killing advantage signal (explained_var->1.0)."
                    ),
                    details={"ratio": ratio},
                    recommendation=f"Set critic_lr to {config.lr / 10:.0e} (10x lower)",
                )
            )

        return verdicts


class EntropyCollapse(Rule):
    """Check if entropy coefficient is too low or too high."""

    @property
    def name(self):
        return "entropy_regime"

    @property
    def description(self):
        return "Entropy coefficient regime analysis"

    def check(self, model, config: TrainingConfig = None):
        if config is None:
            return []
        verdicts = []

        if config.entropy_coeff > 0.1:
            verdicts.append(
                Verdict(
                    rule_name=self.name,
                    severity=Severity.WARNING,
                    message=(
                        f"entropy_coeff={config.entropy_coeff} is very high. "
                        f"Policy may stay near-random and never converge."
                    ),
                    recommendation="Standard range: 0.001 to 0.01",
                )
            )
        elif config.entropy_coeff < 0.0001:
            verdicts.append(
                Verdict(
                    rule_name=self.name,
                    severity=Severity.WARNING,
                    message=(
                        f"entropy_coeff={config.entropy_coeff} is very low. "
                        f"Risk of premature policy collapse -- agent stops "
                        f"exploring before finding optimal strategy."
                    ),
                    recommendation="Standard range: 0.001 to 0.01",
                )
            )

        if config.entropy_coeff_final == 0:
            verdicts.append(
                Verdict(
                    rule_name=self.name,
                    severity=Severity.WARNING,
                    message="entropy_coeff_final=0 means zero exploration at end "
                    "of training. Policy will become fully deterministic.",
                    recommendation="Set entropy_coeff_final to at least 0.001",
                )
            )

        return verdicts


class ClipFractionPrediction(Rule):
    """Predict if clip fraction will be problematic."""

    PPO_REFERENCE_LR = 3e-4
    PPO_REFERENCE_CLIP = 0.2

    @property
    def name(self):
        return "clip_fraction_risk"

    @property
    def description(self):
        return "Predict problematic clip fraction from hyperparameters"

    @property
    def proof(self):
        return FormalBasis(
            proof_name="ppo_clip_epoch_bound",
            strength=ProofStrength.MOTIVATED,
            statement="∀ lr, grad, adv, ε > 0: ε < lr * epochs * grad * adv → ε/(lr*grad*adv) < epochs",
            parameters={"lr": "lr", "num_epochs": "epochs", "clip_epsilon": "clip_ε"},
        )

    def check(self, model, config: TrainingConfig = None):
        if config is None:
            return []
        verdicts = []

        if config.clip_epsilon <= 0:
            return verdicts
        risk_score = (
            (config.lr / self.PPO_REFERENCE_LR) * config.num_epochs * (self.PPO_REFERENCE_CLIP / config.clip_epsilon)
        )

        if risk_score > 5:
            verdicts.append(
                Verdict(
                    rule_name=self.name,
                    severity=Severity.WARNING,
                    message=(
                        f"High clip fraction risk (score={risk_score:.1f}). "
                        f"lr={config.lr:.0e} x {config.num_epochs} epochs "
                        f"x clip={config.clip_epsilon} likely produces "
                        f">30% clipping."
                    ),
                    recommendation="Reduce lr, num_epochs, or increase clip_epsilon",
                )
            )

        if config.target_kl and config.target_kl < 0.01:
            verdicts.append(
                Verdict(
                    rule_name=self.name,
                    severity=Severity.WARNING,
                    message=(
                        f"target_kl={config.target_kl} is very tight. Most updates will be cut short, wasting compute."
                    ),
                    recommendation="Standard target_kl: 0.01 to 0.05",
                )
            )

        return verdicts


class ExpertCollapse(Rule):
    """Check if multi-specialist architecture will collapse."""

    @property
    def name(self):
        return "expert_collapse"

    @property
    def description(self):
        return "Multi-specialist expert collapse risk"

    @property
    def proof(self):
        return FormalBasis(
            proof_name="softmax_denom_decreases",
            strength=ProofStrength.MOTIVATED,
            statement="∀ δ₁ < δ₂, k > 0: 1 + k*exp(-δ₂) < 1 + k*exp(-δ₁) — softmax concentrates as gap grows",
            parameters={"logit_gap": "δ", "num_specialists": "k"},
        )

    def check(self, model, config: TrainingConfig = None):
        if config is None:
            return []
        verdicts = []

        if config.num_specialists <= 1:
            return verdicts

        if config.routing_floor == 0:
            verdicts.append(
                Verdict(
                    rule_name=self.name,
                    severity=Severity.CRITICAL,
                    message=(
                        f"{config.num_specialists} specialists with no routing "
                        f"floor. One specialist will dominate and others will "
                        f"receive zero gradient. Confirmed in multi-specialist experiments and "
                        f"MiniHack multi-specialist experiments."
                    ),
                    recommendation=(f"Set routing_floor to at least {1.0 / config.num_specialists / 3:.2f}"),
                )
            )

        if config.balance_coef == 0 and config.num_specialists > 2:
            verdicts.append(
                Verdict(
                    rule_name=self.name,
                    severity=Severity.WARNING,
                    message="No load balancing loss. Routing may become unbalanced.",
                    recommendation="Set balance_coef to 0.01",
                )
            )

        return verdicts


class BatchSizeInteraction(Rule):
    """Check batch size vs other parameters.

    The minibatch > transitions check is trivially correct: if the
    minibatch is larger than the rollout, zero gradient updates occur.
    The linear scaling rule (Goyal et al. 2017) establishes that
    batch size and learning rate must scale together.
    """

    @property
    def name(self):
        return "batch_size_interaction"

    @property
    def description(self):
        return "Batch size interactions with lr and epochs"

    @property
    def proof(self):
        return FormalBasis(
            proof_name="minibatch_exceeds_rollout",
            strength=ProofStrength.GROUNDED,
            paper="Goyal et al. 2017 (linear scaling rule)",
            statement=(
                "If minibatch_size > num_envs × rollout_length, "
                "the batch cannot fill a single minibatch and zero "
                "gradient updates occur. Trivially provable"
            ),
            parameters={"minibatch_size": "B", "transitions": "N"},
        )

    def check(self, model, config: TrainingConfig = None):
        if config is None:
            return []
        verdicts = []

        total_transitions = config.num_envs * config.rollout_length
        if config.minibatch_size <= 0:
            return verdicts
        n_minibatches = total_transitions / config.minibatch_size

        if n_minibatches < 1:
            verdicts.append(
                Verdict(
                    rule_name=self.name,
                    severity=Severity.CRITICAL,
                    message=(
                        f"Minibatch size ({config.minibatch_size}) > total "
                        f"transitions ({total_transitions}). No gradient updates."
                    ),
                    recommendation=f"Reduce minibatch_size to {total_transitions} or less",
                )
            )

        gradient_steps = n_minibatches * config.num_epochs
        if gradient_steps < 2:
            verdicts.append(
                Verdict(
                    rule_name=self.name,
                    severity=Severity.WARNING,
                    message=f"Only {gradient_steps:.0f} gradient steps per update. Very few.",
                    recommendation="Increase num_epochs or reduce minibatch_size",
                )
            )

        return verdicts


class ParallelismEffect(Rule):
    """Check if parallelism affects learning quality.

    With n actors and discovery probability p per actor per update,
    the probability of at least one discovery is 1 - (1-p)^n.
    When n*p << 1, expected discoveries per update ≈ n*p (binomial).
    """

    @property
    def name(self):
        return "parallelism_effect"

    @property
    def description(self):
        return "Effect of actor count on exploration and learning"

    @property
    def proof(self):
        return FormalBasis(
            proof_name="binomial_discovery_rate",
            strength=ProofStrength.GROUNDED,
            paper="Elementary probability (binomial)",
            statement=(
                "E[discoveries] = n × p. P(≥1 discovery) = 1 - (1-p)^n. "
                "Python checks n×p < 0.1 as threshold for sparse signal"
            ),
            parameters={"n_actors": "n", "discovery_prob": "p"},
        )

    def check(self, model, config: TrainingConfig = None):
        if config is None:
            return []
        verdicts = []

        total_actors = config.num_envs * max(config.num_workers, 1)

        if model.goal_sources:
            for source in model.goal_sources:
                if source.discovery_probability < 0.1:
                    discoveries_per_update = total_actors * source.discovery_probability
                    if discoveries_per_update < 0.1:
                        verdicts.append(
                            Verdict(
                                rule_name=self.name,
                                severity=Severity.WARNING,
                                message=(
                                    f"With {total_actors} actors and "
                                    f"p(goal)={source.discovery_probability:.3f}, "
                                    f"expect only {discoveries_per_update:.2f} goal "
                                    f"discoveries per update. Very sparse signal."
                                ),
                                details={
                                    "total_actors": total_actors,
                                    "discoveries_per_update": discoveries_per_update,
                                },
                                recommendation=(
                                    f"Need >={int(1 / max(source.discovery_probability, 1e-10))} "
                                    f"actors for ~1 discovery per update"
                                ),
                            )
                        )

        return verdicts


class MemoryCapacity(Rule):
    """Check if RNN/LSTM has sufficient capacity for the task."""

    @property
    def name(self):
        return "memory_capacity"

    @property
    def description(self):
        return "RNN memory capacity vs task memory demands"

    def check(self, model, config: TrainingConfig = None):
        if config is None:
            return []
        verdicts = []

        if not config.use_rnn:
            if model.n_states > 1000 and model.max_steps > 100:
                verdicts.append(
                    Verdict(
                        rule_name=self.name,
                        severity=Severity.WARNING,
                        message=(
                            f"Large state space ({model.n_states} states, "
                            f"{model.max_steps} steps) with no recurrence. "
                            f"Task may require memory for exploration."
                        ),
                        recommendation="Consider use_rnn=True with rnn_size=256",
                    )
                )
        else:
            bits_needed = int(max(1, (model.n_states).bit_length()))
            bits_available = config.rnn_size
            if bits_needed > bits_available:
                verdicts.append(
                    Verdict(
                        rule_name=self.name,
                        severity=Severity.INFO,
                        message=(
                            f"State space ({model.n_states} states) may "
                            f"exceed RNN capacity ({config.rnn_size}D hidden). "
                            f"Complex exploration patterns may not be remembered."
                        ),
                    )
                )

        return verdicts


class ReplayBufferRatio(Rule):
    """Check replay buffer size relative to state space and training budget."""

    @property
    def name(self):
        return "replay_buffer_ratio"

    @property
    def description(self):
        return "Replay buffer size relative to state space and training budget"

    def check(self, model, config: TrainingConfig = None):
        if config is None:
            return []
        if config.replay_buffer_size <= 0:
            return []  # on-policy algorithm, no buffer
        verdicts = []

        # Buffer should hold at least a few episodes
        min_episodes = 100
        min_buffer = min_episodes * model.max_steps
        if config.replay_buffer_size < min_buffer:
            verdicts.append(
                Verdict(
                    rule_name=self.name,
                    severity=Severity.WARNING,
                    message=(
                        f"Replay buffer ({config.replay_buffer_size:,}) holds "
                        f"<{min_episodes} episodes ({model.max_steps} steps each). "
                        f"May not have enough diversity for stable off-policy learning."
                    ),
                    recommendation=f"Increase replay_buffer_size to at least {min_buffer:,}",
                )
            )

        # Buffer vs training budget: buffer should be refilled multiple times
        if config.total_steps < config.replay_buffer_size:
            verdicts.append(
                Verdict(
                    rule_name=self.name,
                    severity=Severity.WARNING,
                    message=(
                        f"Training budget ({config.total_steps:,}) is smaller than "
                        f"replay buffer ({config.replay_buffer_size:,}). "
                        f"Buffer will never be fully utilized."
                    ),
                    recommendation="Increase total_steps or reduce replay_buffer_size",
                )
            )

        return verdicts


class TargetNetworkUpdate(Rule):
    """Check target network update frequency for off-policy algorithms."""

    @property
    def name(self):
        return "target_network_update"

    @property
    def description(self):
        return "Target network update frequency for DQN-family algorithms"

    def check(self, model, config: TrainingConfig = None):
        if config is None:
            return []
        if config.target_update_freq <= 0:
            return []  # no target network
        verdicts = []

        # Hard update (DQN-style): freq should be substantial
        if config.algorithm == "DQN":
            if config.target_update_freq < 100:
                verdicts.append(
                    Verdict(
                        rule_name=self.name,
                        severity=Severity.WARNING,
                        message=(
                            f"Target network updates every {config.target_update_freq} steps. "
                            f"Too frequent → target tracks online network too closely, "
                            f"defeating the purpose of a stable target."
                        ),
                        recommendation="Standard: 1000-10000 steps between hard updates",
                    )
                )
            elif config.target_update_freq > config.total_steps / 10:
                verdicts.append(
                    Verdict(
                        rule_name=self.name,
                        severity=Severity.WARNING,
                        message=(
                            f"Target network updates every {config.target_update_freq} steps "
                            f"with {config.total_steps:,} total. Fewer than 10 updates total."
                        ),
                        recommendation="Increase total_steps or reduce target_update_freq",
                    )
                )

        return verdicts


class EpsilonSchedule(Rule):
    """Check epsilon-greedy exploration schedule for DQN."""

    @property
    def name(self):
        return "epsilon_schedule"

    @property
    def description(self):
        return "Epsilon-greedy exploration schedule vs training budget"

    def check(self, model, config: TrainingConfig = None):
        if config is None:
            return []
        if config.epsilon_decay_steps <= 0:
            return []  # no epsilon schedule
        verdicts = []

        # Epsilon should decay over a meaningful fraction of training
        if config.epsilon_decay_steps < config.total_steps * 0.05:
            verdicts.append(
                Verdict(
                    rule_name=self.name,
                    severity=Severity.WARNING,
                    message=(
                        f"Epsilon decays in {config.epsilon_decay_steps:,} steps "
                        f"({100 * config.epsilon_decay_steps / config.total_steps:.0f}% "
                        f"of training). Exploration ends very early — agent exploits "
                        f"before discovering the full state space."
                    ),
                    recommendation=(
                        f"Decay over at least 10-50% of training: "
                        f"{int(config.total_steps * 0.1):,} to "
                        f"{int(config.total_steps * 0.5):,} steps"
                    ),
                )
            )
        elif config.epsilon_decay_steps > config.total_steps * 0.9:
            verdicts.append(
                Verdict(
                    rule_name=self.name,
                    severity=Severity.INFO,
                    message=(
                        f"Epsilon decays over {100 * config.epsilon_decay_steps / config.total_steps:.0f}% "
                        f"of training. Agent explores for nearly the entire run."
                    ),
                )
            )

        # Final epsilon should be > 0 for continued exploration
        if config.epsilon_end <= 0:
            verdicts.append(
                Verdict(
                    rule_name=self.name,
                    severity=Severity.WARNING,
                    message="epsilon_end=0 means zero exploration after decay. Agent can never discover new states.",
                    recommendation="Set epsilon_end >= 0.01 for residual exploration",
                )
            )

        return verdicts


class SoftUpdateRate(Rule):
    """Check soft update coefficient (tau) for SAC/DDPG/TD3."""

    @property
    def name(self):
        return "soft_update_rate"

    @property
    def description(self):
        return "Soft update coefficient (tau) for off-policy actor-critic"

    def check(self, model, config: TrainingConfig = None):
        if config is None:
            return []
        if config.algorithm not in ("SAC", "DDPG", "TD3"):
            return []
        verdicts = []

        if config.tau > 0.1:
            verdicts.append(
                Verdict(
                    rule_name=self.name,
                    severity=Severity.WARNING,
                    message=(
                        f"tau={config.tau} is high for {config.algorithm}. "
                        f"Target network tracks online network too closely. "
                        f"Standard range: 0.001 to 0.01."
                    ),
                    recommendation="Try tau=0.005 (SAC/TD3 default)",
                )
            )
        elif config.tau < 1e-4:
            verdicts.append(
                Verdict(
                    rule_name=self.name,
                    severity=Severity.INFO,
                    message=(
                        f"tau={config.tau} is very small. Target network updates slowly — stable but may lag behind."
                    ),
                )
            )

        return verdicts


class SACAlpha(Rule):
    """Check SAC entropy temperature."""

    @property
    def name(self):
        return "sac_alpha"

    @property
    def description(self):
        return "SAC entropy temperature (alpha) configuration"

    def check(self, model, config: TrainingConfig = None):
        if config is None:
            return []
        if config.algorithm != "SAC":
            return []
        verdicts = []

        if config.auto_alpha:
            verdicts.append(
                Verdict(
                    rule_name=self.name,
                    severity=Severity.INFO,
                    message="auto_alpha=True: entropy temperature will be learned. "
                    "This is the recommended SAC configuration.",
                )
            )
        else:
            if config.alpha > 1.0:
                verdicts.append(
                    Verdict(
                        rule_name=self.name,
                        severity=Severity.WARNING,
                        message=(
                            f"alpha={config.alpha} is high. SAC will prioritize "
                            f"entropy over reward — agent may act randomly."
                        ),
                        recommendation="Standard: alpha=0.2. Consider auto_alpha=True.",
                    )
                )
            elif config.alpha < 0.01:
                verdicts.append(
                    Verdict(
                        rule_name=self.name,
                        severity=Severity.WARNING,
                        message=(
                            f"alpha={config.alpha} is very low. Minimal entropy "
                            f"regularization — SAC loses its exploration advantage."
                        ),
                        recommendation="Standard: alpha=0.2. Consider auto_alpha=True.",
                    )
                )

        return verdicts


# Standard training rule collection
TRAINING_RULES = [
    LearningRateRegime(),
    CriticLearningRate(),
    EntropyCollapse(),
    ClipFractionPrediction(),
    ExpertCollapse(),
    BatchSizeInteraction(),
    ParallelismEffect(),
    MemoryCapacity(),
    # Off-policy rules
    ReplayBufferRatio(),
    TargetNetworkUpdate(),
    EpsilonSchedule(),
    SoftUpdateRate(),
    SACAlpha(),
]
