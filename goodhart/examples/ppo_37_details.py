"""Example: Catching PPO implementation pitfalls.

Based on "The 37 Implementation Details of Proximal Policy Optimization"
(ICLR Blog Track, 2022) and Andy Jones' "Debugging RL Systems."

Shows how to express PPO-specific failure modes using the framework
and add project-specific rules for issues the standard library
doesn't cover.
"""

from goodhart.models import *
from goodhart.engine import *
from goodhart.rules.reward import *
from goodhart.rules.training import *
from goodhart.rules.architecture import PrecedentRule, Precedent


# ---- Project-specific rules from the PPO details paper ----

class AdamEpsilon(PrecedentRule):
    """Adam optimizer epsilon matters for PPO reproducibility."""

    @property
    def name(self): return "adam_epsilon"
    @property
    def description(self): return "Adam epsilon setting for PPO"
    @property
    def precedents(self):
        return [Precedent(
            source="37 Implementation Details of PPO (ICLR Blog 2022)",
            setting="Adam eps=1e-5 (PPO default) vs eps=1e-8 (PyTorch default)",
            outcome="Results differ. PPO implementations use eps=1e-5 but this "
                    "is 'neither mentioned in the paper nor a configurable parameter.'",
            year=2022,
        )]

    def check(self, model, config=None):
        # Can't check without optimizer config, but warn about it
        return [Verdict(
            rule_name=self.name,
            severity=Severity.INFO,
            message="Verify Adam epsilon=1e-5 for PPO compatibility.",
            recommendation=f"Precedent: {self.precedents[0].outcome}",
        )]


class ObservationScaling(PrecedentRule):
    """Image observations must be scaled to [0,1]."""

    @property
    def name(self): return "observation_scaling"
    @property
    def description(self): return "Observation normalization for stability"
    @property
    def precedents(self):
        return [Precedent(
            source="37 Implementation Details of PPO",
            setting="Atari frames not divided by 255",
            outcome="'First policy update results in KL divergence explosion.' "
                    "Layer initialization expects inputs near [0,1].",
            year=2022,
        )]

    def check(self, model, config=None):
        # Framework placeholder — would need obs_range in config
        return []


class ValueTargetRange(PrecedentRule):
    """Value targets should be in reasonable range."""

    @property
    def name(self): return "value_target_range"
    @property
    def description(self): return "Value function target magnitude"
    @property
    def precedents(self):
        return [
            Precedent(
                source="Andy Jones — Debugging RL Systems",
                setting="Rewards not scaled; value targets outside [-10, +10]",
                outcome="Neural network unable to learn value function. "
                        "Loss explodes or converges to mean.",
                year=2020,
            ),
            Precedent(
                source="37 Implementation Details of PPO",
                setting="Reward clipping to [-1, +1] in Atari",
                outcome="Standard practice to keep value targets in range.",
                year=2022,
            ),
        ]

    def check(self, model, config=None):
        verdicts = []
        # Estimate value target range from reward structure
        max_return = 0
        for source in model.reward_sources:
            if source.reward_type == RewardType.PER_STEP:
                max_return += abs(source.value) * model.max_steps
            elif source.reward_type == RewardType.TERMINAL:
                max_return += abs(source.value)

        if max_return > 100:
            verdicts.append(Verdict(
                rule_name=self.name,
                severity=Severity.WARNING,
                message=(f"Estimated max return ~{max_return:.0f}. Value targets "
                         f"may be too large for stable learning."),
                recommendation=(
                    f"Precedent: Jones recommends targets in [-10, +10]. "
                    f"Consider reward scaling by {10/max_return:.4f}."
                ),
            ))
        return verdicts


class KLDivergenceThreshold(PrecedentRule):
    """KL divergence should stay below known thresholds."""

    @property
    def name(self): return "kl_divergence_prediction"
    @property
    def description(self): return "Predicted KL divergence regime"
    @property
    def precedents(self):
        return [
            Precedent(
                source="37 Implementation Details of PPO",
                setting="approx_kl > 0.02 with standard PPO",
                outcome="'Usually means policy is changing too quickly "
                        "and there is a bug.'",
                year=2022,
            ),
            Precedent(
                source="Andy Jones — Debugging RL Systems",
                setting="KL > 0.5 in PPO on-policy",
                outcome="Experience too stale. Actor and learner running "
                        "at incompatible speeds.",
                year=2020,
            ),
            Precedent(
                source="Empirical — KL stabilization",
                setting="target_kl=0.5 with routing-as-policy",
                outcome="Routing logits add entropy that inflates KL. "
                        "Higher threshold needed for multi-action policies.",
                year=2026,
            ),
        ]

    def check(self, model, config=None):
        if config is None:
            return []
        verdicts = []

        # Predict KL regime from lr and epochs
        # Higher lr * more epochs = more policy change = higher KL
        if config.target_kl is None and config.lr > 1e-4 and config.num_epochs > 3:
            verdicts.append(Verdict(
                rule_name=self.name,
                severity=Severity.WARNING,
                message=(f"No target_kl with lr={config.lr:.0e} and "
                         f"{config.num_epochs} epochs. KL may exceed 0.02 "
                         f"(PPO blog threshold) causing policy instability."),
                recommendation=(
                    f"Precedent: PPO blog says approx_kl > 0.02 usually "
                    f"means a bug. Jones says KL > 0.5 means stale experience. "
                    f"Set target_kl=0.03 for safety."
                ),
            ))

        return verdicts


class CorrelatedEnvironments(PrecedentRule):
    """Parallel environments starting from identical states."""

    @property
    def name(self): return "correlated_envs"
    @property
    def description(self): return "Correlated initial states in parallel envs"
    @property
    def precedents(self):
        return [Precedent(
            source="Andy Jones — Debugging RL Systems",
            setting="All parallel envs start from same state, no warm-up",
            outcome="Learner sees highly correlated batches. Optimizes "
                    "for specific phase of episode (steps 0-10, then 10-20) "
                    "instead of general policy.",
            year=2020,
        )]

    def check(self, model, config=None):
        # Would need seed/warm-up info in config
        return [Verdict(
            rule_name=self.name,
            severity=Severity.INFO,
            message="Verify parallel environments are desynchronized.",
            recommendation=(
                f"Precedent: Jones — correlated env starts cause learner to "
                f"optimize for specific episode phases. Run N random warm-up "
                f"steps in each env before collection."
            ),
        )]


# ---- Example usage ----

def run_example():
    """Demonstrate catching PPO pitfalls on a standard Atari setup."""

    # Define an Atari-like environment
    model = EnvironmentModel(
        name="Atari Breakout (example)",
        max_steps=10000,
        n_states=100000,
        n_actions=4,
    )
    model.add_reward_source(RewardSource(
        name="score", reward_type=RewardType.ON_EVENT, value=1.0,
        respawn=RespawnBehavior.NONE, max_occurrences=300,
    ))

    # Deliberately wrong config to show what gets caught
    config = TrainingConfig(
        algorithm="PPO",
        lr=1e-3,              # too high for PPO
        critic_lr=1e-3,       # same as actor
        entropy_coeff=0.5,    # way too high
        num_epochs=10,        # too many
        clip_epsilon=0.1,     # tight clip
        minibatch_size=32,    # tiny
        num_envs=4,           # few
        rollout_length=128,
    )

    # Build engine with standard + project-specific rules
    engine = TrainingAnalysisEngine().add_all_rules()
    engine.add_rules([
        AdamEpsilon(),
        ObservationScaling(),
        ValueTargetRange(),
        KLDivergenceThreshold(),
        CorrelatedEnvironments(),
    ])

    engine.print_report(model, config)


if __name__ == "__main__":
    run_example()
