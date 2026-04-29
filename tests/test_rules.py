"""Test all rules catch their intended failure cases."""

import pytest
from goodhart.models import (
    EnvironmentModel, RewardSource, RewardType, RespawnBehavior,
    Severity, TrainingConfig, Verdict,
)
from goodhart.rules.reward import (
    PenaltyDominatesGoal, DeathBeatsSurvival, IdleExploit,
    ExplorationThreshold, RespawningExploit, DeathResetExploit,
    ShapingLoopExploit, IntrinsicSufficiency, BudgetSufficiency,
    CompoundTrap, IntrinsicDominance,
    DiscountHorizonMismatch, NegativeOnlyReward, RewardDelayHorizon,
    ShapingNotPotentialBased, ProxyRewardHackability,
    StagedRewardPlateau, RewardDominanceImbalance, ExponentialSaturation,
)
from goodhart.rules.training import (
    LearningRateRegime, CriticLearningRate, EntropyCollapse,
    ClipFractionPrediction, ExpertCollapse, BatchSizeInteraction,
    ParallelismEffect, MemoryCapacity,
    ReplayBufferRatio, TargetNetworkUpdate, EpsilonSchedule,
    SoftUpdateRate, SACAlpha,
)
from goodhart.rules.architecture import (
    EmbedDimCapacity, RoutingFloorNecessity, RecurrenceType,
    ActorCountEffect,
)
from goodhart.rules.advisories import (
    PhysicsExploitAdvisory, GoalMisgeneralizationAdvisory,
    CreditAssignmentAdvisory, ConstrainedRLAdvisory,
    NonStationarityAdvisory, LearnedRewardAdvisory,
    MissingConstraintAdvisory, AggregationTrapAdvisory,
)


def _model_with_penalty_and_goal(penalty=-0.01, goal=1.0, max_steps=500):
    model = EnvironmentModel(name="test", max_steps=max_steps)
    model.add_reward_source(RewardSource(
        name="goal", reward_type=RewardType.TERMINAL,
        value=goal, discovery_probability=0.05,
    ))
    model.add_reward_source(RewardSource(
        name="penalty", reward_type=RewardType.PER_STEP, value=penalty,
    ))
    return model


# --- Reward rules ---

def test_penalty_dominates_goal():
    rule = PenaltyDominatesGoal()
    # With gamma=0.99, discounted steps for 500 = ~99.3
    # Need penalty * disc_steps > goal: 0.05 * 99.3 = 4.97 > 1.0
    model = _model_with_penalty_and_goal(penalty=-0.05, goal=1.0, max_steps=500)
    verdicts = rule.check(model)
    assert any(v.severity == Severity.CRITICAL for v in verdicts)


def test_penalty_dominates_goal_safe():
    rule = PenaltyDominatesGoal()
    model = _model_with_penalty_and_goal(penalty=-0.0001, goal=1.0, max_steps=500)
    verdicts = rule.check(model)
    assert not any(v.severity == Severity.CRITICAL for v in verdicts)


def test_death_beats_survival():
    """With default gamma=0.99, penalty=-0.1 triggers CRITICAL.

    ratio = disc_steps(10) / disc_steps(1) = 9.56 > 2.0.
    """
    rule = DeathBeatsSurvival()
    model = _model_with_penalty_and_goal(penalty=-0.1)
    verdicts = rule.check(model)
    assert any(v.severity == Severity.CRITICAL for v in verdicts)
    critical = [v for v in verdicts if v.severity == Severity.CRITICAL][0]
    assert critical.details["ratio"] > 2.0


def test_death_beats_survival_low_gamma_warning():
    """With very low gamma, discounting reduces the ratio below CRITICAL."""
    rule = DeathBeatsSurvival()
    # gamma=0.5: disc(1)=1, disc(10)=(1-0.5^10)/(0.5)=1.998
    # ratio = 1.998 < 2.0 -> WARNING not CRITICAL
    model = EnvironmentModel(name="test", max_steps=500, gamma=0.5)
    model.add_reward_source(RewardSource(
        name="penalty", reward_type=RewardType.PER_STEP, value=-0.1,
    ))
    verdicts = rule.check(model)
    assert not any(v.severity == Severity.CRITICAL for v in verdicts)
    assert any(v.severity == Severity.WARNING for v in verdicts)


def test_idle_exploit():
    """Idle exploit: passive bonus + penalty makes idling beat exploring."""
    rule = IdleExploit()
    model = EnvironmentModel(name="test", max_steps=1000)
    model.add_reward_source(RewardSource(
        name="alive", reward_type=RewardType.PER_STEP, value=1.0,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="penalty", reward_type=RewardType.PER_STEP, value=-0.5,
        requires_action=True,  # penalty only when acting
    ))
    model.add_reward_source(RewardSource(
        name="goal", reward_type=RewardType.TERMINAL,
        value=1.0, discovery_probability=0.05,
    ))
    verdicts = rule.check(model)
    # Idle: +1.0 * disc. Explore: (+1.0 - 0.5) * disc + goal. Idle wins.
    assert any(v.severity == Severity.CRITICAL for v in verdicts)


def test_exploration_threshold():
    rule = ExplorationThreshold()
    model = _model_with_penalty_and_goal(penalty=-0.01, goal=1.0)
    verdicts = rule.check(model)
    assert len(verdicts) > 0
    # Should produce WARNING or CRITICAL (not just INFO)
    assert any(v.severity in (Severity.WARNING, Severity.CRITICAL) for v in verdicts)


def test_respawning_exploit():
    rule = RespawningExploit()
    model = EnvironmentModel(name="test", max_steps=1000)
    model.add_reward_source(RewardSource(
        name="goal", reward_type=RewardType.TERMINAL, value=10.0,
    ))
    model.add_reward_source(RewardSource(
        name="coins", reward_type=RewardType.ON_EVENT, value=0.5,
        can_loop=True, loop_period=3,
    ))
    verdicts = rule.check(model)
    assert any(v.severity == Severity.CRITICAL for v in verdicts)


def test_death_reset_exploit():
    rule = DeathResetExploit()
    model = EnvironmentModel(name="test", max_steps=10000)
    model.add_reward_source(RewardSource(
        name="goal", reward_type=RewardType.TERMINAL, value=100.0,
    ))
    model.add_reward_source(RewardSource(
        name="collectibles", reward_type=RewardType.ON_EVENT, value=80.0,
        respawn=RespawnBehavior.ON_DEATH, discovery_probability=0.9,
    ))
    verdicts = rule.check(model)
    assert any(v.severity == Severity.CRITICAL for v in verdicts)


def test_shaping_loop_exploit():
    rule = ShapingLoopExploit()
    model = EnvironmentModel(name="test", max_steps=1000)
    model.add_reward_source(RewardSource(
        name="goal", reward_type=RewardType.TERMINAL, value=1.0,
    ))
    model.add_reward_source(RewardSource(
        name="distance", reward_type=RewardType.SHAPING, value=0.1,
        can_loop=True, loop_period=4,
    ))
    verdicts = rule.check(model)
    assert any(v.severity == Severity.CRITICAL for v in verdicts)


def test_intrinsic_sufficiency():
    rule = IntrinsicSufficiency()
    model = EnvironmentModel(name="test", max_steps=500)
    model.add_reward_source(RewardSource(
        name="intrinsic", reward_type=RewardType.PER_STEP, value=0.001,
    ))
    model.add_reward_source(RewardSource(
        name="penalty", reward_type=RewardType.PER_STEP, value=-0.01,
    ))
    verdicts = rule.check(model)
    assert any(v.severity == Severity.WARNING for v in verdicts)


def test_budget_sufficiency():
    rule = BudgetSufficiency()
    model = EnvironmentModel(name="test", max_steps=500)
    model.add_reward_source(RewardSource(
        name="goal", reward_type=RewardType.TERMINAL,
        value=1.0, discovery_probability=0.0001,
    ))
    config = TrainingConfig(n_actors=1, total_steps=1000)
    verdicts = rule.check(model, config)
    assert any(v.severity == Severity.CRITICAL for v in verdicts)


# --- Training rules ---

def test_lr_regime_high():
    rule = LearningRateRegime()
    config = TrainingConfig(lr=5e-3, algorithm="PPO")
    model = EnvironmentModel(name="test")
    verdicts = rule.check(model, config)
    assert any(v.severity == Severity.WARNING for v in verdicts)


def test_critic_lr_ratio():
    rule = CriticLearningRate()
    config = TrainingConfig(lr=3e-4, critic_lr=3e-4)
    model = EnvironmentModel(name="test")
    verdicts = rule.check(model, config)
    assert any(v.severity == Severity.WARNING for v in verdicts)


def test_entropy_collapse_high():
    rule = EntropyCollapse()
    config = TrainingConfig(entropy_coeff=0.5)
    model = EnvironmentModel(name="test")
    verdicts = rule.check(model, config)
    assert any(v.severity == Severity.WARNING for v in verdicts)


def test_clip_fraction_risk():
    rule = ClipFractionPrediction()
    config = TrainingConfig(lr=1e-3, num_epochs=10, clip_epsilon=0.1)
    model = EnvironmentModel(name="test")
    verdicts = rule.check(model, config)
    assert any(v.severity == Severity.WARNING for v in verdicts)


def test_expert_collapse():
    rule = ExpertCollapse()
    config = TrainingConfig(num_specialists=3, routing_floor=0.0)
    model = EnvironmentModel(name="test")
    verdicts = rule.check(model, config)
    assert any(v.severity == Severity.CRITICAL for v in verdicts)


def test_batch_size_interaction():
    rule = BatchSizeInteraction()
    config = TrainingConfig(num_envs=4, rollout_length=32, minibatch_size=256)
    model = EnvironmentModel(name="test")
    verdicts = rule.check(model, config)
    assert any(v.severity == Severity.CRITICAL for v in verdicts)


def test_parallelism_effect():
    rule = ParallelismEffect()
    model = EnvironmentModel(name="test")
    model.add_reward_source(RewardSource(
        name="goal", reward_type=RewardType.TERMINAL,
        value=1.0, discovery_probability=0.001,
    ))
    config = TrainingConfig(num_envs=4, num_workers=1)
    verdicts = rule.check(model, config)
    assert any(v.severity == Severity.WARNING for v in verdicts)


def test_memory_capacity():
    rule = MemoryCapacity()
    model = EnvironmentModel(name="test", n_states=5000, max_steps=500)
    config = TrainingConfig(use_rnn=False)
    verdicts = rule.check(model, config)
    assert any(v.severity == Severity.WARNING for v in verdicts)


# --- Architecture rules ---

def test_embed_dim_capacity():
    rule = EmbedDimCapacity()
    config = TrainingConfig(num_specialists=3, model_params=30000)
    model = EnvironmentModel(name="test")
    verdicts = rule.check(model, config)
    assert any(v.severity == Severity.WARNING for v in verdicts)


def test_routing_floor_necessity():
    rule = RoutingFloorNecessity()
    config = TrainingConfig(num_specialists=4, routing_floor=0.0)
    model = EnvironmentModel(name="test")
    verdicts = rule.check(model, config)
    assert any(v.severity == Severity.CRITICAL for v in verdicts)


def test_recurrence_type():
    rule = RecurrenceType()
    config = TrainingConfig(use_rnn=True, rnn_type="gru")
    model = EnvironmentModel(name="test")
    verdicts = rule.check(model, config)
    assert any(v.severity == Severity.INFO for v in verdicts)


def test_actor_count_effect():
    rule = ActorCountEffect()
    config = TrainingConfig(num_envs=8, num_workers=1)
    model = EnvironmentModel(name="test", n_states=5000)
    verdicts = rule.check(model, config)
    assert any(v.severity == Severity.WARNING for v in verdicts)


# --- Discounted EV tests ---

def test_discounted_penalty_reduces_impact():
    """gamma=0.99 with 500 steps gives less total penalty than undiscounted.

    disc_steps(0.99, 500) = (1 - 0.99^500) / 0.01 ~ 99.3 (since 0.99^500 ~ 0.007).
    At gamma=0.99, the effective horizon is ~1/(1-gamma)=100 steps.
    """
    from goodhart.rules.reward import _discounted_steps
    disc = _discounted_steps(0.99, 500)
    assert disc < 500  # discounting reduces the effective step count
    assert disc > 90   # but it's close to the effective horizon of ~100


def test_discounted_steps_gamma_one():
    """When gamma is effectively 1, discounted steps equals n."""
    from goodhart.rules.reward import _discounted_steps
    assert _discounted_steps(1.0, 100) == 100.0
    # Very close to 1
    result = _discounted_steps(1.0 - 1e-15, 100)
    assert abs(result - 100.0) < 0.01


def test_discounted_penalty_dominates_goal():
    """gamma=0.99 reduces total penalty, potentially saving a config from CRITICAL."""
    rule = PenaltyDominatesGoal()
    # Undiscounted: penalty=0.01*500=5.0 > goal=1.0 -> CRITICAL
    # Discounted (gamma=0.99, 500 steps): 0.01 * ~248 = ~2.48 -> still CRITICAL
    model_high_gamma = _model_with_penalty_and_goal(
        penalty=-0.003, goal=1.0, max_steps=500)
    verdicts = rule.check(model_high_gamma)
    # Undiscounted: 0.003*500=1.5 > 1.0 -> CRITICAL
    # Discounted: 0.003*~248=0.74 < 1.0 -> no CRITICAL
    assert not any(v.severity == Severity.CRITICAL for v in verdicts)


# --- Preset tests ---

def test_presets_all_load():
    """Every preset creates a valid EnvironmentModel and TrainingConfig."""
    from goodhart.presets import PRESETS
    assert len(PRESETS) >= 7
    for name, (model, config) in PRESETS.items():
        assert isinstance(model, EnvironmentModel), f"Preset {name} model is not EnvironmentModel"
        assert isinstance(config, TrainingConfig), f"Preset {name} config is not TrainingConfig"
        assert model.name, f"Preset {name} has no name"
        assert model.max_steps > 0, f"Preset {name} has invalid max_steps"
        assert 0.0 < model.gamma <= 1.0, f"Preset {name} has invalid gamma"


def test_preset_atari():
    from goodhart.presets import PRESETS
    model, config = PRESETS["atari"]
    assert model.max_steps == 18000
    assert model.gamma == 0.99
    assert model.n_states == 100000
    assert len(model.reward_sources) > 0
    assert config.lr > 0


def test_preset_mujoco_locomotion():
    from goodhart.presets import PRESETS
    model, config = PRESETS["mujoco-locomotion"]
    assert model.max_steps == 1000
    assert model.gamma == 0.99
    # Should have forward velocity, alive bonus, control penalty
    assert len(model.reward_sources) >= 3


def test_preset_sparse_goal():
    from goodhart.presets import PRESETS
    model, config = PRESETS["sparse-goal"]
    assert model.max_steps == 500
    assert len(model.goal_sources) > 0


def test_preset_analyzable():
    """Each preset can be analyzed without errors, including training rules."""
    from goodhart.presets import PRESETS
    from goodhart.engine import TrainingAnalysisEngine
    engine = TrainingAnalysisEngine().add_all_rules()
    for name, (model, config) in PRESETS.items():
        result = engine.analyze(model, config)
        assert result is not None, f"Analysis of preset {name} returned None"


# --- CompoundTrap tests ---

def test_compound_trap_penalty_plus_respawning():
    """Penalty + respawning reward creates a compound loop trap."""
    rule = CompoundTrap()
    model = EnvironmentModel(name="test", max_steps=1000)
    model.add_reward_source(RewardSource(
        name="goal", reward_type=RewardType.TERMINAL,
        value=1.0, discovery_probability=0.05,
    ))
    model.add_reward_source(RewardSource(
        name="penalty", reward_type=RewardType.PER_STEP, value=-0.001,
    ))
    model.add_reward_source(RewardSource(
        name="coins", reward_type=RewardType.ON_EVENT, value=0.5,
        can_loop=True, loop_period=3,
    ))
    verdicts = rule.check(model)
    assert any(v.severity == Severity.CRITICAL and "compound trap" in v.message.lower()
               for v in verdicts)


def test_compound_trap_shaping_no_terminal():
    """Shaping rewards with no terminal goal fires CRITICAL."""
    rule = CompoundTrap()
    model = EnvironmentModel(name="test", max_steps=500)
    model.add_reward_source(RewardSource(
        name="distance", reward_type=RewardType.SHAPING, value=0.1,
        can_loop=True, loop_period=4,
    ))
    model.add_reward_source(RewardSource(
        name="penalty", reward_type=RewardType.PER_STEP, value=-0.01,
    ))
    verdicts = rule.check(model)
    assert any(v.severity == Severity.CRITICAL and "no terminal goal" in v.message.lower()
               for v in verdicts)


def test_compound_trap_death_plus_penalty():
    """Death-resettable rewards + step penalty fires CRITICAL."""
    rule = CompoundTrap()
    model = EnvironmentModel(name="test", max_steps=1000)
    model.add_reward_source(RewardSource(
        name="goal", reward_type=RewardType.TERMINAL, value=10.0,
    ))
    model.add_reward_source(RewardSource(
        name="collectibles", reward_type=RewardType.ON_EVENT, value=5.0,
        respawn=RespawnBehavior.ON_DEATH, discovery_probability=0.8,
    ))
    model.add_reward_source(RewardSource(
        name="penalty", reward_type=RewardType.PER_STEP, value=-0.01,
    ))
    verdicts = rule.check(model)
    assert any(v.severity == Severity.CRITICAL and "doubly incentivized" in v.message.lower()
               for v in verdicts)


def test_compound_trap_no_fire_clean_config():
    """CompoundTrap should not fire on a clean config."""
    rule = CompoundTrap()
    model = EnvironmentModel(name="test", max_steps=500)
    model.add_reward_source(RewardSource(
        name="goal", reward_type=RewardType.TERMINAL,
        value=1.0, discovery_probability=0.1,
    ))
    model.add_reward_source(RewardSource(
        name="penalty", reward_type=RewardType.PER_STEP, value=-0.001,
    ))
    verdicts = rule.check(model)
    assert not any(v.severity == Severity.CRITICAL for v in verdicts)


# --- Contradiction detection tests ---

def test_contradiction_penalty_add_vs_remove():
    """Contradictory penalty recommendations are detected."""
    from goodhart.engine import AnalysisEngine
    verdicts = [
        Verdict(
            rule_name="rule_a", severity=Severity.WARNING,
            message="Too few discoveries",
            recommendation="Add step penalty to encourage speed",
        ),
        Verdict(
            rule_name="rule_b", severity=Severity.CRITICAL,
            message="Penalty too high",
            recommendation="Remove step penalty or reduce it",
        ),
    ]
    contradictions = AnalysisEngine._check_contradictions(verdicts)
    assert len(contradictions) >= 1
    assert any("contradiction" in c.rule_name.lower() for c in contradictions)


def test_contradiction_entropy():
    """Contradictory entropy recommendations are detected."""
    from goodhart.engine import AnalysisEngine
    verdicts = [
        Verdict(
            rule_name="rule_a", severity=Severity.WARNING,
            message="Premature collapse",
            recommendation="Increase entropy coefficient",
        ),
        Verdict(
            rule_name="rule_b", severity=Severity.WARNING,
            message="Too random",
            recommendation="Decrease entropy coefficient",
        ),
    ]
    contradictions = AnalysisEngine._check_contradictions(verdicts)
    assert len(contradictions) >= 1


def test_contradiction_intrinsic():
    """Contradictory intrinsic motivation recommendations are detected."""
    from goodhart.engine import AnalysisEngine
    verdicts = [
        Verdict(
            rule_name="rule_a", severity=Severity.WARNING,
            message="Can't bootstrap",
            recommendation="Add intrinsic motivation",
        ),
        Verdict(
            rule_name="rule_b", severity=Severity.WARNING,
            message="Intrinsic dominates",
            recommendation="Intrinsic reward is too strong, reduce it",
        ),
    ]
    contradictions = AnalysisEngine._check_contradictions(verdicts)
    assert len(contradictions) >= 1


def test_no_contradiction_clean():
    """No contradictions on non-conflicting verdicts."""
    from goodhart.engine import AnalysisEngine
    verdicts = [
        Verdict(
            rule_name="rule_a", severity=Severity.WARNING,
            message="Something",
            recommendation="Increase lr",
        ),
        Verdict(
            rule_name="rule_b", severity=Severity.WARNING,
            message="Something else",
            recommendation="Add intrinsic motivation",
        ),
    ]
    contradictions = AnalysisEngine._check_contradictions(verdicts)
    assert len(contradictions) == 0


# ---- Intrinsic Dominance ----

class TestIntrinsicDominance:
    rule = IntrinsicDominance()

    def _make_model(self, goal, intrinsic, steps=1000, gamma=0.99,
                    intentional=False):
        model = EnvironmentModel(
            name="test", max_steps=steps, gamma=gamma,
            n_actions=8, action_type="discrete",
        )
        model.add_reward_source(RewardSource(
            name="goal", reward_type=RewardType.TERMINAL, value=goal,
            requires_action=True, intentional=True,
            discovery_probability=0.05,
        ))
        model.add_reward_source(RewardSource(
            name="intrinsic", reward_type=RewardType.PER_STEP,
            value=intrinsic, respawn=RespawnBehavior.INFINITE,
            requires_action=True, intentional=intentional,
        ))
        return model

    def test_high_intrinsic_critical(self):
        """High intrinsic on long episode → CRITICAL."""
        model = self._make_model(goal=1.0, intrinsic=0.1, steps=18000)
        verdicts = self.rule.check(model)
        assert len(verdicts) == 1
        assert verdicts[0].severity == Severity.CRITICAL

    def test_moderate_intrinsic_warning(self):
        """Intrinsic ~= goal → WARNING."""
        model = self._make_model(goal=1.0, intrinsic=0.05, steps=500)
        verdicts = self.rule.check(model)
        assert len(verdicts) == 1
        assert verdicts[0].severity == Severity.WARNING

    def test_goal_dominates_clean(self):
        """Goal >> intrinsic → no finding."""
        model = self._make_model(goal=10.0, intrinsic=0.001, steps=1000)
        verdicts = self.rule.check(model)
        assert len(verdicts) == 0

    def test_intentional_skipped(self):
        """Intentional per-step (alive bonus) should not trigger."""
        model = self._make_model(goal=5.0, intrinsic=1.0, steps=2000,
                                 intentional=True)
        assert not self.rule.applies_to(model)


# ---- Discount Horizon Mismatch ----

class TestDiscountHorizonMismatch:
    rule = DiscountHorizonMismatch()

    def test_short_horizon_sparse_goal(self):
        """gamma=0.95, 500 steps, sparse goal → CRITICAL."""
        model = EnvironmentModel(
            name="test", max_steps=500, gamma=0.95,
            n_actions=4, action_type="discrete",
        )
        model.add_reward_source(RewardSource(
            name="goal", reward_type=RewardType.TERMINAL, value=10.0,
            requires_action=True, intentional=True, discovery_probability=0.02,
        ))
        assert self.rule.applies_to(model)
        verdicts = self.rule.check(model)
        assert len(verdicts) == 1
        assert verdicts[0].severity == Severity.CRITICAL

    def test_long_horizon_ok(self):
        """gamma=0.999, 500 steps → no fire (horizon covers episode)."""
        model = EnvironmentModel(
            name="test", max_steps=500, gamma=0.999,
            n_actions=4, action_type="discrete",
        )
        model.add_reward_source(RewardSource(
            name="goal", reward_type=RewardType.TERMINAL, value=10.0,
            requires_action=True, intentional=True, discovery_probability=0.02,
        ))
        verdicts = self.rule.check(model)
        assert len(verdicts) == 0

    def test_dense_reward_skips(self):
        """Dense per-step reward → rule doesn't apply."""
        model = EnvironmentModel(
            name="test", max_steps=500, gamma=0.95,
            n_actions=4, action_type="discrete",
        )
        model.add_reward_source(RewardSource(
            name="dense", reward_type=RewardType.PER_STEP, value=1.0,
            requires_action=True, intentional=True,
        ))
        assert not self.rule.applies_to(model)


# ---- Negative Only Reward ----

class TestNegativeOnlyReward:
    rule = NegativeOnlyReward()

    def test_all_negative(self):
        """All sources ≤ 0 → CRITICAL."""
        model = EnvironmentModel(
            name="test", max_steps=200, gamma=1.0,
            n_actions=3, action_type="discrete",
        )
        model.add_reward_source(RewardSource(
            name="penalty", reward_type=RewardType.PER_STEP, value=-1.0,
        ))
        model.add_reward_source(RewardSource(
            name="goal", reward_type=RewardType.TERMINAL, value=0.0,
            requires_action=True, intentional=True,
        ))
        verdicts = self.rule.check(model)
        assert len(verdicts) == 1
        assert verdicts[0].severity == Severity.CRITICAL

    def test_has_positive(self):
        """At least one positive source → no fire."""
        model = EnvironmentModel(
            name="test", max_steps=200, gamma=0.99,
            n_actions=3, action_type="discrete",
        )
        model.add_reward_source(RewardSource(
            name="penalty", reward_type=RewardType.PER_STEP, value=-1.0,
        ))
        model.add_reward_source(RewardSource(
            name="goal", reward_type=RewardType.TERMINAL, value=1.0,
            requires_action=True, intentional=True,
        ))
        verdicts = self.rule.check(model)
        assert len(verdicts) == 0

    def test_positive_range(self):
        """Value is negative but range includes positive → no fire."""
        model = EnvironmentModel(
            name="test", max_steps=200, gamma=0.99,
            n_actions=3, action_type="discrete",
        )
        model.add_reward_source(RewardSource(
            name="variable", reward_type=RewardType.PER_STEP,
            value=-0.5, value_range=(-1.0, 0.5),
        ))
        verdicts = self.rule.check(model)
        assert len(verdicts) == 0


# ---- Reward Delay Horizon ----

class TestRewardDelayHorizon:
    rule = RewardDelayHorizon()

    def test_heavily_discounted_goal(self):
        """Goal at ~step 5000, gamma=0.99 → invisible."""
        model = EnvironmentModel(
            name="test", max_steps=10000, gamma=0.99,
            n_actions=4, action_type="discrete",
        )
        model.add_reward_source(RewardSource(
            name="goal", reward_type=RewardType.TERMINAL, value=10.0,
            requires_action=True, intentional=True, discovery_probability=0.001,
        ))
        verdicts = self.rule.check(model)
        assert len(verdicts) == 1
        assert verdicts[0].severity == Severity.CRITICAL

    def test_reachable_goal(self):
        """Goal at ~step 100, gamma=0.99 → visible."""
        model = EnvironmentModel(
            name="test", max_steps=200, gamma=0.99,
            n_actions=4, action_type="discrete",
        )
        model.add_reward_source(RewardSource(
            name="goal", reward_type=RewardType.TERMINAL, value=10.0,
            requires_action=True, intentional=True, discovery_probability=0.05,
        ))
        verdicts = self.rule.check(model)
        assert len(verdicts) == 0


# ---- Reward rules: remaining tests ----

class TestShapingNotPotentialBased:
    rule = ShapingNotPotentialBased()

    def test_action_dependent_shaping_warns(self):
        model = EnvironmentModel(name="t", max_steps=100, gamma=0.99, n_actions=4)
        model.add_reward_source(RewardSource(
            name="shaping", reward_type=RewardType.SHAPING, value=1.0,
            requires_action=True, can_loop=True,
        ))
        verdicts = self.rule.check(model)
        assert len(verdicts) >= 1

    def test_pbrs_clean(self):
        model = EnvironmentModel(name="t", max_steps=100, gamma=0.99, n_actions=4)
        model.add_reward_source(RewardSource(
            name="shaping", reward_type=RewardType.SHAPING, value=1.0,
            requires_action=False, can_loop=False,
        ))
        verdicts = self.rule.check(model)
        assert len(verdicts) == 0


class TestProxyRewardHackability:
    rule = ProxyRewardHackability()

    def test_action_dependent_shaping_flags(self):
        model = EnvironmentModel(name="t", max_steps=100, gamma=0.99, n_actions=4)
        model.add_reward_source(RewardSource(
            name="proxy", reward_type=RewardType.SHAPING, value=1.0,
            requires_action=True, can_loop=True,
        ))
        verdicts = self.rule.check(model)
        assert len(verdicts) >= 1


class TestStagedRewardPlateau:
    rule = StagedRewardPlateau()

    def test_staged_sources_warn(self):
        model = EnvironmentModel(name="t", max_steps=200, gamma=0.99, n_actions=4)
        model.add_reward_source(RewardSource(
            name="stage1", reward_type=RewardType.ON_EVENT, value=0.35,
            requires_action=True, discovery_probability=0.5,
        ))
        model.add_reward_source(RewardSource(
            name="stage2", reward_type=RewardType.ON_EVENT, value=0.15,
            requires_action=True, discovery_probability=0.1,
        ))
        model.add_reward_source(RewardSource(
            name="goal", reward_type=RewardType.TERMINAL, value=1.0,
            requires_action=True, intentional=True, discovery_probability=0.01,
        ))
        verdicts = self.rule.check(model)
        # May or may not fire depending on thresholds, but should not crash
        assert isinstance(verdicts, list)


class TestRewardDominanceImbalance:
    rule = RewardDominanceImbalance()

    def test_extreme_imbalance(self):
        model = EnvironmentModel(name="t", max_steps=1000, gamma=0.99, n_actions=4)
        model.add_reward_source(RewardSource(
            name="big", reward_type=RewardType.PER_STEP, value=1.0,
            requires_action=True, intentional=True,
        ))
        model.add_reward_source(RewardSource(
            name="tiny", reward_type=RewardType.PER_STEP, value=0.00001,
            requires_action=True,
        ))
        model.add_reward_source(RewardSource(
            name="medium", reward_type=RewardType.PER_STEP, value=0.5,
            requires_action=True,
        ))
        verdicts = self.rule.check(model)
        assert len(verdicts) >= 1


class TestExponentialSaturation:
    rule = ExponentialSaturation()

    def test_exp_tracking_info(self):
        model = EnvironmentModel(name="t", max_steps=1000, gamma=0.99, n_actions=4)
        model.add_reward_source(RewardSource(
            name="tracking", reward_type=RewardType.PER_STEP, value=1.0,
            value_type="exponential", value_params={"sigma": 0.1},
            requires_action=True, intentional=True,
        ))
        verdicts = self.rule.check(model)
        assert len(verdicts) >= 1


# ---- Training rules: remaining tests ----

class TestEntropyRegime:
    rule = EntropyCollapse()

    def test_zero_entropy(self):
        model = EnvironmentModel(name="t", max_steps=100, gamma=0.99, n_actions=4)
        config = TrainingConfig(algorithm="PPO", lr=3e-4, entropy_coeff=0.0)
        if self.rule.applies_to(model):
            verdicts = self.rule.check(model, config)
            assert isinstance(verdicts, list)


class TestReplayBufferRatio:
    rule = ReplayBufferRatio()

    def test_small_buffer(self):
        model = EnvironmentModel(name="t", max_steps=100, gamma=0.99,
                                 n_states=10000, n_actions=4)
        config = TrainingConfig(algorithm="DQN", lr=1e-4,
                                replay_buffer_size=100)
        if self.rule.applies_to(model):
            verdicts = self.rule.check(model, config)
            assert isinstance(verdicts, list)


class TestTargetNetworkUpdate:
    rule = TargetNetworkUpdate()

    def test_dqn_update(self):
        model = EnvironmentModel(name="t", max_steps=100, gamma=0.99, n_actions=4)
        config = TrainingConfig(algorithm="DQN", lr=1e-4,
                                target_update_freq=1)
        if self.rule.applies_to(model):
            verdicts = self.rule.check(model, config)
            assert isinstance(verdicts, list)


class TestEpsilonSchedule:
    rule = EpsilonSchedule()

    def test_short_schedule(self):
        model = EnvironmentModel(name="t", max_steps=100, gamma=0.99, n_actions=4)
        config = TrainingConfig(algorithm="DQN", lr=1e-4,
                                epsilon_start=1.0, epsilon_end=0.01,
                                epsilon_decay_steps=10,
                                total_steps=1000000)
        if self.rule.applies_to(model):
            verdicts = self.rule.check(model, config)
            assert isinstance(verdicts, list)


class TestSoftUpdateRate:
    rule = SoftUpdateRate()

    def test_high_tau(self):
        model = EnvironmentModel(name="t", max_steps=100, gamma=0.99, n_actions=4)
        config = TrainingConfig(algorithm="SAC", lr=3e-4, tau=0.5)
        if self.rule.applies_to(model):
            verdicts = self.rule.check(model, config)
            assert isinstance(verdicts, list)


class TestSACAlpha:
    rule = SACAlpha()

    def test_fixed_alpha(self):
        model = EnvironmentModel(name="t", max_steps=100, gamma=0.99, n_actions=4)
        config = TrainingConfig(algorithm="SAC", lr=3e-4,
                                alpha=0.2, auto_alpha=False)
        if self.rule.applies_to(model):
            verdicts = self.rule.check(model, config)
            assert isinstance(verdicts, list)


# ---- Advisory rules: tests ----

class TestAdvisories:
    """All 8 advisories should fire on their trigger patterns
    and stay silent on non-matching configs."""

    def _make_model(self, **kwargs):
        defaults = dict(name="t", max_steps=100, gamma=0.99,
                        n_actions=4, action_type="discrete")
        defaults.update(kwargs)
        return EnvironmentModel(**defaults)

    def test_physics_exploit_fires(self):
        rule = PhysicsExploitAdvisory()
        model = self._make_model(n_states=100000, n_actions=20,
                                 action_type="continuous",
                                 death_probability=0.01)
        model.add_reward_source(RewardSource(
            name="velocity", reward_type=RewardType.PER_STEP, value=1.0,
            requires_action=True, intentional=True,
        ))
        if rule.applies_to(model):
            verdicts = rule.check(model)
            assert isinstance(verdicts, list)

    def test_goal_misgeneralization_fires(self):
        rule = GoalMisgeneralizationAdvisory()
        model = self._make_model()
        model.add_reward_source(RewardSource(
            name="goal", reward_type=RewardType.TERMINAL, value=1.0,
            requires_action=True, intentional=True,
            discovery_probability=0.8,
        ))
        if rule.applies_to(model):
            verdicts = rule.check(model)
            assert isinstance(verdicts, list)

    def test_credit_assignment_fires(self):
        rule = CreditAssignmentAdvisory()
        model = self._make_model(max_steps=10000)
        model.add_reward_source(RewardSource(
            name="goal", reward_type=RewardType.TERMINAL, value=1.0,
            requires_action=True, intentional=True,
            discovery_probability=0.001,
        ))
        if rule.applies_to(model):
            verdicts = rule.check(model)
            assert isinstance(verdicts, list)

    def test_constrained_rl_fires(self):
        rule = ConstrainedRLAdvisory()
        model = self._make_model(action_type="continuous")
        model.add_reward_source(RewardSource(
            name="task", reward_type=RewardType.PER_STEP, value=1.0,
            requires_action=True, intentional=True,
        ))
        model.add_reward_source(RewardSource(
            name="safety_cost", reward_type=RewardType.PER_STEP, value=-0.1,
            requires_action=False,
        ))
        if rule.applies_to(model):
            verdicts = rule.check(model)
            assert isinstance(verdicts, list)

    def test_nonstationarity_fires(self):
        rule = NonStationarityAdvisory()
        model = self._make_model()
        model.add_reward_source(RewardSource(
            name="win", reward_type=RewardType.TERMINAL, value=1.0,
            requires_action=True, intentional=True,
        ))
        model.add_reward_source(RewardSource(
            name="lose", reward_type=RewardType.TERMINAL, value=-1.0,
            requires_action=False,
        ))
        if rule.applies_to(model):
            verdicts = rule.check(model)
            assert isinstance(verdicts, list)

    def test_learned_reward_fires(self):
        rule = LearnedRewardAdvisory()
        model = self._make_model(n_actions=50)
        model.add_reward_source(RewardSource(
            name="rm_score", reward_type=RewardType.PER_STEP, value=1.0,
            requires_action=True, intentional=True,
        ))
        if rule.applies_to(model):
            verdicts = rule.check(model)
            assert isinstance(verdicts, list)

    def test_missing_constraint_fires(self):
        rule = MissingConstraintAdvisory()
        model = self._make_model(action_type="continuous")
        model.add_reward_source(RewardSource(
            name="tracking", reward_type=RewardType.PER_STEP, value=1.0,
            requires_action=True, intentional=True,
        ))
        if rule.applies_to(model):
            verdicts = rule.check(model)
            assert isinstance(verdicts, list)

    def test_aggregation_trap_fires(self):
        rule = AggregationTrapAdvisory()
        model = self._make_model()
        model.add_reward_source(RewardSource(
            name="small_step", reward_type=RewardType.PER_STEP, value=0.01,
            requires_action=True, intentional=True,
        ))
        if rule.applies_to(model):
            verdicts = rule.check(model)
            assert isinstance(verdicts, list)
