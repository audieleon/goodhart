"""Analysis rule library.

Three categories of rules:
- reward: rules analyzing MDP reward structure
- training: rules analyzing training hyperparameters
- architecture: rules analyzing model architecture (precedent-based)
"""

from goodhart.rules.reward import (
    PenaltyDominatesGoal,
    DeathBeatsSurvival,
    IdleExploit,
    ExplorationThreshold,
    RespawningExploit,
    DeathResetExploit,
    ShapingLoopExploit,
    ShapingNotPotentialBased,
    ProxyRewardHackability,
    IntrinsicSufficiency,
    BudgetSufficiency,
    CompoundTrap,
    StagedRewardPlateau,
    RewardDominanceImbalance,
    ExponentialSaturation,
    IntrinsicDominance,
    DiscountHorizonMismatch,
    NegativeOnlyReward,
    RewardDelayHorizon,
    REWARD_RULES,
)
from goodhart.rules.training import (
    LearningRateRegime,
    CriticLearningRate,
    EntropyCollapse,
    ClipFractionPrediction,
    ExpertCollapse,
    BatchSizeInteraction,
    ParallelismEffect,
    MemoryCapacity,
    ReplayBufferRatio,
    TargetNetworkUpdate,
    EpsilonSchedule,
    SoftUpdateRate,
    SACAlpha,
    TRAINING_RULES,
)
from goodhart.rules.architecture import (
    PrecedentRule,
    Precedent,
    EmbedDimCapacity,
    RoutingFloorNecessity,
    RecurrenceType,
    ActorCountEffect,
    ARCHITECTURE_RULES,
)

from goodhart.rules.advisories import (
    PhysicsExploitAdvisory,
    GoalMisgeneralizationAdvisory,
    CreditAssignmentAdvisory,
    ConstrainedRLAdvisory,
    NonStationarityAdvisory,
    LearnedRewardAdvisory,
    MissingConstraintAdvisory,
    AggregationTrapAdvisory,
    ADVISORY_RULES,
)

ALL_RULES = (list(REWARD_RULES) + list(TRAINING_RULES)
             + list(ARCHITECTURE_RULES) + list(ADVISORY_RULES))
RULE_COUNT = len(ALL_RULES)

__all__ = [
    # Reward rules
    "PenaltyDominatesGoal", "DeathBeatsSurvival", "IdleExploit",
    "ExplorationThreshold", "RespawningExploit", "DeathResetExploit",
    "ShapingLoopExploit", "ShapingNotPotentialBased", "ProxyRewardHackability",
    "IntrinsicSufficiency", "BudgetSufficiency", "CompoundTrap",
    "StagedRewardPlateau", "RewardDominanceImbalance", "ExponentialSaturation", "IntrinsicDominance",
    "DiscountHorizonMismatch", "NegativeOnlyReward", "RewardDelayHorizon",
    "REWARD_RULES",
    # Training rules
    "LearningRateRegime", "CriticLearningRate", "EntropyCollapse",
    "ClipFractionPrediction", "ExpertCollapse", "BatchSizeInteraction",
    "ParallelismEffect", "MemoryCapacity",
    "ReplayBufferRatio", "TargetNetworkUpdate", "EpsilonSchedule",
    "SoftUpdateRate", "SACAlpha", "TRAINING_RULES",
    # Architecture rules
    "PrecedentRule", "Precedent", "EmbedDimCapacity",
    "RoutingFloorNecessity", "RecurrenceType", "ActorCountEffect",
    "ARCHITECTURE_RULES",
    # Advisory rules
    "PhysicsExploitAdvisory", "GoalMisgeneralizationAdvisory",
    "CreditAssignmentAdvisory", "ConstrainedRLAdvisory",
    "NonStationarityAdvisory", "LearnedRewardAdvisory",
    "MissingConstraintAdvisory", "AggregationTrapAdvisory",
    "ADVISORY_RULES",
    # Aggregates
    "ALL_RULES", "RULE_COUNT",
]
