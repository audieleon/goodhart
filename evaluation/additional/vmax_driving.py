"""V-Max autonomous driving framework (Charraut et al. 2025, RLC).

Hierarchical reward: safety → navigation → behavior. Documents
reward term conflicts: comfort vs progress tradeoff. SAC on Waymax.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "vmax_driving",
    "source_paper": (
        'Charraut, Doulazmi, Tournaire & Buhet, "V-Max: A Reinforcement '
        'Learning Framework for Autonomous Driving," '
        'Reinforcement Learning Journal 2025'
    ),
    "paper_url": "https://arxiv.org/abs/2503.08388",
    "source_code_url": "https://github.com/valeoai/v-max",
    "reward_location": (
        "Section 4 page 10: r_safety = -collision - offroad - redlight. "
        "r_navigation = r_safety - 0.2*offroute + 0.2*progress. "
        "r_behavior = r_navigation + 0.2*comfort - 0.1*speeding. "
        "Table 3: comparison of all 3 levels. Table 12 (appendix): "
        "grid search weights. SAC, LQ encoder, 10Hz, WOMD."
    ),
    "year": 2025,
    "domain": "driving",
    "encoding_basis": "primary_source",
    "verification_date": "2026-05-01",
    "brief_summary": (
        "Open RL framework for autonomous driving on Waymax. Hierarchical "
        "reward: safety penalties (collision/offroad/redlight) → navigation "
        "(progress shaping) → behavior (comfort + speed compliance). "
        "Documents reward term conflicts: comfort weight too high → overly "
        "cautious, progress weight too high → aggressive."
    ),
    "documented_failure": (
        "Safety-only reward (r_safety) produces conservative policy with "
        "low progress (78.66 vs 155.38 for navigation). Individual reward "
        "terms conflict: high comfort weights reduce speed, high progress "
        "weights increase collision risk. Multi-objective reward design "
        "requires careful weight tuning."
    ),
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Hierarchical reward building from safety → navigation → behavior. "
        "Grid search over weights (Table 12). Best config: navigation "
        "level achieves 97.45% accuracy with 155.38 progress."
    ),
    "compute_cost_class": "high",
    "is_negative_example": True,
    "encoding_rationale": {
        "hierarchical_levels": (
            "Each level builds on the previous. Safety is the base "
            "(binary penalties). Navigation adds progress shaping. "
            "Behavior adds comfort and speed compliance. Encoding "
            "the full behavior reward (most complete)."
        ),
        "indicator_rewards": (
            "All components are binary indicators (0 or 1). Weights "
            "are explicit: collision/offroad/redlight = -1.0 each, "
            "offroute = -0.2, progress = +0.2, comfort = +0.2, "
            "speeding = -0.1."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # r_behavior = -collision - offroad - redlight
    #              - 0.2*offroute + 0.2*progress
    #              + 0.2*comfort - 0.1*speeding

    model = EnvironmentModel(
        name="V-Max Driving (Charraut et al. 2025)",
        max_steps=80,  # 8 seconds at 10Hz
        gamma=0.99,
        n_actions=2,  # acceleration + steering
        action_type="continuous",
        death_probability=0.05,  # collision/offroad terminates
    )

    # Safety penalties (binary, terminal-like)
    model.add_reward_source(RewardSource(
        name="collision_penalty",
        reward_type=RewardType.ON_EVENT,
        value=-1.0,
        requires_action=True,
    ))

    model.add_reward_source(RewardSource(
        name="offroad_penalty",
        reward_type=RewardType.ON_EVENT,
        value=-1.0,
        requires_action=True,
    ))

    model.add_reward_source(RewardSource(
        name="redlight_penalty",
        reward_type=RewardType.ON_EVENT,
        value=-1.0,
        requires_action=True,
    ))

    # Navigation: progress shaping
    model.add_reward_source(RewardSource(
        name="progress_reward",
        reward_type=RewardType.PER_STEP,
        value=0.2,
        requires_action=True,
        intentional=True,
        state_dependent=True,
    ))

    model.add_reward_source(RewardSource(
        name="offroute_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.2,
        requires_action=True,
        state_dependent=True,
    ))

    # Behavior: comfort and speed
    model.add_reward_source(RewardSource(
        name="comfort_reward",
        reward_type=RewardType.PER_STEP,
        value=0.2,
        requires_action=True,
        state_dependent=True,
    ))

    model.add_reward_source(RewardSource(
        name="speeding_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.1,
        requires_action=True,
        state_dependent=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
