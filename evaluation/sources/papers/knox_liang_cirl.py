"""CIRL vision-based driving (Liang et al. 2018) — heavy penalties (Knox 2023).

Large collision penalties (-100 vehicle/ped, -50 other), sidewalk
(-100), opposite lane (-100). Knox showed this still fails the
preference ordering despite having the strongest collision penalties
in the survey, because per-step speed reward dominates over time.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "knox_liang_cirl",
    "source_paper": (
        'Knox, Allievi, Banzhaf, Schmitt & Stone, "Reward (Mis)design '
        'for Autonomous Driving," Artificial Intelligence 2023. '
        'Original reward from Liang, Wang, Yang, Xing, "CIRL: '
        'Controllable Imitative RL for Vision-based Self-driving," '
        'ECCV 2018'
    ),
    "paper_url": "https://arxiv.org/abs/2104.13906",
    "source_code_url": None,
    "reward_location": (
        "Appendix A.7: 5-attribute unweighted sum. (1) steering "
        "angle penalty (incorrect for current command). (2) speed "
        "(km/h) with turn/speed-limit penalties. (3) -100 collision "
        "with vehicles/pedestrians, -50 other objects. (4) -100 "
        "sidewalk overlap. (5) -100 opposite lane overlap. "
        "gamma=0.9, dt=100ms, CARLA."
    ),
    "year": 2023,
    "domain": "driving",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Vision-based AD with the heaviest collision penalties in "
        "Knox's survey (-100), yet still uses unsafe reward shaping "
        "(steering angle, lane penalties) and low discount factor "
        "(gamma=0.9) that creates extreme temporal myopia."
    ),
    "documented_failure": (
        "Knox Section 4.1: all 5 attributes are shaping (none measure "
        "true outcomes). Knox Section 4.2: despite -100 collision, "
        "the speed reward accumulated over the episode can still make "
        "crash preferable to idle. gamma=0.9 exacerbates this by "
        "heavily discounting future penalties."
    ),
    "failure_mechanism": "ontological_proxy",
    "detection_type": "structural",
    "discovery_stage": "post_training",
    "fix_known": (
        "Knox: separate true reward from shaping. Use potential-based "
        "shaping for steering and lane-keeping. Increase gamma or use "
        "undiscounted evaluation."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "low_gamma": (
            "gamma=0.9 means the agent only cares about ~10 steps "
            "ahead. A collision 20 steps away is discounted by 0.9^20 "
            "= 0.12. discount_horizon_mismatch should fire."
        ),
        "all_shaping": (
            "Knox identified that ALL 5 reward attributes are shaping "
            "(heuristics for good driving), not measurements of the "
            "true objective. shaping_not_potential_based should fire."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Liang et al. 2018: CIRL vision-based driving
    # gamma=0.9, dt=100ms, CARLA
    model = EnvironmentModel(
        name="CIRL Vision Driving (Liang 2018, via Knox 2023)",
        max_steps=1000,
        gamma=0.9,
        n_states=100000,
        n_actions=2,
        action_type="continuous",
        death_probability=0.0,
    )

    # Speed reward: speed in km/h with limits
    # Typical driving: ~30 km/h → value ~30
    model.add_reward_source(RewardSource(
        name="speed_reward",
        reward_type=RewardType.PER_STEP,
        value=30.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Steering penalty
    model.add_reward_source(RewardSource(
        name="steering_penalty",
        reward_type=RewardType.PER_STEP,
        value=-5.0,
        state_dependent=False,
        requires_action=True,
        intentional=False,
    ))

    # Collision: -100 (vehicle/pedestrian)
    model.add_reward_source(RewardSource(
        name="collision_penalty",
        reward_type=RewardType.TERMINAL,
        value=-100.0,
        state_dependent=True,
        requires_action=False,
        intentional=True,
    ))

    # Sidewalk overlap: -100
    model.add_reward_source(RewardSource(
        name="sidewalk_penalty",
        reward_type=RewardType.PER_STEP,
        value=-100.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Opposite lane overlap: -100
    model.add_reward_source(RewardSource(
        name="opposite_lane_penalty",
        reward_type=RewardType.PER_STEP,
        value=-100.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
