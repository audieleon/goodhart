"""CARLA driving (Dosovitskiy 2017) — near-zero collision penalty (Knox 2023).

Collision weight is -0.00002, meaning a collision that changes the
damage variable by 1.0 costs less than 1/50000th of a meter of
progress. Knox et al. identified this as an extreme calibration error.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "knox_carla_collision",
    "source_paper": (
        'Knox, Allievi, Banzhaf, Schmitt & Stone, "Reward (Mis)design '
        'for Autonomous Driving," Artificial Intelligence 2023. '
        'Original reward from Dosovitskiy, Ros, Codevilla, Lopez & '
        'Koltun, "CARLA: An Open Urban Driving Simulator," CoRL 2017'
    ),
    "paper_url": "https://arxiv.org/abs/2104.13906",
    "source_code_url": None,
    "reward_location": (
        "Appendix A.3: r = (1)Δd + (0.05)Δv + (-0.00002)Δc + "
        "(-2)Δs + (-2)Δo. Δd=distance progress (m), Δv=speed change "
        "(km/h), Δc=collision damage [0,1], Δs=sidewalk overlap, "
        "Δo=opposite lane overlap. gamma=0.99, dt=0.1s."
    ),
    "year": 2023,
    "domain": "driving",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "CARLA benchmark reward with collision weight -0.00002 — "
        "a full collision costs less than 0.00002 units, while 1m "
        "of progress earns 1.0. The agent has essentially zero "
        "incentive to avoid collisions."
    ),
    "documented_failure": (
        "Knox et al. Section 4.2-4.3: collision weight -0.00002 is "
        "so small that a full collision (Δc=1) costs 50000x less than "
        "1 meter of progress. Knox's indifference point analysis shows "
        "this function would approve driving with near-continuous "
        "collisions. Also uses unsafe reward shaping (Section 4.1)."
    ),
    "failure_mechanism": "ontological_proxy",
    "detection_type": "structural",
    "discovery_stage": "post_training",
    "fix_known": (
        "Increase collision weight by 4-5 orders of magnitude. Knox "
        "Section 4.3: calibrate via indifference points against "
        "human risk tolerance (e.g., US 50-60 year olds: ~1M km per "
        "collision)."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "reward_dominance": (
            "Progress reward (1.0 per meter) dominates collision "
            "penalty (-0.00002). Ratio is 50000:1. "
            "reward_dominance_imbalance should fire."
        ),
        "shaping": (
            "Δv (speed change) and Δs/Δo (overlap changes) are "
            "reward shaping — they encourage behaviors correlated "
            "with good driving but are not the true objective. "
            "shaping_not_potential_based should fire."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Dosovitskiy et al. 2017: CARLA benchmark
    # dt=0.1s, gamma=0.99, episodic
    model = EnvironmentModel(
        name="CARLA Driving — Near-Zero Collision (Knox 2023)",
        max_steps=1000,
        gamma=0.99,
        n_states=100000,
        n_actions=2,
        action_type="continuous",
        death_probability=0.0,
    )

    # Distance progress: +1.0 per meter
    model.add_reward_source(RewardSource(
        name="distance_progress",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Speed change: +0.05 * Δv
    model.add_reward_source(RewardSource(
        name="speed_shaping",
        reward_type=RewardType.SHAPING,
        value=0.05,
        state_dependent=True,
        requires_action=True,
        intentional=False,
    ))

    # Collision damage: -0.00002 * Δc
    model.add_reward_source(RewardSource(
        name="collision_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.00002,
        state_dependent=True,
        requires_action=False,
        intentional=True,
    ))

    # Sidewalk overlap: -2.0 * Δs
    model.add_reward_source(RewardSource(
        name="sidewalk_penalty",
        reward_type=RewardType.PER_STEP,
        value=-2.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Opposite lane overlap: -2.0 * Δo
    model.add_reward_source(RewardSource(
        name="opposite_lane_penalty",
        reward_type=RewardType.PER_STEP,
        value=-2.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
