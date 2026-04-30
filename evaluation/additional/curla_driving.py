"""CuRLA curriculum driving (Uppuluri et al. 2025, CARLA).

Multiplicative reward r = angle * centering * speed + collision.
Curriculum progressively adds traffic and collision penalty.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "curla_driving",
    "source_paper": (
        'Uppuluri, Patel, Mehta, Kamath & Chakraborty, "CuRLA: '
        'Curriculum Learning Based Deep Reinforcement Learning for '
        'Autonomous Driving," arXiv:2501.04982, 2025'
    ),
    "paper_url": "https://arxiv.org/abs/2501.04982",
    "source_code_url": None,
    "reward_location": (
        "Section 3.2A: Eq. 3-4: r=r_a*r_d*r_v (original), "
        "r'=r_a*r_d*r_v'+r_c (CuRLA). Eq. 5: angle max(1-|a|/20°,0). "
        "Eq. 6: centering (1-d/3m). Eq. 7-8: speed (original flat vs "
        "CuRLA increasing). r_c=collision penalty. Table 2: gamma=0.99, "
        "lr=1e-4, epochs=3, clip=0.2, horizon=128."
    ),
    "year": 2025,
    "domain": "driving",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Curriculum-based autonomous driving in CARLA. Multiplicative "
        "reward r = angle * centering * speed ensures all three must "
        "be good simultaneously. Curriculum adds traffic density and "
        "collision penalty progressively."
    ),
    "documented_failure": (
        "Original SCA baseline has misleading flat speed reward "
        "(Eq. 7): constant 1.0 in [v_min, v_target] confuses agent "
        "about what speed to target. CuRLA fixes with increasing "
        "function (Eq. 8) that clearly rewards higher speed."
    ),
    "failure_mechanism": "proxy_reward",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Replace flat speed reward with monotonically increasing "
        "function (Eq. 8). Add collision penalty r_c. Use curriculum "
        "to progressively introduce traffic."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": True,
    "encoding_rationale": {
        "multiplicative_reward": (
            "r = r_a * r_d * r_v is multiplicative — all factors "
            "in [0,1]. Same pattern as CaRL. Penalties are modifiers "
            "not additive. Encoding the CuRLA (fixed) version."
        ),
        "collision_additive": (
            "r_c (collision penalty) is additive on top of the "
            "multiplicative base. Only component that can make "
            "reward negative."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # CuRLA reward: r' = r_a * r_d * r_v' + r_c
    # All factors in [0,1], multiplicative base
    # CARLA Town 7 highway, PPO

    model = EnvironmentModel(
        name="CuRLA Driving (Uppuluri et al. 2025)",
        max_steps=1000,  # 3 laps or termination
        gamma=0.99,
        n_actions=3,  # acceleration, steering, speed
        action_type="continuous",
        death_probability=0.01,
    )

    # Multiplicative base: angle * centering * speed
    # All in [0,1], product is the reward
    model.add_reward_source(RewardSource(
        name="angle_centering_speed",
        reward_type=RewardType.PER_STEP,
        value=1.0,  # max when all factors = 1
        requires_action=True,
        intentional=True,
        state_dependent=True,
    ))

    # Collision penalty (additive, introduced by curriculum)
    model.add_reward_source(RewardSource(
        name="collision_penalty",
        reward_type=RewardType.ON_EVENT,
        value=-1.0,
        requires_action=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
