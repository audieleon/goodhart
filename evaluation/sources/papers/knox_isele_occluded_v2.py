"""Isele et al. 2018 occluded variant — longer horizon (Knox 2023).

Same reward as knox_isele_intersection but with 60s occluded
episodes (300 steps). Tests whether correct preference ordering
holds with longer episodes. This is the occluded scenario variant.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "knox_liu_driving",
    "source_paper": (
        'Knox, Allievi, Banzhaf, Schmitt & Stone, "Reward (Mis)design '
        'for Autonomous Driving," Artificial Intelligence 2023. '
        'Original reward from Liu, Siravuru, Prabhakar, Veloso & '
        'Kantor, "Learning End-to-End Multimodal Sensor Policies for '
        'Autonomous Navigation," CoRL 2017'
    ),
    "paper_url": "https://arxiv.org/abs/2104.13906",
    "source_code_url": None,
    "reward_location": (
        "Appendix A.9 + references [29]: end-to-end multimodal "
        "sensor fusion for autonomous navigation. Reward includes "
        "speed tracking and obstacle avoidance terms. Knox analyzed "
        "as one of the 19 publications."
    ),
    "year": 2023,
    "domain": "navigation",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "End-to-end multimodal sensor driving with speed and obstacle "
        "avoidance rewards. Knox identified unsafe reward shaping in "
        "the obstacle distance term."
    ),
    "documented_failure": (
        "Knox Section 4.1: includes reward shaping for getting close "
        "to other vehicles (drifting sideways penalty). Knox Section "
        "B: reward shaping identified in the obstacle distance term."
    ),
    "failure_mechanism": "ontological_proxy",
    "detection_type": "structural",
    "discovery_stage": "post_training",
    "fix_known": (
        "Use potential-based shaping for obstacle avoidance. Separate "
        "true reward (safe arrival) from heuristic shaping."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "obstacle_shaping": (
            "Reward for obstacle distance is shaping — correlated "
            "with safety but not the true objective."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    model = EnvironmentModel(
        name="Liu 2017 Multimodal Navigation (Knox 2023)",
        max_steps=500,
        gamma=0.99,
        n_states=100000,
        n_actions=2,
        action_type="continuous",
        death_probability=0.0,
    )

    # Speed tracking reward
    model.add_reward_source(RewardSource(
        name="speed_reward",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Obstacle distance shaping
    model.add_reward_source(RewardSource(
        name="obstacle_distance",
        reward_type=RewardType.SHAPING,
        value=0.5,
        state_dependent=True,
        requires_action=False,
        intentional=False,
    ))

    # Collision penalty
    model.add_reward_source(RewardSource(
        name="collision_penalty",
        reward_type=RewardType.TERMINAL,
        value=-10.0,
        state_dependent=True,
        requires_action=False,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
