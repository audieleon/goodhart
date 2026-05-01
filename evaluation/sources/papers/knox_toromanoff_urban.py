"""Toromanoff et al. 2020 urban driving — weighted composite (Knox 2023).

r = r_speed + 0.5*r_dist + 0.5*r_heading, with -1 on termination.
Knox noted this is one of only 2 papers that acknowledged reward
shaping and its limitations. The reward function was designed
through trial-and-error process.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "knox_toromanoff_urban",
    "source_paper": (
        'Knox, Allievi, Banzhaf, Schmitt & Stone, "Reward (Mis)design '
        'for Autonomous Driving," Artificial Intelligence 2023. '
        'Original reward from Toromanoff, Wirbel & Moutarde, '
        '"End-to-End Model-Free RL for Urban Driving Using Implicit '
        'Affordances," CVPR 2020'
    ),
    "paper_url": "https://arxiv.org/abs/2104.13906",
    "source_code_url": None,
    "reward_location": (
        "Appendix A.10: r = r_speed + 0.5*r_dist + 0.5*r_heading + "
        "(-1)*termination. r_speed: 1-|s_desired - s_ego|/40 in [0,1]. "
        "r_dist: -d_path/d_max in [-1,0], d_max=2.0m. r_heading: "
        "clip(-1,0, b*|theta_ego - theta_path|) in [-1,0]. -1 on "
        "termination. gamma=0.99, dt=100ms, CARLA."
    ),
    "year": 2023,
    "domain": "driving",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Urban driving with weighted reward: speed tracking, path "
        "distance, heading alignment, termination penalty. One of "
        "only 2 papers that acknowledged their use of reward shaping. "
        "Trial-and-error designed."
    ),
    "documented_failure": (
        "Knox Section 4.1: acknowledged reward shaping but did not "
        "discuss safety guarantees. Section 4.5: trial-and-error "
        "design with unexamined consequences. No training terminated "
        "successfully during training — all episodes ended via "
        "undesirable termination conditions."
    ),
    "failure_mechanism": "ontological_proxy",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Knox: separate true reward (safe arrival) from shaping "
        "(speed, heading). Use potential-based shaping with safety "
        "guarantees."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "no_positive_goal": (
            "No positive terminal reward for reaching destination. "
            "The only terminal signal is -1 penalty. Combined with "
            "per-step positive reward (speed), agent has no incentive "
            "to actually reach the goal."
        ),
        "trial_and_error": (
            "Knox Section 4.5: designed through trial-and-error, "
            "which means the reward was tuned to produce 'good enough' "
            "behavior on test runs, potentially overfitting."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    model = EnvironmentModel(
        name="Toromanoff 2020 Urban Driving (Knox 2023)",
        max_steps=1000,
        gamma=0.99,
        n_states=100000,
        n_actions=9,
        action_type="discrete",
        death_probability=0.0,
    )

    # r_speed: 1 - |s_desired - s_ego|/40, in [0, 1]
    model.add_reward_source(RewardSource(
        name="speed_tracking",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # 0.5 * r_dist: -d_path/d_max in [-1, 0]
    model.add_reward_source(RewardSource(
        name="path_distance",
        reward_type=RewardType.PER_STEP,
        value=-0.5,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # 0.5 * r_heading: clip(-1, 0, ...) in [-1, 0]
    model.add_reward_source(RewardSource(
        name="heading_alignment",
        reward_type=RewardType.PER_STEP,
        value=-0.5,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Termination: -1
    model.add_reward_source(RewardSource(
        name="termination_penalty",
        reward_type=RewardType.TERMINAL,
        value=-1.0,
        state_dependent=False,
        requires_action=False,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
