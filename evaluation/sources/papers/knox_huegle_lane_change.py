"""Huegle et al. 2019 dynamic driving — lane change penalty (Knox 2023).

Reward: +1/step alive bonus, velocity tracking, -0.01 lane change
penalty. Knox et al. identified the lane change penalty as reward
shaping that could DISCOURAGE necessary lane changes (e.g., moving
away from a pedestrian). Continuing task with no time limit.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "knox_huegle_driving",
    "source_paper": (
        'Knox, Allievi, Banzhaf, Schmitt & Stone, "Reward (Mis)design '
        'for Autonomous Driving," Artificial Intelligence 2023. '
        'Original reward from Huegle, Kalweit, Mirchevska, Werling & '
        'Boedecker, "Dynamic Input for Deep RL in AD," IROS 2019'
    ),
    "paper_url": "https://arxiv.org/abs/2104.13906",
    "source_code_url": None,
    "reward_location": (
        "Appendix A.4: 3-attribute unweighted sum. (1) +1 given "
        "every step (alive bonus). (2) -|v - v_desired|/v_desired "
        "(velocity tracking). (3) -0.01 if lane change action. "
        "gamma=0.99, dt=2s, continuing task (no time limit or "
        "termination criterion)."
    ),
    "year": 2023,
    "domain": "driving",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "AD agent with alive bonus +1/step, velocity tracking, and "
        "tiny lane-change penalty. Knox showed the lane-change "
        "penalty is counterproductive shaping: it discourages moving "
        "away from pedestrians or obstacles in adjacent lanes."
    ),
    "documented_failure": (
        "Knox Section 4.1: lane change penalty (-0.01) is reward "
        "shaping that discourages lane changes, which may be "
        "necessary for safety. Section 4.2: this is one of 7 "
        "reward functions that prefer crash over idle."
    ),
    "failure_mechanism": "ontological_proxy",
    "detection_type": "structural",
    "discovery_stage": "post_training",
    "fix_known": (
        "Remove lane-change penalty or make it a soft constraint. "
        "Knox argues shaping should be designed separately from the "
        "true reward function."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "alive_bonus_exploit": (
            "+1/step alive bonus in a continuing task with no "
            "termination creates idle_exploit risk — staying alive "
            "doing nothing earns +1/step with no collision risk."
        ),
        "velocity_tracking": (
            "-|v - v_desired|/v_desired is a tracking reward. At "
            "v=0 (idle), penalty is -1.0. So idle: +1-1=0/step. "
            "But any collision risk makes driving strictly worse."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Huegle et al. 2019: Dynamic input AD
    # gamma=0.99, dt=2s, continuing (no time limit)
    model = EnvironmentModel(
        name="Huegle 2019 Dynamic Driving (Knox 2023)",
        max_steps=5000,
        gamma=0.99,
        n_states=100000,
        n_actions=5,
        action_type="discrete",
        death_probability=0.0,
    )

    # Alive bonus: +1 per step
    model.add_reward_source(RewardSource(
        name="alive_bonus",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        state_dependent=False,
        requires_action=False,
        intentional=True,
    ))

    # Velocity tracking: -|v - v_desired|/v_desired
    # At desired speed: 0. At idle: -1.0.
    model.add_reward_source(RewardSource(
        name="velocity_tracking",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Lane change penalty: -0.01
    model.add_reward_source(RewardSource(
        name="lane_change_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.01,
        state_dependent=False,
        requires_action=True,
        intentional=False,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
