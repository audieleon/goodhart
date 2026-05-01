"""Chen et al. 2019 urban driving — crash preferred over idle (Knox 2023).

Knox et al. (2023) showed this reward function fails the preference
ordering sanity check: with collision penalty of only -10 and time
penalty of -0.1/step, staying idle for 500 steps costs -50, but
crashing costs only ~-10. The reward function prefers crashing.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "knox_chen_driving",
    "source_paper": (
        'Knox, Allievi, Banzhaf, Schmitt & Stone, "Reward (Mis)design '
        'for Autonomous Driving," Artificial Intelligence 2023. '
        'Original reward from Chen, Yuan & Tomizuka, "Model-free Deep '
        'RL for Urban AD," ITSC 2019'
    ),
    "paper_url": "https://arxiv.org/abs/2104.13906",
    "source_code_url": None,
    "reward_location": (
        "Appendix A.2: 5-attribute unweighted sum. (1) min(speed, "
        "10-speed) speed reward. (2) -0.5*(steering_angle)^2 steering "
        "penalty. (3) -10 collision. (4) -1 lane departure. (5) -0.1 "
        "per step time penalty. gamma=0.99, T=500 (50s at 400ms steps). "
        "Section 4.2 + Figure 3: fails preference ordering — prefers "
        "crash trajectory over idle trajectory."
    ),
    "year": 2023,
    "domain": "driving",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "AD agent rewarded with speed + small collision penalty (-10). "
        "Knox et al. showed this prefers a crash trajectory over "
        "staying idle, because idle accumulates -0.1/step × 500 = -50 "
        "while crashing costs only -10."
    ),
    "documented_failure": (
        "Knox et al. Section 4.2: Of 9 reward functions evaluated, 7 "
        "preferred crash over idle. This reward has collision penalty "
        "-10, but time penalty -0.1/step accumulates to -50 over 500 "
        "steps, making crash a 'cheaper' outcome than inaction."
    ),
    "failure_mechanism": "ontological_proxy",
    "detection_type": "structural",
    "discovery_stage": "post_training",
    "fix_known": (
        "Knox et al. sanity check 2: compare G(tau_crash) vs "
        "G(tau_idle). If crash > idle, increase collision penalty or "
        "remove per-step time penalty. Use indifference point analysis "
        "(Section 4.3) to calibrate risk tolerance."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "speed_reward": (
            "min(speed, 10-speed) peaks at 5 m/s. Per-step positive "
            "reward that incentivizes driving. Estimated value ~5.0 "
            "at optimal speed."
        ),
        "collision_penalty": (
            "Only -10 on collision event. Knox showed this is far "
            "too small relative to accumulated time penalties."
        ),
        "time_penalty": (
            "-0.1 per step for 500 steps = -50 total. This exceeds "
            "the collision penalty, creating the crash > idle ordering."
        ),
        "preference_ordering_failure": (
            "G(tau_idle) = -0.1 * 500 = -50. "
            "G(tau_crash) = speed_reward * steps_before_crash - 10. "
            "Even crashing immediately: G = -10 > -50. The reward "
            "prefers any crash over staying idle."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Chen et al. 2019: Urban AD in CARLA
    # gamma=0.99, T=500 (50s at 400ms frame skip), 1 continuous action
    model = EnvironmentModel(
        name="Chen 2019 Urban AD — Crash > Idle (Knox 2023)",
        max_steps=500,
        gamma=0.99,
        n_states=100000,
        n_actions=2,
        action_type="continuous",
        death_probability=0.0,
    )

    # Speed reward: min(speed, 10-speed), peaks at ~5.0
    model.add_reward_source(RewardSource(
        name="speed_reward",
        reward_type=RewardType.PER_STEP,
        value=5.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Steering penalty: -0.5 * steering_angle^2
    model.add_reward_source(RewardSource(
        name="steering_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.5,
        state_dependent=False,
        requires_action=True,
        intentional=True,
    ))

    # Collision penalty: -10 (ONE-TIME on collision)
    model.add_reward_source(RewardSource(
        name="collision_penalty",
        reward_type=RewardType.TERMINAL,
        value=-10.0,
        state_dependent=True,
        requires_action=False,
        intentional=True,
    ))

    # Lane departure: -1 per step while out of lane
    model.add_reward_source(RewardSource(
        name="lane_departure",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
        state_dependent=True,
        requires_action=False,
        intentional=True,
    ))

    # Time penalty: -0.1 per step
    model.add_reward_source(RewardSource(
        name="time_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.1,
        state_dependent=False,
        requires_action=False,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
