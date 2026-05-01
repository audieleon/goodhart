"""Jaritz et al. 2018 race driving — 4 reward variants (Knox 2023).

End-to-end race driving with 4 reward functions, 3 with intentional
shaping. Jaritz acknowledged their shaping prevents optimal
trajectories: "bots do not achieve optimal trajectories... [because]
the car will always try to remain in the track center."
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "knox_jaritz_racing",
    "source_paper": (
        'Knox, Allievi, Banzhaf, Schmitt & Stone, "Reward (Mis)design '
        'for Autonomous Driving," Artificial Intelligence 2023. '
        'Original reward from Jaritz, De Charette, Toromanoff, Perot, '
        'Nashashibi, "End-to-End Race Driving with Deep RL," ICRA 2018'
    ),
    "paper_url": "https://arxiv.org/abs/2104.13906",
    "source_code_url": None,
    "reward_location": (
        "Appendix A.6: 4 reward variants. (1) r=v (speed only). "
        "(2) r=v(cos(alpha)-d) ('Ours'). (3) r=v(cos(alpha)-"
        "max(d-0.5w,0)) ('Ours-margin'). (4) r=v(cos(alpha)-"
        "sigmoid(|d|-0.5w)) ('Ours-sigmoid'). v=velocity along lane, "
        "alpha=heading-lane angle, d=lane center distance, w=road "
        "width. dt=33ms (30 FPS), gamma=0.99."
    ),
    "year": 2023,
    "domain": "driving",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Race driving with velocity × lane-centering reward. Jaritz "
        "acknowledged the shaping prevents optimal racing lines — "
        "the car always tries to stay centered rather than taking "
        "optimal apex trajectories."
    ),
    "documented_failure": (
        "Knox Section 4.1: Jaritz acknowledged their shaping creates "
        "suboptimal trajectories. The lane-centering term (d) prevents "
        "the car from using the full track width for optimal racing "
        "lines. One of only 2 papers that acknowledged shaping issues."
    ),
    "failure_mechanism": "ontological_proxy",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Remove lane-centering shaping for racing. Use lap time as "
        "true reward. Safe shaping should be potential-based."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "multiplicative_velocity": (
            "Reward is v * f(alignment, distance). This creates "
            "a multiplicative structure where higher speed amplifies "
            "all other reward components."
        ),
        "acknowledged_failure": (
            "Jaritz is one of only 2/19 papers in Knox's survey "
            "that acknowledged their reward shaping caused problems."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    model = EnvironmentModel(
        name="Jaritz 2018 Race Driving (Knox 2023)",
        max_steps=3000,
        gamma=0.99,
        n_states=100000,
        n_actions=2,
        action_type="continuous",
        death_probability=0.01,
    )

    # r = v * (cos(alpha) - d): speed × alignment reward
    # At good driving: v~20 m/s, cos(alpha)~1, d~0 → r~20
    model.add_reward_source(RewardSource(
        name="speed_alignment",
        reward_type=RewardType.PER_STEP,
        value=20.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Lane centering penalty (embedded in multiplicative term)
    # Separately encoded for analysis: -v*d, at d=0.5: -10
    model.add_reward_source(RewardSource(
        name="lane_centering_shaping",
        reward_type=RewardType.SHAPING,
        value=-10.0,
        state_dependent=True,
        requires_action=True,
        intentional=False,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
