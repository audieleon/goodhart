"""DribbleBot quadruped soccer dribbling (Ji et al. 2023, ICRA).

17-component reward: 8 task terms (ball tracking, alignment) ×
exp(9 regularization penalties). Sim-to-real on diverse terrain.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "dribblebot",
    "source_paper": (
        'Ji, Margolis & Agrawal, "DribbleBot: Dynamic Legged '
        'Manipulation in the Wild," ICRA 2023'
    ),
    "paper_url": "https://arxiv.org/abs/2304.01159",
    "source_code_url": "https://gmargoll.github.io/dribblebot",
    "reward_location": (
        "Appendix VII, Table III: 17 reward terms. "
        "8 positive task rewards (ball velocity/distance/alignment, "
        "gait scheduling) with weights 0.5-4.0. "
        "9 negative regularization penalties (joint limits, torque, "
        "collision, gravity, smoothing) with weights -0.0001 to -10.0. "
        "Total: r_t = r_pos * exp(r_neg). PPO, 50Hz, 40s episodes."
    ),
    "year": 2023,
    "domain": "locomotion",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Quadruped robot dribbles soccer ball on diverse terrain "
        "(sand, gravel, mud, snow). 17-component reward balances "
        "ball tracking with locomotion safety. Multiplicative: "
        "positive task × exp(negative penalties). Sim-to-real."
    ),
    "documented_failure": "None — successful sim-to-real deployment.",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "post_deployment",
    "fix_known": None,
    "compute_cost_class": "high",
    "is_negative_example": True,
    "encoding_rationale": {
        "multiplicative_structure": (
            "r = r_pos * exp(r_neg). Positive and negative groups "
            "combined multiplicatively. The exp ensures penalties "
            "scale the task reward down, not add independently."
        ),
        "high_component_count": (
            "17 terms spanning 5 orders of magnitude (0.5 to -10.0). "
            "Simplified to 3 representative sources for encoding: "
            "task tracking, gait regularization, safety penalties."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Table III: 17 terms, simplified to representative sources
    # r = r_pos * exp(r_neg), Unitree Go1, PPO

    model = EnvironmentModel(
        name="DribbleBot (Ji et al. 2023)",
        max_steps=2000,  # 40s at 50Hz
        gamma=0.99,
        n_actions=12,  # 12 joint targets
        action_type="continuous",
        death_probability=0.01,
    )

    # Task rewards: ball velocity tracking + alignment (weights 0.5-4.0)
    model.add_reward_source(RewardSource(
        name="ball_tracking",
        reward_type=RewardType.PER_STEP,
        value=4.0,  # dominant task reward (ball distance, alignment)
        requires_action=True,
        intentional=True,
        state_dependent=True,
        scales_with="velocity",
    ))

    # Gait regularization: swing/stance phase schedule
    model.add_reward_source(RewardSource(
        name="gait_schedule",
        reward_type=RewardType.PER_STEP,
        value=4.0,  # gait phase tracking
        requires_action=True,
        state_dependent=True,
    ))

    # Safety/regularization penalties (multiplicative via exp)
    # Joint limits (-10), collision (-5), gravity (-5),
    # torque/vel/accel (-0.0001 to -2.5e-7), smoothing (-0.1)
    model.add_reward_source(RewardSource(
        name="safety_penalties",
        reward_type=RewardType.PER_STEP,
        value=-0.5,  # aggregate of 9 small penalties
        requires_action=True,
        state_dependent=True,
        modifies="ball_tracking",
        modifier_type="multiplicative",
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
