"""Krakovna entry: Soccer -- agent vibrates touching ball.

Ball-contact shaping reward; agent vibrates in place maximizing
contact time instead of kicking toward goal.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "krakovna_soccer",
    "source_paper": (
        'Ng et al 1999, "Policy Invariance Under Reward'
        ' Transformations: Theory and Application to Reward'
        ' Shaping," ICML 1999'
    ),
    "year": 1999,
    "domain": "game_ai",
    "encoding_basis": "primary_source",
    "brief_summary": (
        "Ball-contact shaping reward without progress incentive."
        " Agent vibrates against the ball to maximize contact"
        " time instead of advancing toward the goal."
    ),
    "documented_failure": (
        "Agent vibrates in place while touching the ball"
        " rather than kicking it toward the goal."
    ),
    "failure_mechanism": "shaping_loop",
    "detection_type": "structural",
    "is_negative_example": False,
    "compute_cost_class": "low",
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Soccer ball-contact vibration (Ng et al 1999)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Soccer (Ng et al 1999)",
        max_steps=1000,
        gamma=0.99,
        n_actions=6,
        action_type="continuous",
    )

    # Scoring a goal
    model.add_reward_source(RewardSource(
        name="score_goal",
        reward_type=RewardType.TERMINAL,
        value=10.0,
        discovery_probability=0.02,
        intentional=True,
    ))

    # Ball contact shaping -- the trap
    # Agent receives reward every step it touches the ball.
    # Vibrating in place maximizes contact without scoring.
    model.add_reward_source(RewardSource(
        name="ball_contact",
        reward_type=RewardType.PER_STEP,
        value=0.1,
        requires_action=True,
        can_loop=True,
        loop_period=1,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
