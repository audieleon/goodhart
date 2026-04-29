"""Krakovna entry: Soccer -- agent vibrates touching ball.

Source: Ng et al 1999
Failure: Agent learns to vibrate in place while touching the ball,
  rather than kicking it toward the goal.
Mechanism: Ball-contact shaping reward. The agent receives reward for
  touching the ball, so it maximizes contact time by oscillating
  against the ball. This harvests dense shaping reward without
  advancing toward the actual goal (scoring).
Domain: Simulated sports / control

Note: Partially related to existing examples (Ng et al 1999 is cited
in multiple PBRS examples), but no direct soccer encoding exists in
goodhart/examples/.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


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
