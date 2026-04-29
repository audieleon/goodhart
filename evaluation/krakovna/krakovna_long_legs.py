"""Krakovna entry: Long legs -- agent grows legs and falls forward.

Source: Ha 2018
Failure: Agent grows extremely long legs and falls toward the goal
  rather than walking.
Mechanism: Distance-to-goal reward combined with body modification
  (morphology optimization). Growing long legs and toppling forward
  covers more distance than learning a stable gait. The proxy
  (distance traveled) does not penalize falling or unnatural
  morphology.
Domain: Evolutionary / morphology optimization
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Long legs (Ha 2018)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Long legs (Ha 2018)",
        max_steps=1000,
        gamma=0.99,
        n_actions=12,
        action_type="continuous",
    )

    # Forward distance reward -- the trap
    # Maximized by growing long legs and falling forward,
    # covering distance without a real gait.
    model.add_reward_source(RewardSource(
        name="forward_distance",
        reward_type=RewardType.SHAPING,
        value=1.0,
        scales_with="distance",
        state_dependent=True,
        can_loop=False,
        intentional=True,
    ))

    # No penalty for falling or unnatural morphology
    # This absence is the structural problem: the reward does not
    # constrain HOW distance is achieved.

    # Alive bonus (small, easily dominated)
    model.add_reward_source(RewardSource(
        name="alive_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.01,
        requires_action=False,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
