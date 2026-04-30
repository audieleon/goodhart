"""Krakovna entry: Long legs -- agent grows legs and falls forward.

Distance reward with morphology optimization; agent grows long legs
and topples forward instead of walking.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "krakovna_long_legs",
    "source_paper": (
        'Ha 2018, "Reinforcement Learning for Improving'
        ' Agent Design"'
    ),
    "paper_url": "https://arxiv.org/abs/1804.03720",
    "year": 2018,
    "domain": "locomotion",
    "encoding_basis": "primary_source",
    "brief_summary": (
        "Distance-to-goal reward with morphology optimization."
        " Growing long legs and falling forward covers more"
        " distance than learning a stable gait."
    ),
    "documented_failure": (
        "Agent grows extremely long legs and falls toward the"
        " goal rather than walking."
    ),
    "failure_mechanism": "proxy_reward",
    "detection_type": "structural",
    "is_negative_example": False,
    "compute_cost_class": "low",
}


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
