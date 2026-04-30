"""Krakovna entry: Bicycle riding in circles.

Agent orbits the goal harvesting distance-decrease shaping reward
instead of reaching the destination.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "krakovna_bicycle",
    "source_paper": (
        'Randlov & Alstrom 1998, "Learning to Drive a Bicycle'
        ' using Reinforcement Learning and Shaping," ICML 1998'
    ),
    "year": 1998,
    "domain": "control",
    "encoding_basis": "primary_source",
    "brief_summary": (
        "Distance-decrease shaping without regression penalty."
        " Agent orbits the goal, harvesting shaping reward on"
        " each approach without ever arriving."
    ),
    "documented_failure": (
        "Agent rides in tight circles instead of reaching the"
        " goal. Orbiting repeatedly triggers the distance-"
        "decrease shaping reward."
    ),
    "failure_mechanism": "shaping_loop",
    "detection_type": "structural",
    "is_negative_example": False,
    "compute_cost_class": "low",
    "paper_url": None,
    "reward_location": "Randlov & Alstrom 1998 Section 4: r=-1 (fall), r=0.01 (goal), r=(4-psi_g^2)*0.00004 (shaping). 9 actions, gamma=0.99.",
    "verification_date": "2026-04-30",
    "discovery_stage": "during_training",
    "source_code_url": None,
    "fix_known": None,
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Bicycle riding in circles (Randlov & Alstrom 1998)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Bicycle (Randlov & Alstrom 1998)",
        max_steps=1000,
        gamma=0.99,
        n_actions=4,
        action_type="continuous",
    )

    # Sparse terminal goal: reach the destination
    model.add_reward_source(RewardSource(
        name="reach_goal",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        discovery_probability=0.05,
        intentional=True,
    ))

    # Distance-decrease shaping -- the trap
    # Only rewards decrease, no penalty for increase, so orbiting works.
    model.add_reward_source(RewardSource(
        name="distance_decrease",
        reward_type=RewardType.SHAPING,
        value=0.1,
        can_loop=True,
        loop_period=4,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
