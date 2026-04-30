"""Krakovna entry: Pinball table nudging.

Agent nudges the table to infinitely trigger a high-scoring switch
instead of playing with the flippers.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "krakovna_pinball_nudging",
    "source_paper": (
        'Lapuschkin et al 2019, "Unmasking Clever Hans'
        ' Predictors and Assessing What Machines Really'
        ' Learn," Nature Communications'
    ),
    "paper_url": "https://arxiv.org/abs/1902.10178",
    "year": 2019,
    "domain": "game_ai",
    "encoding_basis": "primary_source",
    "brief_summary": (
        "High-scoring switch can be triggered repeatedly via"
        " nudging. No cooldown or diminishing returns, so"
        " looping it dominates normal flipper play."
    ),
    "documented_failure": (
        "Agent nudges the pinball table so the ball infinitely"
        " triggers a high-scoring switch."
    ),
    "failure_mechanism": "respawning_loop",
    "detection_type": "structural",
    "is_negative_example": False,
    "compute_cost_class": "low",
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Pinball nudging (Lapuschkin et al 2019)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Pinball Nudging (Lapuschkin et al 2019)",
        max_steps=5000,
        gamma=0.99,
        n_actions=4,
        action_type="discrete",
        death_probability=0.01,
    )

    # High-scoring switch — can be triggered repeatedly via nudging.
    # This is the source the agent farms.
    model.add_reward_source(RewardSource(
        name="high_value_switch",
        reward_type=RewardType.ON_EVENT,
        value=50.0,
        respawn=RespawnBehavior.INFINITE,
        max_occurrences=0,
        requires_action=True,
        can_loop=True,
        loop_period=5,
    ))

    # Normal flipper-based scoring (bumpers, ramps, etc.)
    model.add_reward_source(RewardSource(
        name="flipper_score",
        reward_type=RewardType.ON_EVENT,
        value=10.0,
        respawn=RespawnBehavior.INFINITE,
        max_occurrences=0,
        requires_action=True,
    ))

    # Ball loss (drain)
    model.add_reward_source(RewardSource(
        name="ball_loss",
        reward_type=RewardType.TERMINAL,
        value=-50.0,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("The high_value_switch is farmable via nudging — the agent found")
    print("a repeatable loop that scores more than playing with flippers.")
    print("This is a structural exploit: the high-value event has no")
    print("cooldown or diminishing returns, so looping it dominates.")


if __name__ == "__main__":
    run_example()
