"""Krakovna entry: Sonic wall clipping for forward progress.

Reward structure is sound; agent clips through walls via a physics
bug to accumulate forward progress faster.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "krakovna_sonic",
    "source_paper": (
        'Nichol et al 2018, "Gotta Learn Fast: A New'
        ' Benchmark for Generalization in RL"'
    ),
    "paper_url": "https://arxiv.org/abs/1804.03720",
    "year": 2018,
    "domain": "game_ai",
    "encoding_basis": "catalog_derived",
    "brief_summary": (
        "Forward progress reward plus level completion bonus."
        " Reward is appropriate for a platformer; the exploit"
        " is a collision detection bug in the game engine."
    ),
    "documented_failure": (
        "Agent clips through walls to move rightward faster"
        " than intended platforming gameplay."
    ),
    "failure_mechanism": "physics_exploit",
    "detection_type": "dynamic",
    "is_negative_example": True,
    "compute_cost_class": "low",
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Sonic wall clipping (Hesse et al 2018)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Sonic (Hesse et al 2018)",
        max_steps=4500,
        gamma=0.99,
        n_actions=8,
        action_type="discrete",
        death_probability=0.02,
    )

    # Per-step forward progress — rightward position delta.
    model.add_reward_source(RewardSource(
        name="forward_progress",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        value_type="proportional",
        value_params={"scale": 1.0},
        scales_with="position",
        state_dependent=True,
        requires_action=True,
        intentional=True,
        max_occurrences=0,
    ))

    # Level completion bonus
    model.add_reward_source(RewardSource(
        name="level_completion",
        reward_type=RewardType.TERMINAL,
        value=100.0,
        intentional=True,
        discovery_probability=0.1,
    ))

    # Death penalty
    model.add_reward_source(RewardSource(
        name="death_penalty",
        reward_type=RewardType.TERMINAL,
        value=-50.0,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("Advisory: physics_exploit — the wall clipping is a collision")
    print("detection bug in the Sonic engine. The reward structure")
    print("(forward progress + level completion) is appropriate for a")
    print("platformer. The agent simply found a faster way to move right")
    print("by exploiting broken collision physics.")


if __name__ == "__main__":
    run_example()
