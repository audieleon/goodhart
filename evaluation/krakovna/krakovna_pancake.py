"""Krakovna entry: Pancake flipping -- agent throws pancake high."""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "krakovna_pancake",
    "source_paper": "Unity 2018 (ML-Agents demo)",
    "paper_url": None,
    "year": 2018,
    "domain": "manipulation",
    "encoding_basis": "catalog_derived",
    "verification_date": "2026-04-29",
    "brief_summary": (
        "Agent was supposed to flip a pancake and catch it. Instead"
        " it throws the pancake as high as possible to maximize"
        " airtime proxy."
    ),
    "documented_failure": (
        "Agent launches pancake vertically instead of flipping."
        " Airtime proxy is maximized by throwing, not by"
        " executing a controlled flip and catch."
    ),
    "failure_mechanism": "proxy_reward",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "source_quality": (
            "Informal source (Unity 2018, ML-Agents demo). Not a"
            " peer-reviewed paper. Encoding derived from Krakovna"
            " catalog description."
        ),
        "airtime": (
            "Per-step reward scaling with z-position. Throwing"
            " maximizes this without any catch behavior."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Pancake flip (Unity 2018)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Pancake flip (Unity 2018)",
        max_steps=500,
        gamma=0.99,
        n_actions=4,
        action_type="continuous",
    )

    # True task: flip pancake and catch it
    model.add_reward_source(RewardSource(
        name="successful_flip",
        reward_type=RewardType.TERMINAL,
        value=10.0,
        discovery_probability=0.01,
        intentional=True,
    ))

    # Time-away-from-surface proxy -- the trap
    # Throwing the pancake maximizes airtime without any catch.
    model.add_reward_source(RewardSource(
        name="airtime",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=True,
        scales_with="z_position",
        state_dependent=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
