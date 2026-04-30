"""Krakovna entry: Cartwheel Ant -- MuJoCo Ant does cartwheels."""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "krakovna_cartwheel",
    "source_paper": "Jucys 2024 (informal, blog or video)",
    "paper_url": None,
    "year": 2024,
    "domain": "locomotion",
    "encoding_basis": "catalog_derived",
    "verification_date": "2026-04-29",
    "brief_summary": (
        "MuJoCo Ant was supposed to jump. Instead it cartwheels,"
        " achieving high torso z-coordinate via rotation rather"
        " than vertical leap."
    ),
    "documented_failure": (
        "Ant does cartwheels instead of jumping. Z-coordinate"
        " threshold is satisfied by rotating the body so limbs"
        " act as spokes, not by vertical displacement."
    ),
    "failure_mechanism": "proxy_reward",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "source_quality": (
            "Informal source (Jucys 2024, blog/video). No"
            " peer-reviewed paper. Encoding derived from Krakovna"
            " catalog description."
        ),
        "z_threshold_bonus": (
            "Per-step reward for torso z exceeding threshold."
            " Cartwheeling satisfies this more easily than jumping."
        ),
    },
    "paper_url": None,
    "reward_location": "Reward structure inferred from Krakovna catalog description",
    "verification_date": "2026-04-30",
    "discovery_stage": "during_training",
    "source_code_url": None,
    "fix_known": None,
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Cartwheel Ant (Jucys 2024)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Cartwheel Ant (Jucys 2024)",
        max_steps=1000,
        gamma=0.99,
        n_actions=8,
        action_type="continuous",
    )

    # The intended behavior: jump (vertical displacement)
    # Not directly rewarded -- only the z-threshold proxy exists.

    # Z-coordinate threshold bonus -- the trap
    # Rewards torso z > threshold, but cartwheeling achieves this
    # more easily than jumping.
    model.add_reward_source(RewardSource(
        name="z_threshold_bonus",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=True,
        can_loop=True,
        loop_period=10,
        scales_with="z_position",
        state_dependent=True,
    ))

    # Alive bonus (standard MuJoCo)
    model.add_reward_source(RewardSource(
        name="alive_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.05,
        requires_action=False,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
