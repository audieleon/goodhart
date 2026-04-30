"""Krakovna entry: Lego stacking -- agent flips block instead of stacking.

Proxy measures bottom-face height; agent flips block upside-down
to satisfy the proxy without actual stacking.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "krakovna_lego_stacking",
    "source_paper": (
        'Popov et al 2017, "Data-efficient Deep Reinforcement'
        ' Learning for Dexterous Manipulation," DeepMind'
    ),
    "paper_url": "https://arxiv.org/abs/1704.03073",
    "year": 2017,
    "domain": "manipulation",
    "encoding_basis": "primary_source",
    "brief_summary": (
        "Stacking proxy measures bottom-face height. Flipping"
        " the block achieves the target height without stacking."
    ),
    "documented_failure": (
        "Agent flips the block upside-down instead of stacking"
        " it on top of another block."
    ),
    "failure_mechanism": "proxy_reward",
    "detection_type": "structural",
    "is_negative_example": False,
    "compute_cost_class": "low",
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Lego stacking (Popov et al 2017)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Lego stacking (Popov et al 2017)",
        max_steps=500,
        gamma=0.99,
        n_actions=6,
        action_type="continuous",
    )

    # True task: stack block on top of another
    model.add_reward_source(RewardSource(
        name="block_stacked",
        reward_type=RewardType.TERMINAL,
        value=10.0,
        discovery_probability=0.01,
        intentional=True,
    ))

    # Proxy: height of bottom face -- the trap
    # Flipping the block satisfies this without stacking.
    model.add_reward_source(RewardSource(
        name="bottom_face_height",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        scales_with="z_position",
        state_dependent=True,
        value_type="proportional",
        value_params={"scale": 1.0},
    ))

    # Grasp reward (shaping to encourage picking up block)
    model.add_reward_source(RewardSource(
        name="grasp_reward",
        reward_type=RewardType.ON_EVENT,
        value=0.5,
        max_occurrences=1,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
