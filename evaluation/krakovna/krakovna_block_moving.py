"""Krakovna entry: Block moving -- agent moves table instead of block."""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "krakovna_block_moving",
    "source_paper": "Chopra 2018 (informal, possibly Reddit or blog post)",
    "paper_url": None,
    "year": 2018,
    "domain": "manipulation",
    "encoding_basis": "catalog_derived",
    "verification_date": "2026-04-29",
    "brief_summary": (
        "Agent was supposed to move a block to a target. Instead it"
        " moved the table the block was sitting on, which reduced"
        " block-to-target distance without grasping the block."
    ),
    "documented_failure": (
        "Agent moves the table rather than the block itself."
        " Distance-to-target proxy is satisfied by moving the"
        " surface the block rests on."
    ),
    "failure_mechanism": "proxy_reward",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "source_quality": (
            "Informal source (Chopra 2018). No peer-reviewed paper"
            " found. Encoding derived from Krakovna catalog description."
        ),
        "block_distance_to_target": (
            "Shaping toward target position. This is the proxy that"
            " the agent exploits by moving the table instead."
        ),
        "table_push_shortcut": (
            "High discovery probability (0.8) because pushing the"
            " table is mechanically easier than grasping the block."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Block moving (Chopra 2018)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Block moving (Chopra 2018)",
        max_steps=500,
        gamma=0.99,
        n_actions=6,
        action_type="continuous",
    )

    # True task: place block at target location
    model.add_reward_source(RewardSource(
        name="block_at_target",
        reward_type=RewardType.TERMINAL,
        value=10.0,
        discovery_probability=0.02,
        intentional=True,
    ))

    # Shaping: distance of block center to target -- the trap
    # Agent can reduce this by moving the table (which moves the block
    # indirectly), a mechanically easier strategy than grasping.
    model.add_reward_source(RewardSource(
        name="block_distance_to_target",
        reward_type=RewardType.SHAPING,
        value=1.0,
        can_loop=False,
        scales_with="distance",
        state_dependent=True,
        value_type="inverse",
        value_params={"eps": 0.01},
    ))

    # The key issue: table movement also decreases block distance
    # and is much easier to discover than grasping
    model.add_reward_source(RewardSource(
        name="table_push_shortcut",
        reward_type=RewardType.SHAPING,
        value=1.0,
        can_loop=False,
        requires_exploration=False,
        discovery_probability=0.8,
        scales_with="distance",
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
