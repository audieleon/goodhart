"""Example: Block Stacking — semantic proxy mismatch.

Robot flips the red block upside-down to maximize bottom-face height
instead of stacking it on the blue block as intended.

Source: Popov et al. 2017 (DeepMind)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "block_stacking",
    "source_paper": "Popov et al. 2017, 'Data-efficient Deep RL for Dexterous Manipulation' (DeepMind)",
    "paper_url": "https://arxiv.org/abs/1704.03073",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2017,
    "domain": "manipulation",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to stack blocks. Instead it flips the red block upside-down to maximize bottom-face height proxy.",
    "documented_failure": "Reward measures height(bottom_face(red_block)) as a proxy for stacking. The robot flips the red block upside-down, achieving maximum bottom-face height without stacking. The proxy doesn't capture the intended semantics.",
    "failure_mechanism": None,
    "detection_type": "specification",
    "discovery_stage": "during_training",
    "fix_known": "Use a reward that directly measures block-on-block contact or relative position",
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "shaping_action_dependent": "Non-potential-based shaping is structurally exploitable",
        "semantic_gap": "Bottom-face height proxy does not mean 'stacked on blue block'",
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Block Stacking — semantic proxy mismatch")
    print("=" * 70)
    print()
    print("Source: Popov et al. 2017 (DeepMind)")
    print("Reward: height(bottom_face(red_block))")
    print("Exploit: robot flips block upside-down (max bottom face height)")
    print()

    model = EnvironmentModel(
        name="Block Stacking (height proxy)",
        max_steps=200,
        gamma=0.99,
        n_states=50000,
        n_actions=6,
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="bottom_face_height",
        reward_type=RewardType.SHAPING,
        value=1.0,
        state_dependent=True,
        requires_action=True,
        can_loop=False,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="grasp_shaping",
        reward_type=RewardType.SHAPING,
        value=0.5,
        requires_action=True,
        can_loop=False,
    ))

    config = TrainingConfig(
        lr=3e-4,
        num_envs=16,
        n_actors=16,
        total_steps=5_000_000,
    )
    result = engine.print_report(model, config)

    print()
    print("The tool catches that shaping is action-dependent (not")
    print("potential-based per Ng 1999), which IS a real warning.")
    print("But the deeper problem is semantic: 'bottom face height'")
    print("doesn't MEAN 'stacked on top of blue block.'")
    print()
    print("Structural analysis catches proxy STRUCTURE problems.")
    print("It cannot catch proxy MEANING problems — those require")
    print("understanding what the reward function measures in the")
    print("physical world, not just its mathematical properties.")


if __name__ == "__main__":
    run_example()
