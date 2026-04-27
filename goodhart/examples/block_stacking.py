"""Example: Block Stacking — semantic proxy mismatch.

The robot should stack a red block on a blue block. The reward
measures height(bottom_face(red_block)). The robot flips the red
block upside-down — maximum bottom-face height without stacking.

The tool catches part of this: the shaping is action-dependent
(not potential-based) and therefore exploitable. But the real
exploit is SEMANTIC — the proxy (bottom face height) doesn't
mean what the designer intended (block placement).

Source: Popov et al. 2017 (DeepMind), "Data-efficient Deep RL
for Dexterous Manipulation"
Tool result: catches proxy warnings, but not the semantic gap
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


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
