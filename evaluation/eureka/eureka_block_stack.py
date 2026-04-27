"""Example: Eureka Block Stack — GPT-4 generated block stacking reward.

Eureka uses GPT-4 to write reward functions for Isaac Gym tasks.
The Block Stack reward passes with no criticals: a sparse terminal
goal anchors the shaping, and the shaping term cannot loop. However,
warnings about proxy hackability of shaping are expected — the
height and alignment proxies could be gamed without actually stacking.

Source: Ma et al. 2024 (ICLR), Eureka project — GPT-4 generated reward
Tool should catch: no critical issues (PASS), possible warnings
  about proxy hackability of shaping terms
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Block Stack — GPT-4 generated block stacking reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Block Stack task.")
    print("Sparse goal with shaping — warnings but no criticals.")
    print()

    model = EnvironmentModel(
        name="Eureka Block Stack",
        max_steps=500,
        gamma=0.99,
        n_actions=7,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="stack_success",
        reward_type=RewardType.TERMINAL,
        value=10.0,
        requires_action=True,
        intentional=True,
        discovery_probability=0.1,
    ))
    model.add_reward_source(RewardSource(
        name="height_shaping",
        reward_type=RewardType.SHAPING,
        value=0.5,
        requires_action=True,
        can_loop=False,
    ))
    model.add_reward_source(RewardSource(
        name="alignment_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.3,
        requires_action=True,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("This reward passes because the terminal goal (stack_success)")
    print("anchors the shaping, and height_shaping cannot loop. However,")
    print("the height and alignment proxies are hackable: the agent could")
    print("learn to lift the block high (maximizing height_shaping) and")
    print("hover it in alignment (maximizing alignment_bonus) without")
    print("actually completing the stack. The 10.0 terminal reward should")
    print("eventually overcome this, but proxy gaming may slow learning.")


if __name__ == "__main__":
    run_example()
