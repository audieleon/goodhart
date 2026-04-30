"""Eureka Block Stack — GPT-4 generated block stacking reward.

Terminal goal anchors shaping; warnings about proxy hackability expected.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "eureka_block_stack",
    "source_paper": (
        'Ma et al. 2024, "Eureka: Human-Level Reward Design via Coding'
        ' Large Language Models," ICLR 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2310.12931",
    "source_code_url": (
        "https://eureka-research.github.io/assets/reward_functions/"
        "block_stack.txt"
    ),
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "domain": "manipulation",
    "brief_summary": (
        "GPT-4 generated Block Stack reward. Passes clean —"
        " terminal goal anchors shaping, warnings only."
    ),
    "documented_failure": "None — reward is well-designed",
    "failure_mechanism": None,
    "discovery_stage": "post_training",
    "fix_known": None,
    "compute_cost_class": "low",
    "is_negative_example": True,
    "encoding_rationale": {
        "stack_success": (
            "Terminal goal (10.0) anchors all shaping. Sparse"
            " (discovery_probability=0.1) but large enough to"
            " dominate once discovered."
        ),
        "height_shaping": (
            "Active shaping (can_loop=False) that guides toward"
            " goal. Proxy is hackable (lift without stacking)"
            " but cannot cycle."
        ),
        "alignment_bonus": (
            "Active (requires_action=True) but unintentional —"
            " alignment is a proxy that could be gamed without"
            " completing the stack."
        ),
    },
}


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
