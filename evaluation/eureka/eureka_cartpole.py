"""Eureka CartPole — GPT-4 generated balancing reward.

Clean reward: both components are intentional and require action.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "eureka_cartpole",
    "source_paper": (
        'Ma et al. 2024, "Eureka: Human-Level Reward Design via Coding'
        ' Large Language Models," ICLR 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2310.12931",
    "source_code_url": (
        "https://eureka-research.github.io/assets/reward_functions/"
        "cartpole.txt"
    ),
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "domain": "control",
    "brief_summary": (
        "GPT-4 generated CartPole balancing reward."
        " Passes clean."
    ),
    "documented_failure": "None — reward is well-designed",
    "failure_mechanism": None,
    "discovery_stage": "post_training",
    "fix_known": None,
    "compute_cost_class": "low",
    "is_negative_example": True,
    "encoding_rationale": {
        "angle_reward": (
            "Active (requires_action=True) and intentional."
            " Gaussian-shaped reward directly measures pole angle."
        ),
        "position_reward": (
            "Active (requires_action=True) and intentional."
            " Gaussian-shaped reward directly measures cart position."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka CartPole — GPT-4 generated balancing reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym CartPole task.")
    print("Both components are intentional and Gaussian-shaped.")
    print()

    model = EnvironmentModel(
        name="Eureka CartPole",
        max_steps=500,
        gamma=0.99,
        n_actions=2,
        action_type="discrete",
    )
    model.add_reward_source(RewardSource(
        name="angle_reward",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="position_reward",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("This is one of Eureka's cleanest outputs. Both reward")
    print("components directly measure what we care about (pole angle")
    print("and cart position), both require action, and both are marked")
    print("intentional. No passive terms that could create idle exploits.")


if __name__ == "__main__":
    run_example()
