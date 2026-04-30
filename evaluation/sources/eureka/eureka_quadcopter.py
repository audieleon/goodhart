"""Eureka Quadcopter — GPT-4 generated flight reward.

Penalties total 12/step vs sparse goal of 15; idle exploit and blocked exploration.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "eureka_quadcopter",
    "source_paper": (
        'Ma et al. 2024, "Eureka: Human-Level Reward Design via Coding'
        ' Large Language Models," ICLR 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2310.12931",
    "source_code_url": (
        "https://eureka-research.github.io/assets/reward_functions/"
        "quadcopter.txt"
    ),
    "reward_location": "GPT-4 generated reward code at https://eureka-research.github.io/assets/reward_functions/quadcopter.txt",
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "year": 2024,
    "domain": "control",
    "brief_summary": (
        "GPT-4 generated Quadcopter flight reward. Penalties"
        " (12/step) dwarf sparse goal (15 total), creating idle"
        " exploit and blocking exploration."
    ),
    "documented_failure": (
        "Three simultaneous failures: (1) penalties total 12/step"
        " while goal is worth 15 and achievable ~1% of the time;"
        " (2) all penalties are passive, so the agent minimizes"
        " them by staying still; (3) any exploratory movement"
        " incurs immediate cost, preventing goal discovery."
    ),
    "failure_mechanism": "compound_trap",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Scale penalties well below the goal reward, make them"
        " active (requires_action=True), and add dense shaping"
        " toward the target."
    ),
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "distance_penalty": (
            "Passive (requires_action=False). -6.0/step for being"
            " away from target, but staying still avoids it."
        ),
        "linvel_penalty": (
            "Passive (requires_action=False). -4.0/step penalizes"
            " any linear velocity; stillness scores zero."
        ),
        "angvel_penalty": (
            "Passive (requires_action=False). -2.0/step penalizes"
            " any angular velocity; stillness scores zero."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Quadcopter — GPT-4 generated flight reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Quadcopter task.")
    print("Penalties total 12.0/step vs sparse goal of 15.0.")
    print()

    model = EnvironmentModel(
        name="Eureka Quadcopter",
        max_steps=500,
        gamma=0.99,
        n_actions=4,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="distance_to_target",
        reward_type=RewardType.TERMINAL,
        value=15.0,
        requires_action=True,
        intentional=True,
        discovery_probability=0.01,
    ))
    model.add_reward_source(RewardSource(
        name="distance_penalty",
        reward_type=RewardType.PER_STEP,
        value=-6.0,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="linvel_penalty",
        reward_type=RewardType.PER_STEP,
        value=-4.0,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="angvel_penalty",
        reward_type=RewardType.PER_STEP,
        value=-2.0,
        requires_action=False,
    ))

    engine.print_report(model)

    print()
    print("Three simultaneous failures: (1) penalties total 12.0/step")
    print("while the goal is worth 15.0 and achievable ~1% of the time,")
    print("so doing nothing avoids 6000 penalty over 500 steps;")
    print("(2) all penalties are passive (requires_action=False), so")
    print("the agent minimizes them by staying still; (3) any exploratory")
    print("movement incurs immediate cost, preventing discovery of the")
    print("goal. GPT-4 added the penalties to encourage smooth flight,")
    print("but their scale makes stillness the dominant strategy.")


if __name__ == "__main__":
    run_example()
