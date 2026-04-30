"""Eureka Shadow Hand Switch — GPT-4 generated reward.

Single passive touch reward with no task-completion signal; the
switch toggle objective is entirely absent from the reward.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "eureka_shadow_hand_switch",
    "source_paper": (
        'Ma et al. 2024, "Eureka: Human-Level Reward Design via Coding'
        ' Large Language Models," ICLR 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2310.12931",
    "source_code_url": (
        "https://eureka-research.github.io/assets/reward_functions/"
        "shadow_hand_switch.txt"
    ),
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "year": 2024,
    "domain": "manipulation",
    "brief_summary": (
        "GPT-4 generated Shadow Hand Switch reward."
        " Single passive touch reward with no task-completion"
        " signal; switch toggle objective is absent."
    ),
    "documented_failure": (
        "touch_reward is the ONLY component. It measures"
        " fingertip proximity to switch handles using"
        " pow(sum(exp(-d)),2), entirely passive: fingertips"
        " start near handles. No component measures whether"
        " the switch was toggled. Task objective is completely"
        " absent from the reward function."
    ),
    "failure_mechanism": "idle_exploit",
    "discovery_stage": "during_training",
    "fix_known": (
        "Add a switch-state reward measuring whether the"
        " switch was actually toggled."
    ),
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "touch_reward": (
            "Passive (requires_action=False). Fingertips start"
            " near handles, saturating pow(sum(exp(-d)),2) at"
            " rest. No intentional component exists."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Switch — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Shadow Hand Switch.")
    print("Passive touch reward with no task completion signal.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Switch",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="touch_reward",
        reward_type=RewardType.PER_STEP,
        value=1.0,  # pow(sum(exp(-dist)),2) * pow(sum(exp(-dist)),2)
        value_range=(0.0, 25.0),  # idle: all fingertips near handle
        requires_action=False,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("The touch_reward is the ONLY reward component. It measures")
    print("fingertip proximity to switch handles using pow(sum(exp(-d)),2)")
    print("for each hand, then multiplies them. This is entirely passive:")
    print("fingertips start near handles, giving high reward at idle.")
    print("Critically, there is NO component measuring whether the switch")
    print("was toggled — the task objective is completely absent from the")
    print("reward function.")


if __name__ == "__main__":
    run_example()
