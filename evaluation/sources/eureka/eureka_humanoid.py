"""Eureka Humanoid — GPT-4 generated locomotion reward.

Time penalty (~20/step) dwarfs alive bonus (2/step); dying is 9.6x better.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "eureka_humanoid",
    "source_paper": (
        'Ma et al. 2024, "Eureka: Human-Level Reward Design via Coding'
        ' Large Language Models," ICLR 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2310.12931",
    "source_code_url": (
        "https://eureka-research.github.io/assets/reward_functions/"
        "humanoid.txt"
    ),
    "reward_location": "GPT-4 generated reward code at https://eureka-research.github.io/assets/reward_functions/humanoid.txt",
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "year": 2024,
    "domain": "locomotion",
    "brief_summary": (
        "GPT-4 generated Humanoid locomotion reward."
        " exp(x/0.01) time penalty creates ~20.0/step cost,"
        " making death 9.6x better than surviving."
    ),
    "documented_failure": (
        "exp(x/0.01) time penalty produces ~20.0/step effective"
        " cost, dwarfing the alive bonus (2.0/step) and forward"
        " velocity (1.5/step). Dying ends the penalty stream,"
        " making suicide the optimal policy."
    ),
    "failure_mechanism": "death_beats_survival",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Scale the time penalty below the alive bonus, or remove"
        " it entirely — the alive bonus already incentivizes"
        " survival."
    ),
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "alive_bonus": (
            "Passive (requires_action=False). Intended to keep"
            " the agent alive, but overwhelmed by time_penalty."
        ),
        "time_penalty": (
            "Passive (requires_action=False). exp(x/0.01) creates"
            " ~20.0/step cost that dominates all positive terms."
            " Dying eliminates the penalty stream."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Humanoid — GPT-4 generated locomotion reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Humanoid task.")
    print("The exp(x/0.01) time penalty creates ~20.0/step effective cost.")
    print()

    model = EnvironmentModel(
        name="Eureka Humanoid",
        max_steps=1000,
        gamma=0.99,
        n_actions=21,
        action_type="continuous",
        death_probability=0.05,
    )
    model.add_reward_source(RewardSource(
        name="forward_velocity",
        reward_type=RewardType.PER_STEP,
        value=1.5,
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="alive_bonus",
        reward_type=RewardType.PER_STEP,
        value=2.0,
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="time_penalty",
        reward_type=RewardType.PER_STEP,
        value=-20.0,
        requires_action=False,
    ))

    engine.print_report(model)

    print()
    print("The exp(x/0.01) time penalty produces ~20.0/step cost,")
    print("dwarfing the alive bonus (2.0/step) and forward velocity")
    print("(1.5/step). Dying ends the episode and stops accumulating")
    print("penalty, making death 9.6x more rewarding than surviving.")
    print("This is a classic Goodhart failure: GPT-4 added the penalty")
    print("to encourage efficiency, but the scale makes suicide optimal.")


if __name__ == "__main__":
    run_example()
