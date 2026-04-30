"""Eureka Shadow Hand Scissors — GPT-4 generated reward.

Unbounded exponential opened_reward incentivizes ripping scissors
apart; passive reaching penalty cannot counterbalance.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "eureka_shadow_hand_scissors",
    "source_paper": (
        'Ma et al. 2024, "Eureka: Human-Level Reward Design via Coding'
        ' Large Language Models," ICLR 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2310.12931",
    "source_code_url": (
        "https://eureka-research.github.io/assets/reward_functions/"
        "shadow_hand_scissors.txt"
    ),
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "year": 2024,
    "domain": "manipulation",
    "brief_summary": (
        "GPT-4 generated Shadow Hand Scissors reward."
        " Unbounded exp(5*(dist-0.3)) incentivizes ripping"
        " scissors apart far beyond target opening."
    ),
    "documented_failure": (
        "opened_reward uses exp(5*(handle_dist-0.3)), which"
        " grows exponentially as handles separate beyond 0.3."
        " Unbounded: agent is incentivized to rip scissors"
        " apart maximally, not open to functional angle."
        " Linear reaching penalty (-0.2) cannot counterbalance."
    ),
    "failure_mechanism": "unbounded_reward",
    "discovery_stage": "during_training",
    "fix_known": (
        "Cap opened_reward at target distance or use sigmoid"
        " to saturate at desired opening angle."
    ),
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "opened_reward": (
            "Active (requires_action=True) and intentional but"
            " unbounded: exp(5*(dist-0.3)) grows without limit"
            " as handles separate."
        ),
        "handle_reaching_penalty": (
            "Passive (requires_action=False). Hands start near"
            " handles, so penalty is small at idle."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Scissors — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Shadow Hand Scissors.")
    print("Unbounded exponential opened_reward and passive reaching.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Scissors",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="opened_reward",
        reward_type=RewardType.PER_STEP,
        value=2.0,  # exp(5*(dist-0.3)), unbounded growth
        requires_action=True,
        intentional=True,
        can_loop=False,
    ))
    model.add_reward_source(RewardSource(
        name="handle_reaching_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.2,  # -0.5*(right_dist + left_dist)
        value_range=(-1.0, 0.0),  # idle: hands near handles = low penalty
        requires_action=False,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("The opened_reward uses exp(5*(handle_dist-0.3)), which grows")
    print("exponentially as handles separate beyond 0.3. This is")
    print("unbounded: the agent is incentivized to rip the scissors")
    print("apart as far as possible, not just open them to a functional")
    print("angle. The reaching penalty is linear and small (-0.2) so")
    print("it cannot counterbalance the exponential opened_reward.")


if __name__ == "__main__":
    run_example()
