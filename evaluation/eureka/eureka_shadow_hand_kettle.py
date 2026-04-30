"""Eureka Shadow Hand Kettle — GPT-4 generated reward.

Passive handle-grasping rewards and a hardcoded no-spill constant
dominate the reward landscape.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "eureka_shadow_hand_kettle",
    "source_paper": (
        'Ma et al. 2024, "Eureka: Human-Level Reward Design via Coding'
        ' Large Language Models," ICLR 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2310.12931",
    "source_code_url": (
        "https://eureka-research.github.io/assets/reward_functions/"
        "shadow_hand_kettle.txt"
    ),
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "year": 2024,
    "domain": "manipulation",
    "brief_summary": (
        "GPT-4 generated Shadow Hand Kettle reward."
        " Passive handle-grasping rewards and a hardcoded"
        " no-spill constant dominate the reward landscape."
    ),
    "documented_failure": (
        "Kettle and bucket handle rewards are passive: hands"
        " start near handles, giving high exp(-0.5*dist) at"
        " idle. reward_no_spill uses hardcoded constant (0.5)"
        " instead of actual ball count, producing fixed"
        " log(2)=0.69 every step. Pure bias, no signal."
    ),
    "failure_mechanism": "idle_exploit",
    "discovery_stage": "during_training",
    "fix_known": (
        "Gate handle rewards on contact force. Replace"
        " hardcoded no_spill with actual ball count."
    ),
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "reward_kettle_handle": (
            "Passive (requires_action=False). Hand starts near"
            " handle, saturating exp(-0.5*dist) at rest."
        ),
        "reward_bucket_handle": (
            "Passive (requires_action=False). Hand starts near"
            " handle, saturating exp(-0.5*dist) at rest."
        ),
        "reward_spout_bucket": (
            "Active (requires_action=True) and intentional."
            " Measures spout-bucket alignment distance."
        ),
        "reward_spout_bucket_z": (
            "Active (requires_action=True) and intentional."
            " Measures spout-bucket vertical alignment."
        ),
        "reward_no_spill": (
            "Passive (requires_action=False). Hardcoded constant"
            " produces fixed 0.69 every step. No signal."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Shadow Hand Kettle — GPT-4 generated reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Shadow Hand Kettle.")
    print("Passive handle grasping rewards and hardcoded no_spill term.")
    print()

    model = EnvironmentModel(
        name="Eureka Shadow Hand Kettle",
        max_steps=1000,
        gamma=0.99,
        n_actions=40,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="reward_kettle_handle",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # active: hand moves away
        value_range=(0.0, 1.0),  # idle: hand starts near handle
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="reward_bucket_handle",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # active: hand moves away
        value_range=(0.0, 1.0),  # idle: hand starts near handle
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="reward_spout_bucket",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # exp(-0.3*dist_spout_bucket)
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="reward_spout_bucket_z",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # exp(-2.5*z_diff)
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="reward_no_spill",
        reward_type=RewardType.PER_STEP,
        value=0.69,  # log(1+exp(-5*(0.5-0.5))) = log(2), constant
        requires_action=False,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("The kettle and bucket handle rewards are passive: hands start")
    print("near the handles, giving high exp(-0.5*dist) at idle. The")
    print("reward_no_spill uses a hardcoded constant (0.5) instead of")
    print("actual ball count, producing a fixed log(2)=0.69 every step.")
    print("This is pure bias that adds no training signal. Together the")
    print("passive components dominate the reward landscape.")


if __name__ == "__main__":
    run_example()
