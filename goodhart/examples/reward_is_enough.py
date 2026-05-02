"""Example: Reward-is-Enough hypothesis environments.

Well-designed reward examples from the thesis that reward maximization suffices for general intelligence.
Source: Silver et al. 2021 ("Reward is Enough", Artificial Intelligence)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "reward_is_enough",
    "source_paper": "Silver et al. 2021 ('Reward is Enough', Artificial Intelligence)",
    "paper_url": "https://doi.org/10.1016/j.artint.2021.103535",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2021,
    "domain": "control",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Theoretical/discussion piece arguing reward maximization suffices for intelligence. Well-designed rewards pass clean by construction.",
    "documented_failure": "N/A — these are well-designed by construction since the paper argues reward is sufficient",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "low",
    "is_negative_example": True,
    "encoding_rationale": {
        "theoretical_baseline": "Paper argues the right reward suffices; goodhart detects when the reward you wrote is not the right one",
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Reward-is-Enough — well-designed reward examples")
    print("=" * 70)
    print()
    print("Source: Silver et al. 2021 (Artificial Intelligence)")
    print("'All of intelligence can be understood as subserving the")
    print("maximisation of reward by an agent acting in its environment.'")
    print()

    # Simple survival environment (Section 3, Perception example)
    model = EnvironmentModel(
        name="Reward-is-Enough (survival perception)",
        max_steps=1000,
        gamma=0.99,
        n_states=100000,
        n_actions=10,
        death_probability=0.05,
    )
    model.add_reward_source(
        RewardSource(
            name="survival",
            reward_type=RewardType.PER_STEP,
            value=1.0,
            respawn=RespawnBehavior.INFINITE,
            requires_action=False,
            intentional=True,
        )
    )
    model.add_reward_source(
        RewardSource(
            name="food_gathering",
            reward_type=RewardType.ON_EVENT,
            value=5.0,
            respawn=RespawnBehavior.TIMED,
            respawn_time=100,
            requires_action=True,
            intentional=True,
        )
    )
    model.add_reward_source(
        RewardSource(
            name="predator_avoidance",
            reward_type=RewardType.ON_EVENT,
            value=-10.0,
            requires_action=False,
        )
    )

    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.01,
        num_envs=64,
        n_actors=64,
        total_steps=50_000_000,
    )
    result = engine.print_report(model, config)

    print()
    if result.passed:
        print("Clean pass — consistent with the paper's thesis that")
        print("well-designed reward is sufficient for intelligence.")
    print()
    print("The Reward-is-Enough thesis doesn't claim all rewards work.")
    print("It claims the *right* reward suffices. Goodhart's contribution")
    print("is detecting when the reward you wrote ISN'T the right one.")


if __name__ == "__main__":
    run_example()
