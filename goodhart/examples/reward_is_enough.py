"""Example: Reward-is-Enough hypothesis environments.

Silver et al. 2021 argue that reward maximization suffices for general
intelligence. We model three of their example domains to show how
goodhart analyzes environments where the reward IS well-designed:
perception (visual classification), language (dialogue), and
social intelligence (negotiation).

Source: Silver et al. 2021 ("Reward is Enough", Artificial Intelligence)
Tool should: pass clean (these are well-designed by construction,
  since the paper argues reward is sufficient)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


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
    model.add_reward_source(RewardSource(
        name="survival",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        respawn=RespawnBehavior.INFINITE,
        requires_action=False,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="food_gathering",
        reward_type=RewardType.ON_EVENT,
        value=5.0,
        respawn=RespawnBehavior.TIMED,
        respawn_time=100,
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="predator_avoidance",
        reward_type=RewardType.ON_EVENT,
        value=-10.0,
        requires_action=False,
    ))

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
