"""Example: Fetch Reach — sparse binary reward + HER.

Fetch Reach gives -1 per step until the gripper reaches the target,
then 0. With sparse reward, random exploration rarely succeeds.
Hindsight Experience Replay (HER) solves this by relabeling goals.

Source: Plappert et al. 2018 (Fetch environments), Andrychowicz et al.
2017 (HER, NeurIPS)
Tool should catch: exploration_threshold (very sparse reward signal)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Fetch Reach — sparse binary reward")
    print("=" * 70)
    print()
    print("Source: Plappert et al. 2018, Andrychowicz et al. 2017")
    print("Reward: -1 per step until ||gripper - target|| < epsilon,")
    print("then 0. Extremely sparse without HER.")
    print()

    model = EnvironmentModel(
        name="Fetch Reach (sparse)",
        max_steps=50,
        gamma=0.98,
        n_states=10000,
        n_actions=4,
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="step_penalty",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
    ))
    model.add_reward_source(RewardSource(
        name="goal_reached",
        reward_type=RewardType.TERMINAL,
        value=50.0,  # offsets all step penalties when reached
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.05,
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.01,
        num_envs=16,
        n_actors=16,
        total_steps=1_000_000,
    )
    engine.print_report(model, config)

    print()
    print("Without HER, the 5% discovery rate means the agent needs")
    print("~20 episodes to see a single success. With PPO's on-policy")
    print("updates, this is barely learnable. HER dramatically improves")
    print("data efficiency by relabeling failed trajectories as successes")
    print("for different goals — but that's an algorithm fix, not a")
    print("reward structure fix.")


if __name__ == "__main__":
    run_example()
