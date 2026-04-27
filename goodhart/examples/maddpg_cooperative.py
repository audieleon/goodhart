"""Example: MADDPG cooperative navigation (MPE).

Multi-agent particle environment: N agents must cover N landmarks.
Each agent is penalized by its distance to nearest landmark, creating
cooperative reward. With shared reward, credit assignment is ambiguous;
with individual reward, agents may compete for the same landmark.

Source: Lowe et al. 2017 (NeurIPS, MADDPG), Mordatch & Abbeel 2018
Tool should catch: potential idle exploit (distance penalty means
  standing near any landmark gives 0 penalty, same as optimal)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("MADDPG Cooperative Navigation (MPE)")
    print("=" * 70)
    print()
    print("Source: Lowe et al. 2017 (NeurIPS)")
    print("N agents cover N landmarks. Reward = -sum(min_distance).")
    print("With shared reward, credit assignment is ambiguous.")
    print()

    # Model from the perspective of a single agent
    model = EnvironmentModel(
        name="MPE Cooperative Navigation",
        max_steps=25,
        gamma=0.95,
        n_states=1000,
        n_actions=5,
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="distance_penalty",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
        value_range=(-3.0, 0.0),
        state_dependent=True,
        scales_with="distance",
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="collision_penalty",
        reward_type=RewardType.ON_EVENT,
        value=-1.0,
        requires_action=False,
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=7e-4,
        entropy_coeff=0.01,
        num_envs=128,
        n_actors=128,
        total_steps=2_000_000,
    )
    engine.print_report(model, config)

    print()
    print("The cooperative navigation reward is purely distance-based.")
    print("Unlike locomotion tasks, there's no passive alive bonus —")
    print("standing still gives constant negative reward unless you")
    print("happen to start on a landmark. The tool correctly sees the")
    print("intentional distance penalty as the primary signal.")


if __name__ == "__main__":
    run_example()
