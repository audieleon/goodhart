"""Example: Isaac Gym Ant — massively parallel locomotion.

Isaac Gym Ant uses progress + alive bonus + energy penalties at
massive scale (4096 parallel envs). The alive bonus (0.5/step) is
small relative to progress reward (~1.0/step), making idle exploit
less likely than in MuJoCo Humanoid. However, the up_weight term
creates a passive reward for maintaining orientation.

Source: Makoviychuk et al. 2021 (NeurIPS), IsaacGymEnvs
Tool should catch: moderate idle exploit risk from combined
  alive_bonus + up_weight
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Isaac Gym Ant — massively parallel locomotion")
    print("=" * 70)
    print()
    print("Source: Makoviychuk et al. 2021 (NeurIPS)")
    print("4096 parallel environments, 8 reward components.")
    print()

    model = EnvironmentModel(
        name="Isaac Gym Ant",
        max_steps=1000,
        gamma=0.99,
        n_states=100000,
        n_actions=8,
        death_probability=0.02,
    )
    model.add_reward_source(RewardSource(
        name="progress",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        state_dependent=True,
        scales_with="velocity",
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="alive_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        respawn=RespawnBehavior.INFINITE,
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="up_weight",
        reward_type=RewardType.PER_STEP,
        value=0.1,
        state_dependent=True,
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="actions_cost",
        reward_type=RewardType.PER_STEP,
        value=-0.005,
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="energy_cost",
        reward_type=RewardType.PER_STEP,
        value=-0.05,
        state_dependent=True,
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="joints_at_limit_cost",
        reward_type=RewardType.PER_STEP,
        value=-0.1,
        state_dependent=True,
        requires_action=True,
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.0,
        num_epochs=5,
        clip_epsilon=0.2,
        num_envs=4096,
        n_actors=4096,
        total_steps=100_000_000,
        rollout_length=16,
        minibatch_size=32768,
    )
    engine.print_report(model, config)

    print()
    print("Isaac Gym Ant's alive_bonus (0.5) + up_weight (0.1) give")
    print("0.6/step passive reward vs ~1.0/step for locomotion.")
    print("At massive scale (4096 envs), the budget is generous but")
    print("the idle floor is still 38% of walking reward.")


if __name__ == "__main__":
    run_example()
