"""Example: Isaac Gym Ant — massively parallel locomotion.

Alive bonus (0.5) + up_weight (0.1) give 0.6/step passive reward vs
~1.0/step for locomotion, creating a moderate idle exploit floor at 38%.

Source: Makoviychuk et al. 2021 (NeurIPS), IsaacGymEnvs
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "isaac_gym_ant",
    "source_paper": "Makoviychuk et al. 2021 (NeurIPS), IsaacGymEnvs",
    "paper_url": "https://arxiv.org/abs/2108.10470",
    "source_code_url": "https://github.com/NVIDIA-Omniverse/IsaacGymEnvs",
    "reward_location": "Reward structure from paper description",
    "year": 2021,
    "domain": "locomotion",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to run forward. Alive bonus + up_weight give 0.6/step passive vs ~1.0/step for locomotion, creating a 38% idle floor.",
    "documented_failure": "alive_bonus (0.5/step) + up_weight (0.1/step) create 0.6/step passive reward vs ~1.0/step for locomotion; idle exploit is moderate but present at 38% of walking reward",
    "failure_mechanism": "idle_exploit",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Massive parallelism (4096 envs) and large compute budget mitigate in practice",
    "compute_cost_class": "high",
    "is_negative_example": True,
    "encoding_rationale": {
        "moderate_idle_risk": "Passive reward is 38% of active, lower than Humanoid but still notable",
        "massive_parallel": "4096 parallel envs provide sufficient exploration to overcome idle floor",
    },
}


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
    model.add_reward_source(
        RewardSource(
            name="progress",
            reward_type=RewardType.PER_STEP,
            value=1.0,
            state_dependent=True,
            scales_with="velocity",
            requires_action=True,
            intentional=True,
        )
    )
    model.add_reward_source(
        RewardSource(
            name="alive_bonus",
            reward_type=RewardType.PER_STEP,
            value=0.5,
            respawn=RespawnBehavior.INFINITE,
            requires_action=False,
            intentional=False,
        )
    )
    model.add_reward_source(
        RewardSource(
            name="up_weight",
            reward_type=RewardType.PER_STEP,
            value=0.1,
            state_dependent=True,
            requires_action=False,
            intentional=False,
        )
    )
    model.add_reward_source(
        RewardSource(
            name="actions_cost",
            reward_type=RewardType.PER_STEP,
            value=-0.005,
            requires_action=True,
        )
    )
    model.add_reward_source(
        RewardSource(
            name="energy_cost",
            reward_type=RewardType.PER_STEP,
            value=-0.05,
            state_dependent=True,
            requires_action=True,
        )
    )
    model.add_reward_source(
        RewardSource(
            name="joints_at_limit_cost",
            reward_type=RewardType.PER_STEP,
            value=-0.1,
            state_dependent=True,
            requires_action=True,
        )
    )

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
