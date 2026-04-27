"""Example: DeepMind Control Suite Dog — rich physics, many rewards.

The Dog environment in DMC has 26 reward terms for locomotion quality,
posture, joint limits, foot contact, etc. This is exactly the pattern
where physics exploits emerge: rich dynamics, many interacting rewards,
low termination (the dog doesn't die, just falls and gets up).

Source: Tunyasuvunakool et al. 2020 (SIGGRAPH), Tassa et al. 2020 (DM Control)
Advisory fires: advisory_physics_exploit (complex env, low death, many rewards)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("DMC Dog — rich physics, many reward components")
    print("=" * 70)
    print()
    print("Source: Tunyasuvunakool et al. 2020 (SIGGRAPH)")
    print("26 reward terms, no termination. Perfect storm for")
    print("emergent physics exploits.")
    print()

    model = EnvironmentModel(
        name="DM Control Dog (run)",
        max_steps=1000,
        gamma=0.99,
        n_states=200000,
        n_actions=38,  # 38 actuators
        death_probability=0.0,  # no termination
    )
    model.add_reward_source(RewardSource(
        name="velocity_tracking",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        value_type="exponential",
        value_params={"sigma": 0.5},
        state_dependent=True,
        scales_with="velocity",
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="upright_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.1,
        state_dependent=True,
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="posture_reward",
        reward_type=RewardType.PER_STEP,
        value=0.1,
        state_dependent=True,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="joint_limit_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.05,
        state_dependent=True,
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="torque_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.001,
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="foot_contact",
        reward_type=RewardType.PER_STEP,
        value=0.05,
        requires_action=False,
        intentional=False,
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.003,
        num_epochs=3,
        num_envs=2048,
        n_actors=2048,
        total_steps=500_000_000,
        rollout_length=16,
    )
    result = engine.print_report(model, config)

    print()
    print("The advisory_physics_exploit fires here: 200K states, 38")
    print("actions, 6 reward components, zero termination. In practice,")
    print("DMC Dog agents discover sliding gaits, leg-crossing exploits,")
    print("and torque-minimizing twitches that technically maximize")
    print("reward but look nothing like dog locomotion.")


if __name__ == "__main__":
    run_example()
