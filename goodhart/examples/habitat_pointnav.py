"""Example: Habitat PointNav — distance decrease shaping.

Habitat PointNav uses delta-distance shaping: r_t = d_{t-1} - d_t.
This is potential-based (Phi = -d_t), so by Ng 1999 it preserves
optimal policy. However, the slack reward (+2.5 for reaching within
0.2m) combined with success reward (+10) creates a large discontinuity.

Source: Savva et al. 2019 (ICCV, Habitat), Wijmans et al. 2020 (ICLR)
Tool should catch: potential oscillation risk if not properly bounded
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Habitat PointNav — distance shaping + slack reward")
    print("=" * 70)
    print()
    print("Source: Savva et al. 2019, Wijmans et al. 2020")
    print("r_t = (d_{t-1} - d_t) + 2.5*success + 10*success")
    print("Distance shaping is potential-based (Ng 1999), but")
    print("slack + success creates a large reward discontinuity.")
    print()

    model = EnvironmentModel(
        name="Habitat PointNav",
        max_steps=500,
        gamma=0.99,
        n_states=100000,
        n_actions=4,
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="distance_decrease",
        reward_type=RewardType.SHAPING,
        value=1.0,
        value_range=(-0.25, 0.25),
        state_dependent=True,
        scales_with="distance",
        requires_action=True,
        can_loop=False,  # potential-based
    ))
    model.add_reward_source(RewardSource(
        name="success_reward",
        reward_type=RewardType.TERMINAL,
        value=10.0,
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.2,
    ))
    model.add_reward_source(RewardSource(
        name="slack_reward",
        reward_type=RewardType.ON_EVENT,
        value=2.5,
        requires_action=True,
        discovery_probability=0.2,
    ))
    model.add_reward_source(RewardSource(
        name="step_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.01,
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=2.5e-4,
        entropy_coeff=0.01,
        num_epochs=4,
        clip_epsilon=0.2,
        num_envs=4,
        n_actors=4,
        total_steps=250_000_000,
        rollout_length=128,
    )
    engine.print_report(model, config)

    print()
    print("The distance shaping is potential-based (can_loop=False),")
    print("so it correctly won't trigger the respawning exploit rule.")
    print("The tool focuses on the step penalty ratio and budget.")


if __name__ == "__main__":
    run_example()
