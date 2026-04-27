"""Example: LunarLander-v2 — well-designed reward (Gymnasium).

LunarLander uses potential-based shaping (distance decrease, velocity
decrease) plus sparse landing bonus and leg contact rewards. By Ng 1999,
the shaping component preserves optimal policy. This config should pass
clean — demonstrating the tool doesn't false-positive on good design.

Source: Brockman et al. 2016 (OpenAI Gym), Ng et al. 1999 (PBRS theory)
Tool should: pass clean (no structural reward traps)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("LunarLander-v2 — potential-based shaping (should pass clean)")
    print("=" * 70)
    print()
    print("Source: Brockman et al. 2016, Ng et al. 1999")
    print("LunarLander uses distance-based shaping that is potential-based")
    print("(Phi = -distance - velocity), so it preserves optimal policy.")
    print()

    model = EnvironmentModel(
        name="LunarLander-v2",
        max_steps=1000,
        gamma=0.99,
        n_states=10000,
        n_actions=4,
        death_probability=0.1,
    )
    model.add_reward_source(RewardSource(
        name="distance_shaping",
        reward_type=RewardType.SHAPING,
        value=1.0,
        value_range=(-1.0, 1.0),
        state_dependent=True,
        scales_with="distance",
        requires_action=True,
        can_loop=False,  # potential-based: cycles net zero
    ))
    model.add_reward_source(RewardSource(
        name="velocity_shaping",
        reward_type=RewardType.SHAPING,
        value=0.5,
        state_dependent=True,
        scales_with="velocity",
        requires_action=True,
        can_loop=False,  # potential-based
    ))
    model.add_reward_source(RewardSource(
        name="landing_bonus",
        reward_type=RewardType.TERMINAL,
        value=100.0,
        requires_action=True,
        discovery_probability=0.3,
    ))
    model.add_reward_source(RewardSource(
        name="crash_penalty",
        reward_type=RewardType.ON_EVENT,
        value=-100.0,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="leg_contact",
        reward_type=RewardType.ON_EVENT,
        value=10.0,
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="fuel_cost",
        reward_type=RewardType.PER_STEP,
        value=-0.03,
        requires_action=True,
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.01,
        num_envs=16,
        n_actors=16,
        total_steps=1_000_000,
    )
    result = engine.print_report(model, config)

    print()
    if result.passed:
        print("As expected: LunarLander passes clean.")
        print("Potential-based shaping (Ng 1999) preserves optimal policy,")
        print("and the reward components are well-balanced.")
    else:
        print("Unexpected findings — investigate whether the model")
        print("accurately captures LunarLander's reward structure.")


if __name__ == "__main__":
    run_example()
