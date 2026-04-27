"""Example: PettingZoo adversarial — non-stationarity advisory.

Simple tag (predator-prey) has symmetric rewards: predator gets +1
for catching, prey gets -1 for being caught. The advisory fires
because symmetric terminal rewards suggest competitive dynamics.

Source: Terry et al. 2021 (PettingZoo), inspired by Bansal et al. 2018
Advisory fires: advisory_nonstationarity (symmetric win/lose)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("PettingZoo Simple Tag — adversarial non-stationarity")
    print("=" * 70)
    print()
    print("Source: Terry et al. 2021, Bansal et al. 2018")
    print("Predator +1 for catch, prey -1 for caught.")
    print("Symmetric rewards hint at competitive dynamics.")
    print()

    # Predator's perspective
    model = EnvironmentModel(
        name="Simple Tag (predator)",
        max_steps=100,
        gamma=0.99,
        n_states=10000,
        n_actions=5,
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="catch",
        reward_type=RewardType.TERMINAL,
        value=10.0,
        requires_action=True,
        discovery_probability=0.3,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="timeout",
        reward_type=RewardType.TERMINAL,
        value=-10.0,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="distance_shaping",
        reward_type=RewardType.SHAPING,
        value=0.1,
        state_dependent=True,
        scales_with="distance",
        requires_action=True,
        can_loop=False,
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.01,
        num_envs=8,
        n_actors=8,
        total_steps=5_000_000,
    )
    result = engine.print_report(model, config)

    print()
    print("The advisory_nonstationarity fires because +10 catch")
    print("and -10 timeout are roughly symmetric, suggesting")
    print("competitive dynamics. In self-play, the prey improves")
    print("while the predator trains — the effective MDP shifts")
    print("continuously. goodhart can't model this, but the hint")
    print("tells you to watch for forgetting cycles and collapse.")


if __name__ == "__main__":
    run_example()
