"""Example: Safety Gym with explicit constraint budget (advisory demo).

This example shows how the advisory_constrained_rl rule fires when
it detects negative event rewards that look like they approximate
safety constraints. The advisory explains that soft penalties are
NOT equivalent to hard cost budgets.

Source: Ray et al. 2019 (Safety Gym), Achiam et al. 2017 (CPO)
Advisory fires: advisory_constrained_rl (negative events as costs)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Safety Gym PointGoal2 — constraint advisory demo")
    print("=" * 70)
    print()
    print("Source: Ray et al. 2019, Achiam et al. 2017")
    print("Real config: cost budget = 25 hazard entries per episode.")
    print("Our model: hazard penalty = -0.1 (soft approximation).")
    print()

    model = EnvironmentModel(
        name="Safety Gym PointGoal2",
        max_steps=1000,
        gamma=0.99,
        n_states=100000,
        n_actions=2,
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="goal_reward",
        reward_type=RewardType.ON_EVENT,
        value=1.0,
        respawn=RespawnBehavior.ON_EPISODE,
        requires_action=True,
        discovery_probability=0.1,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="hazard_entry",
        reward_type=RewardType.ON_EVENT,
        value=-0.1,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="vase_contact",
        reward_type=RewardType.ON_EVENT,
        value=-0.1,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="distance_shaping",
        reward_type=RewardType.SHAPING,
        value=0.01,
        state_dependent=True,
        scales_with="distance",
        requires_action=True,
        can_loop=False,
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.01,
        num_envs=16,
        n_actors=16,
        total_steps=10_000_000,
    )
    result = engine.print_report(model, config)

    print()
    print("The advisory fires on hazard_entry and vase_contact.")
    print("Key insight: with soft penalty -0.1, the agent can enter")
    print("100 hazards per episode for only -10.0 total cost. If the")
    print("goal reward is +1.0, that's always worth it.")
    print()
    print("With a HARD constraint (budget=25), the 26th entry is")
    print("forbidden regardless of reward. Different optimization")
    print("landscape entirely.")


if __name__ == "__main__":
    run_example()
