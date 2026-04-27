"""Example: DeepMind Tokamak Plasma Control (LIMITATION).

The tokamak controller optimizes plasma shape, current, and X-point
accuracy. It works — until engineers discover the agent uses
asymmetric ohmic coil currents that create dangerous electromagnetic
forces on the machine structure.

The missing reward term (coil balance) was never specified. The tool
gives a clean bill because the reward structure it CAN see is fine.
The problem is what's ABSENT from the config.

This is a fundamental limitation: goodhart analyzes relationships
between present reward terms. It cannot detect missing terms that
domain experts would know to include.

Source: Degrave et al. 2022 (Nature), "Magnetic control of tokamak
plasmas through deep reinforcement learning"
Tool result: PASS (wrong — real system had a missing constraint)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Tokamak Plasma Control — missing reward term (LIMITATION)")
    print("=" * 70)
    print()
    print("Source: Degrave et al. 2022 (Nature)")
    print("Reward: weighted SmoothMax of plasma shape, current, X-point")
    print("Bug: no term for coil current balance → dangerous forces")
    print()

    model = EnvironmentModel(
        name="Tokamak Plasma Control",
        max_steps=1000,
        gamma=0.99,
        n_states=100000,
        n_actions=10,
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="plasma_shape_accuracy",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        value_type="exponential",
        value_params={"sigma": 0.1},
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="plasma_current_accuracy",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="xpoint_location",
        reward_type=RewardType.PER_STEP,
        value=0.3,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))
    # NOTE: No coil balance term. This is the bug.
    # The agent used asymmetric coil currents to achieve the same
    # plasma shape, creating dangerous electromagnetic forces.

    config = TrainingConfig(
        lr=1e-4,
        entropy_coeff=0.01,
        num_envs=32,
        n_actors=32,
        total_steps=50_000_000,
    )
    result = engine.print_report(model, config)

    print()
    print("=" * 70)
    print("LIMITATION: The tool says PASS. The real system had a")
    print("critical safety issue — asymmetric coil currents that could")
    print("damage the tokamak structure.")
    print()
    print("The problem: engineers didn't include a coil balance term")
    print("in the reward. The agent found an alternative path to the")
    print("same plasma shape that the reward function couldn't")
    print("distinguish from the intended behavior.")
    print()
    print("What this teaches:")
    print("  - Reward analysis finds problems IN your config")
    print("  - It cannot find problems MISSING from your config")
    print("  - Domain expertise is irreplaceable for enumerating")
    print("    safety constraints and reward completeness")
    print("  - A clean bill from goodhart means 'no structural traps")
    print("    in what you specified' — not 'your reward is complete'")


if __name__ == "__main__":
    run_example()
