"""Example: DeepMind Tokamak Plasma Control (LIMITATION).

Reward structure is clean but missing coil balance term causes dangerous electromagnetic forces.
Source: Degrave et al. 2022 (Nature), "Magnetic control of tokamak plasmas through deep RL"
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "tokamak_plasma",
    "source_paper": "Degrave et al. 2022 (Nature), 'Magnetic control of tokamak plasmas through deep reinforcement learning'",
    "paper_url": "https://doi.org/10.1038/s41586-021-04301-9",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2022,
    "domain": "industrial",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to control plasma shape. It succeeded, but used asymmetric coil currents creating dangerous electromagnetic forces on the machine structure.",
    "documented_failure": "Missing coil balance term in reward. Agent found alternative path using asymmetric ohmic coil currents that achieved same plasma shape but created dangerous electromagnetic forces.",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "post_training",
    "fix_known": "Add coil current balance term to reward — requires domain expertise to enumerate safety constraints",
    "compute_cost_class": "high",
    "is_negative_example": True,
    "encoding_rationale": {
        "missing_term": "The problem is what is ABSENT from the config, not what is present",
        "positive_structure": "Reward structure that IS specified is clean and well-designed",
    },
}


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
    model.add_reward_source(
        RewardSource(
            name="plasma_shape_accuracy",
            reward_type=RewardType.PER_STEP,
            value=1.0,
            value_type="exponential",
            value_params={"sigma": 0.1},
            state_dependent=True,
            requires_action=True,
            intentional=True,
        )
    )
    model.add_reward_source(
        RewardSource(
            name="plasma_current_accuracy",
            reward_type=RewardType.PER_STEP,
            value=0.5,
            state_dependent=True,
            requires_action=True,
            intentional=True,
        )
    )
    model.add_reward_source(
        RewardSource(
            name="xpoint_location",
            reward_type=RewardType.PER_STEP,
            value=0.3,
            state_dependent=True,
            requires_action=True,
            intentional=True,
        )
    )
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
    engine.print_report(model, config)

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
