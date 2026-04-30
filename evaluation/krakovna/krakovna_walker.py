"""Krakovna entry: Walker -- walks on one leg to match target speed."""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "krakovna_walker",
    "source_paper": "Lee et al 2021",
    "paper_url": None,
    "year": 2021,
    "domain": "locomotion",
    "encoding_basis": "catalog_derived",
    "verification_date": "2026-04-29",
    "brief_summary": (
        "Walker was supposed to walk with a natural gait. Instead"
        " it hops on one leg, matching target velocity without"
        " any naturalness constraint."
    ),
    "documented_failure": (
        "Walker hops on one leg. Speed-matching reward has no"
        " style or naturalness penalty, so the degenerate"
        " one-leg gait dominates."
    ),
    "failure_mechanism": "proxy_reward",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Add a gait symmetry term or motion-capture reference"
        " to constrain the solution to natural bipedal walking."
    ),
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "source_quality": (
            "Lee et al 2021. Paper is known but not read in detail."
            " Encoding derived from Krakovna catalog description."
        ),
        "speed_matching": (
            "Per-step exponential reward for matching target velocity."
            " No constraint on gait style or body posture."
        ),
    },
    "paper_url": None,
    "reward_location": "Reward structure inferred from Krakovna catalog description",
    "verification_date": "2026-04-30",
    "discovery_stage": "during_training",
    "source_code_url": None,
    "fix_known": None,
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Walker one-leg (Lee et al 2021)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Walker (Lee et al 2021)",
        max_steps=1000,
        gamma=0.99,
        n_actions=6,
        action_type="continuous",
    )

    # Speed-matching reward -- the trap
    # Reward is exp(-|v - v_target|^2 / sigma^2).
    # Matches target speed but does not constrain gait style.
    model.add_reward_source(RewardSource(
        name="speed_matching",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=True,
        scales_with="velocity",
        state_dependent=True,
        value_type="exponential",
        value_params={"sigma": 0.25},
        intentional=True,
    ))

    # Alive bonus
    model.add_reward_source(RewardSource(
        name="alive_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.1,
        requires_action=False,
        intentional=True,
    ))

    # No naturalness / symmetry penalty -- the structural gap.
    # Adding a gait symmetry term or motion-capture reference
    # would constrain the solution to natural bipedal walking.

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
