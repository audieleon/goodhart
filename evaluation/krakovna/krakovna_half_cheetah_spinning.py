"""Krakovna entry: Half Cheetah spinning in place."""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "krakovna_half_cheetah_spinning",
    "source_paper": "Zhang et al 2021 (uncertain attribution)",
    "paper_url": None,
    "year": 2021,
    "domain": "locomotion",
    "encoding_basis": "unverified_folklore",
    "verification_date": "2026-04-29",
    "brief_summary": (
        "Half Cheetah was supposed to run forward. Instead it spins"
        " in place, exploiting MuJoCo contact overflow to achieve"
        " extreme velocity magnitude."
    ),
    "documented_failure": (
        "Half Cheetah spins via MuJoCo overflow. Velocity reward"
        " tracks magnitude without constraining direction, so"
        " spinning scores higher than forward locomotion."
    ),
    "failure_mechanism": "physics_exploit",
    "detection_type": "dynamic",
    "discovery_stage": "during_training",
    "fix_known": (
        "Use forward velocity (dot product with heading) instead"
        " of raw speed magnitude."
    ),
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "source_quality": (
            "Uncertain attribution to Zhang et al 2021. Paper not"
            " confirmed or read. Encoding derived from Krakovna"
            " catalog description."
        ),
        "velocity_reward": (
            "Per-step reward proportional to velocity magnitude."
            " No directional constraint. Even without the MuJoCo"
            " overflow, the reward incentivizes any high-speed"
            " movement equally."
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
    print("Krakovna: Half Cheetah spinning (Zhang et al 2021)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Half Cheetah Spinning (Zhang et al 2021)",
        max_steps=1000,
        gamma=0.99,
        n_actions=6,
        action_type="continuous",
    )

    # Per-step velocity reward — no directional constraint.
    # The agent is rewarded for speed, not forward progress.
    # Structural flaw: velocity magnitude instead of forward velocity.
    model.add_reward_source(RewardSource(
        name="velocity_reward",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        value_range=(0.0, 100.0),
        value_type="proportional",
        value_params={"scale": 1.0},
        scales_with="velocity",
        state_dependent=True,
        requires_action=True,
        intentional=True,
        max_occurrences=0,
    ))

    # Small control cost penalty (standard in MuJoCo envs)
    model.add_reward_source(RewardSource(
        name="control_cost",
        reward_type=RewardType.PER_STEP,
        value=-0.1,
        requires_action=True,
        max_occurrences=0,
        intentional=True,
    ))

    engine.print_report(model)

    print()
    print("The structural flaw is that velocity reward has no directional")
    print("constraint. The MuJoCo overflow amplifies the problem, but the")
    print("reward would incentivize spinning even without the overflow —")
    print("any movement that achieves high speed is rewarded equally.")


if __name__ == "__main__":
    run_example()
