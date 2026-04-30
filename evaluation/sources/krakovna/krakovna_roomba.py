"""Krakovna entry: Roomba -- drives backward to avoid bumper penalty."""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "krakovna_roomba",
    "source_paper": "Custard Smingleigh (Twitter anecdote)",
    "paper_url": None,
    "year": 2018,
    "domain": "control",
    "encoding_basis": "unverified_folklore",
    "verification_date": "2026-04-29",
    "brief_summary": (
        "Roomba was supposed to drive forward and clean. Instead it"
        " drives backward to avoid triggering the front bumper"
        " collision sensor."
    ),
    "documented_failure": (
        "Drives backward to avoid front bumper penalty. Speed"
        " reward is direction-agnostic but bumper penalty only"
        " applies when driving forward, making backward motion"
        " strictly dominant."
    ),
    "failure_mechanism": "specification_gap",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "source_quality": (
            "Anecdotal source (Custard Smingleigh, Twitter). No"
            " peer-reviewed paper or verifiable source. Widely"
            " cited in AI safety discussions but unverified."
            " Encoding derived from Krakovna catalog description."
        ),
        "speed_reward": (
            "Per-step reward scaling with velocity, earned"
            " regardless of direction."
        ),
        "front_bumper_penalty": (
            "Asymmetric penalty only triggers on forward collisions."
            " Driving backward avoids it entirely."
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
    print("Krakovna: Roomba backward driving (Custard Smingleigh)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Roomba (Custard Smingleigh)",
        max_steps=1000,
        gamma=0.99,
        n_actions=4,
        action_type="continuous",
    )

    # Speed reward -- earned regardless of direction
    model.add_reward_source(RewardSource(
        name="speed_reward",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=True,
        scales_with="velocity",
        state_dependent=True,
        intentional=True,
    ))

    # Front bumper penalty -- the asymmetry
    # Only triggers when driving forward into obstacles.
    # Driving backward avoids this entirely.
    model.add_reward_source(RewardSource(
        name="front_bumper_penalty",
        reward_type=RewardType.ON_EVENT,
        value=-5.0,
        max_occurrences=0,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
