"""Krakovna entry: Line following robot -- goes backward along the line."""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "krakovna_line_follower",
    "source_paper": "Vamplew 2004",
    "paper_url": "https://figshare.utas.edu.au/articles/conference_contribution/Lego_Mindstorms_Robots_as_a_Platform_for_Teaching_Reinforcement_Learning/23212277",
    "year": 2004,
    "domain": "control",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-29",
    "brief_summary": (
        "Robot was supposed to follow a line forward. Instead it"
        " reversed because on-track reward had no directional"
        " constraint."
    ),
    "documented_failure": (
        "Robot follows the line backward. On-track reward gives"
        " bonus for staying on the line regardless of direction."
        " Going backward is equally rewarded."
    ),
    "failure_mechanism": "specification_gap",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "source_quality": (
            "Vamplew 2004. Paper is known but not read in detail."
            " Encoding derived from Krakovna catalog description."
        ),
        "on_track_bonus": (
            "Per-step reward for staying on the line, direction-"
            "agnostic. Going backward collects the same reward."
        ),
    },
    "paper_url": "https://figshare.utas.edu.au/articles/conference_contribution/Lego_Mindstorms_Robots_as_a_Platform_for_Teaching_Reinforcement_Learning/23212277",
    "reward_location": "Vamplew 2004, AISAT2004 pp.70-75: Sarsa on Lego Mindstorms, line-following task. On-track reward without directional constraint.",
    "verification_date": "2026-04-30",
    "discovery_stage": "during_training",
    "source_code_url": None,
    "fix_known": None,
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Line following robot (Vamplew 2004)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Line follower (Vamplew 2004)",
        max_steps=1000,
        gamma=0.99,
        n_actions=3,
        action_type="discrete",
        death_probability=0.0,
    )

    # Intended: reach end of track
    model.add_reward_source(RewardSource(
        name="reach_end",
        reward_type=RewardType.TERMINAL,
        value=10.0,
        discovery_probability=0.02,
        intentional=True,
    ))

    # On-track bonus -- the trap
    # Rewards staying on the line regardless of direction.
    # Going backward collects the same per-step reward.
    model.add_reward_source(RewardSource(
        name="on_track_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.1,
        requires_action=True,
        can_loop=True,
        loop_period=2,
    ))

    # Off-track penalty
    model.add_reward_source(RewardSource(
        name="off_track_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.5,
        requires_action=False,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
