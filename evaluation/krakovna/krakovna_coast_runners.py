"""Krakovna entry: CoastRunners boat race -- agent loops for points.

Agent drives in circles collecting respawning waypoint bonuses
instead of finishing the race.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "krakovna_coast_runners",
    "source_paper": (
        'Clark & Amodei 2016, "Faulty Reward Functions in the'
        ' Wild," OpenAI Blog'
    ),
    "year": 2016,
    "domain": "game_ai",
    "encoding_basis": "primary_source",
    "brief_summary": (
        "Waypoint rewards respawn on a loop. Per-lap waypoint"
        " total exceeds the race-completion bonus, so the"
        " optimal proxy policy is to never finish."
    ),
    "documented_failure": (
        "Agent loops collecting turbo powerups instead of"
        " finishing the race."
    ),
    "failure_mechanism": "respawning_loop",
    "detection_type": "structural",
    "is_negative_example": False,
    "compute_cost_class": "low",
    "paper_url": "https://blog.openai.com/faulty-reward-functions/",
    "reward_location": "Clark & Amodei 2016 blog: turbo pads respawn, race finish is terminal score.",
    "verification_date": "2026-04-30",
    "discovery_stage": "post_training",
    "source_code_url": None,
    "fix_known": None,
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: CoastRunners boat race (Amodei & Clark 2016)")
    print("=" * 70)

    model = EnvironmentModel(
        name="CoastRunners (Amodei & Clark 2016)",
        max_steps=2000,
        gamma=0.99,
        n_actions=9,
        action_type="discrete",
    )

    # Finishing the race
    model.add_reward_source(RewardSource(
        name="finish_race",
        reward_type=RewardType.TERMINAL,
        value=10.0,
        discovery_probability=0.1,
        intentional=True,
    ))

    # Waypoint bonuses -- the trap
    # Respawn each lap; looping is more profitable than finishing.
    model.add_reward_source(RewardSource(
        name="waypoint_bonus",
        reward_type=RewardType.ON_EVENT,
        value=3.0,
        respawn=RespawnBehavior.TIMED,
        respawn_time=50,
        max_occurrences=0,
        can_loop=True,
        loop_period=60,
    ))

    # Occasionally catching fire (collision penalty)
    model.add_reward_source(RewardSource(
        name="collision_penalty",
        reward_type=RewardType.ON_EVENT,
        value=-1.0,
        max_occurrences=0,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
