"""Krakovna entry: Go -- agent passes forever to maximize average score."""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "krakovna_go_pass",
    "source_paper": "Chew 2019 (informal)",
    "paper_url": None,
    "year": 2019,
    "domain": "game_ai",
    "encoding_basis": "catalog_derived",
    "verification_date": "2026-04-29",
    "brief_summary": (
        "Go agent was supposed to play and win. Instead it passes"
        " every turn to avoid risking a loss, maximizing average"
        " score by forcing draws."
    ),
    "documented_failure": (
        "Agent passes forever. Average-score reward makes draws"
        " (0 points) preferable to risky play that might lose."
    ),
    "failure_mechanism": "idle_exploit",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "source_quality": (
            "Informal source (Chew 2019). No peer-reviewed paper"
            " found. Encoding derived from Krakovna catalog description."
        ),
        "draw_by_passing": (
            "Terminal reward of 0.0 with discovery_probability=1.0."
            " Passing is trivially discoverable and yields a safe"
            " draw, dominating the EV of risky play."
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
    print("Krakovna: Go pass forever (Chew 2019)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Go pass (Chew 2019)",
        max_steps=500,
        gamma=1.0,
        n_actions=362,  # 19x19 board + pass
        action_type="discrete",
        death_probability=0.0,
    )

    # Win reward
    model.add_reward_source(RewardSource(
        name="win_game",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        discovery_probability=0.3,
        intentional=True,
    ))

    # Lose penalty
    model.add_reward_source(RewardSource(
        name="lose_game",
        reward_type=RewardType.TERMINAL,
        value=-1.0,
        discovery_probability=0.3,
    ))

    # Draw by passing -- the trap
    # Passing every turn yields 0.0 terminal reward, which dominates
    # the EV of risky play when the agent is weak.
    model.add_reward_source(RewardSource(
        name="draw_by_passing",
        reward_type=RewardType.TERMINAL,
        value=0.0,
        discovery_probability=1.0,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
