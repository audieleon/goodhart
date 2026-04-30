"""Krakovna entry: Football -- agent kicks ball out of bounds.

Agent kicks out of bounds for throw-in positioning instead of
dribbling through defenders.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "krakovna_football",
    "source_paper": (
        'Kurach et al 2020, "Google Research Football: A Novel'
        ' Reinforcement Learning Environment," AAAI 2020'
    ),
    "paper_url": "https://arxiv.org/abs/1907.11180",
    "year": 2020,
    "domain": "game_ai",
    "encoding_basis": "catalog_derived",
    "brief_summary": (
        "Sparse goal reward lets the agent discover that"
        " throw-in positioning via out-of-bounds kicks is"
        " higher-EV than direct play."
    ),
    "documented_failure": (
        "Agent kicks the ball out of bounds to trigger a"
        " throw-in for favorable positioning."
    ),
    "failure_mechanism": "proxy_reward",
    "detection_type": "structural",
    "is_negative_example": False,
    "compute_cost_class": "low",
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Football out-of-bounds exploit (Kurach et al 2019)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Football (Kurach et al 2019)",
        max_steps=3000,
        gamma=0.997,
        n_actions=19,
        action_type="discrete",
    )

    # Scoring a goal -- the intended reward
    model.add_reward_source(RewardSource(
        name="goal_scored",
        reward_type=RewardType.ON_EVENT,
        value=1.0,
        max_occurrences=0,
        discovery_probability=0.05,
        intentional=True,
    ))

    # Conceding a goal
    model.add_reward_source(RewardSource(
        name="goal_conceded",
        reward_type=RewardType.ON_EVENT,
        value=-1.0,
        max_occurrences=0,
    ))

    # Throw-in exploit -- the trap
    # Kicking out of bounds triggers a throw-in that grants favorable
    # positioning. This is effectively a repeatable event with
    # implicit positive value from the resulting field position.
    model.add_reward_source(RewardSource(
        name="throw_in_positioning",
        reward_type=RewardType.ON_EVENT,
        value=0.0,
        max_occurrences=0,
        can_loop=True,
        loop_period=30,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
