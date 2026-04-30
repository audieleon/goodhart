"""Krakovna entry: Road Runner -- agent kills itself at end of level 1.

Agent dies at end of level 1 to replay easier content rather than
risk losing points in harder level 2.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "krakovna_road_runner",
    "source_paper": (
        'Saunders et al 2017, "Trial without Error: Towards'
        ' Safe Reinforcement Learning via Human Intervention"'
    ),
    "paper_url": "https://arxiv.org/abs/1707.05173",
    "year": 2017,
    "domain": "game_ai",
    "encoding_basis": "primary_source",
    "brief_summary": (
        "Score reward without level-progress incentive. Dying"
        " after maximizing level 1 score preserves the high"
        " score by avoiding harder level 2."
    ),
    "documented_failure": (
        "Agent commits suicide at end of level 1 to avoid"
        " losing points in harder level 2."
    ),
    "failure_mechanism": "death_reset_exploit",
    "detection_type": "structural",
    "is_negative_example": False,
    "compute_cost_class": "low",
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Road Runner (Saunders et al 2017)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Road Runner (Saunders et al 2017)",
        max_steps=4500,
        gamma=0.99,
        n_actions=18,
        action_type="discrete",
    )

    # Level 1 score (collectible points)
    model.add_reward_source(RewardSource(
        name="level_1_score",
        reward_type=RewardType.ON_EVENT,
        value=5.0,
        max_occurrences=20,
        intentional=True,
    ))

    # Level 2 score (harder, risky)
    model.add_reward_source(RewardSource(
        name="level_2_score",
        reward_type=RewardType.ON_EVENT,
        value=5.0,
        max_occurrences=20,
        discovery_probability=0.1,
        prerequisite="level_1_score",
    ))

    # Death at level transition -- the exploit
    # Agent intentionally dies to avoid level 2's harder content.
    # Death resets via ON_DEATH, and the level 1 score is preserved
    # in the episode return.
    model.add_reward_source(RewardSource(
        name="intentional_death",
        reward_type=RewardType.ON_EVENT,
        value=0.0,
        max_occurrences=1,
        respawn=RespawnBehavior.ON_DEATH,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
