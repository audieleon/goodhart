"""Krakovna entry: Reward modeling -- Hero (Atari).

Learned reward model overvalues shooting motion; agent deliberately
misses spiders to maximize the learned signal.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "krakovna_reward_modeling_hero",
    "source_paper": (
        'Ibarz et al 2018, "Reward Learning from Human'
        ' Preferences and Demonstrations in Atari," NeurIPS'
        ' 2018'
    ),
    "paper_url": "https://arxiv.org/abs/1811.06521",
    "year": 2018,
    "domain": "game_ai",
    "encoding_basis": "primary_source",
    "brief_summary": (
        "Learned reward model trained from human demos"
        " overvalues the shooting motion relative to the"
        " hit outcome. Agent deliberately barely misses."
    ),
    "documented_failure": (
        "Agent shoots at spiders but deliberately misses,"
        " exploiting biases in the learned reward model."
    ),
    "failure_mechanism": "learned_reward",
    "detection_type": "specification",
    "is_negative_example": True,
    "compute_cost_class": "low",
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Reward modeling — Hero (Ibarz et al 2018)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Reward Modeling Hero (Ibarz et al 2018)",
        max_steps=4500,
        gamma=0.99,
        n_actions=18,
        action_type="discrete",
        death_probability=0.05,
    )

    # Learned reward model — provides per-step signal.
    # We encode the structural form, but the internal biases of
    # the learned model are not representable.
    model.add_reward_source(RewardSource(
        name="learned_reward_signal",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
        max_occurrences=0,
    ))

    engine.print_report(model)

    print()
    print("Advisory: learned_reward — the reward signal comes from a")
    print("learned model trained on human demonstrations. The model")
    print("overvalues the shooting motion relative to the hit outcome.")
    print("This bias is internal to the learned model and cannot be")
    print("detected by structural reward analysis. The encoding is")
    print("minimal because the reward function is opaque.")


if __name__ == "__main__":
    run_example()
