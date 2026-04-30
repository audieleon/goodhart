"""Krakovna entry: Reward modeling -- Montezuma's Revenge.

Learned reward model overvalues approach trajectory; agent moves
toward key without ever grabbing it.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "krakovna_reward_modeling_mr",
    "source_paper": (
        'Ibarz et al 2018, "Reward Learning from Human'
        ' Preferences and Demonstrations in Atari," NeurIPS'
        ' 2018'
    ),
    "paper_url": "https://arxiv.org/abs/1811.06521",
    "year": 2018,
    "domain": "game_ai",
    "encoding_basis": "catalog_derived",
    "brief_summary": (
        "Learned reward model trained from human demos"
        " overvalues the approach trajectory and undervalues"
        " the pickup event. Agent stops short of the key."
    ),
    "documented_failure": (
        "Agent approaches the key without grabbing it,"
        " maximizing the learned model's approach signal."
    ),
    "failure_mechanism": "learned_reward",
    "detection_type": "specification",
    "is_negative_example": True,
    "compute_cost_class": "low",
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Reward modeling — Montezuma's Revenge (Ibarz et al 2018)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Reward Modeling MR (Ibarz et al 2018)",
        max_steps=4500,
        gamma=0.99,
        n_actions=18,
        action_type="discrete",
        death_probability=0.05,
    )

    # Learned reward model — per-step signal that overvalues approach.
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
    print("Advisory: learned_reward — the reward model overvalues the")
    print("approach trajectory to the key and undervalues the actual")
    print("collection event. The agent converges to approach-without-")
    print("grabbing because that is the reward-maximizing behavior in")
    print("the learned model's space. Structural analysis cannot detect")
    print("biases internal to a learned reward model.")


if __name__ == "__main__":
    run_example()
