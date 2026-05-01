"""Humanoid SaW — held-out evaluation (negative example).

van Marum et al. 2024 humanoid standing and walking. Structurally
sound; physics exploit advisory should fire but passes structurally.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "heldout_humanoid_saw",
    "source_paper": (
        'van Marum, Shrestha, Duan, Dugar, Dao & Fern, "Revisiting '
        'Reward Design and Evaluation for Robust Humanoid Standing '
        'and Walking," IROS 2024, pp. 11256-11263'
    ),
    "paper_url": "https://arxiv.org/abs/2404.19173",
    "source_code_url": "https://b-vm.github.io/Robust-SaW/",
    "reward_location": "Reward structure from paper description",
    "year": 2024,
    "domain": "locomotion",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Structurally sound locomotion reward. Alive bonus (0.5)"
        " is moderate relative to velocity tracking (1.0). Physics"
        " exploits (hopping gaits) are beyond structural detection."
    ),
    "documented_failure": "None — reward is well-designed.",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "unknown",
    "is_negative_example": True,
    "encoding_rationale": {
        "physics_advisory": (
            "Paper's actual issues (hopping gaits, simulator"
            " exploitation) are blind-spot advisories, not"
            " structural failures."
        ),
        "alive_bonus_intentional": (
            "Alive bonus marked intentional because survival"
            " IS the goal in locomotion."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Humanoid SaW — held-out evaluation")
    print("=" * 70)
    print()
    print("Source: van Marum et al. 2024, IROS 2024")
    print("Humanoid standing and walking with velocity tracking.")
    print("Structurally sound; physics advisory should fire.")
    print()

    model = EnvironmentModel(
        name="Humanoid SaW (Held-Out)",
        max_steps=1000,
        gamma=0.99,
        n_actions=21,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="velocity_tracking",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="alive_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        requires_action=False,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="smoothness_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.1,
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="energy_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.05,
        requires_action=True,
    ))

    engine.print_report(model)

    print()
    print("The tool should pass structurally: alive_bonus (0.5) is")
    print("moderate relative to velocity_tracking (1.0), and the")
    print("penalties (-0.1 smoothness, -0.05 energy) are well-scaled.")
    print("The physics exploit advisory should fire for continuous")
    print("control locomotion, noting that sim-to-real gaps and")
    print("physics exploits (hopping gaits, ground glitching) are")
    print("possible but cannot be detected from structure alone.")


if __name__ == "__main__":
    run_example()
