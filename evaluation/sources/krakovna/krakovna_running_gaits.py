"""Krakovna entry: Running gaits -- unusual locomotion strategies.

Forward velocity reward without naturalness constraints; agent
develops bizarre gaits that exploit simulator physics.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "krakovna_running_gaits",
    "source_paper": (
        'Kidzinski et al 2018, "Learning to Run Challenge"'
    ),
    "paper_url": "https://arxiv.org/abs/1804.00361",
    "year": 2018,
    "domain": "locomotion",
    "encoding_basis": "primary_source",
    "brief_summary": (
        "Forward velocity reward without naturalness constraint."
        " Agent exploits simulator physics to find gaits that"
        " are efficient in simulation but physically unnatural."
    ),
    "documented_failure": (
        "Agent develops bizarre, unnatural gaits (hopping,"
        " skipping, dragging limbs) instead of human-like"
        " running."
    ),
    "failure_mechanism": "proxy_reward",
    "detection_type": "structural",
    "is_negative_example": False,
    "compute_cost_class": "low",
    "paper_url": "https://arxiv.org/abs/1804.00361",
    "reward_location": "Kidzinski et al. 2018 Section 1: r_t = delta_x_pelvis - ligaments/10M. 41D state, 18D actions, 1000 steps.",
    "verification_date": "2026-04-30",
    "discovery_stage": "during_training",
    "source_code_url": None,
    "fix_known": None,
    "encoding_rationale": {
        "basis": "Encoded from Krakovna specification gaming catalog with paper verification where available.",
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Running gaits (Kidzinski et al 2018)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Running gaits (Kidzinski et al 2018)",
        max_steps=1000,
        gamma=0.99,
        n_actions=18,
        action_type="continuous",
    )

    # Forward velocity -- the only objective
    # Without naturalness constraints, any gait that maximizes
    # velocity is optimal, including unnatural ones.
    model.add_reward_source(RewardSource(
        name="forward_velocity",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=True,
        scales_with="velocity",
        state_dependent=True,
        intentional=True,
    ))

    # Alive bonus (standard)
    model.add_reward_source(RewardSource(
        name="alive_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.1,
        requires_action=False,
        intentional=True,
    ))

    # No naturalness penalty exists -- the structural gap.
    # A style or energy-efficiency penalty would constrain
    # the gait space to more realistic locomotion.

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
