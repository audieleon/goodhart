"""Humanoid SaW — held-out evaluation (negative example).

van Marum et al. 2024. 13-component reward from Table I.
No alive bonus. Feet airtime (1.0) is dominant term.
Structurally sound; physics exploits are outside scope.
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
    "reward_location": (
        "Section IV-B, Table I: 13 reward terms with exact formulas "
        "and weights. Dominant: feet airtime (1.0), x,y velocity "
        "(0.15 each), roll/pitch orient (0.2), base accel (0.1), "
        "feet contact (0.1), yaw orient (0.1). Penalties: action "
        "diff (0.02), torque (0.02), arm (0.03). 20 actuators, "
        "50Hz, PPO, Digit V3 humanoid."
    ),
    "year": 2024,
    "domain": "locomotion",
    "encoding_basis": "primary_source",
    "verification_date": "2026-05-01",
    "brief_summary": (
        "13-component humanoid locomotion reward with NO alive bonus. "
        "Feet airtime (weight 1.0) is the dominant term — prevents "
        "hopping by rewarding single-foot contact. Deployed on "
        "Digit V3 robot. Paper's actual issues (hopping gaits, "
        "sim-to-real style gaps) are physics exploits beyond "
        "structural detection."
    ),
    "documented_failure": "None — reward is well-designed.",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "post_deployment",
    "fix_known": None,
    "compute_cost_class": "high",
    "is_negative_example": True,
    "encoding_rationale": {
        "no_alive_bonus": (
            "Paper explicitly does NOT use an alive bonus. The "
            "feet airtime term (weight 1.0) serves a similar "
            "function by rewarding ground contact."
        ),
        "13_terms_simplified": (
            "Simplified to representative groups: velocity tracking "
            "(0.3 total), orientation (0.3), feet contact/airtime "
            "(1.1), regularization (0.07). Full Table I weights "
            "documented in reward_location."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Table I: 13 reward terms, Digit V3, 20 actuators, PPO

    model = EnvironmentModel(
        name="Humanoid SaW (van Marum et al. 2024)",
        max_steps=800,  # 16 seconds at 50Hz
        gamma=0.99,
        n_actions=20,  # 20 actuators
        action_type="continuous",
        death_probability=0.05,  # falls terminate
    )

    # Velocity tracking: x,y velocity (0.15 each = 0.30 total)
    # exp(-5·(v - c)) when standing command
    model.add_reward_source(RewardSource(
        name="velocity_tracking",
        reward_type=RewardType.PER_STEP,
        value=0.30,  # 0.15 x + 0.15 y
        requires_action=True,
        intentional=True,
        state_dependent=True,
        scales_with="velocity",
        value_type="exponential",
    ))

    # Orientation: yaw (0.1) + roll/pitch (0.2) = 0.30
    model.add_reward_source(RewardSource(
        name="orientation_tracking",
        reward_type=RewardType.PER_STEP,
        value=0.30,
        requires_action=True,
        intentional=True,
        state_dependent=True,
    ))

    # Feet contact (0.1) + airtime (1.0) = 1.10
    # Airtime is the DOMINANT term — rewards single-foot contact
    model.add_reward_source(RewardSource(
        name="feet_contact_airtime",
        reward_type=RewardType.PER_STEP,
        value=1.10,  # dominant term
        requires_action=True,
        state_dependent=True,
    ))

    # Base height (0.05) + feet orient (0.05) + feet position (0.05)
    model.add_reward_source(RewardSource(
        name="posture_regularization",
        reward_type=RewardType.PER_STEP,
        value=0.15,
        requires_action=True,
        state_dependent=True,
    ))

    # Penalties: base accel (0.1), arm (0.03), action diff (0.02),
    # torque (0.02) = 0.17 total
    model.add_reward_source(RewardSource(
        name="smoothness_penalties",
        reward_type=RewardType.PER_STEP,
        value=-0.17,
        requires_action=True,
        state_dependent=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
