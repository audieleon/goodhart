"""Krakovna entry: Wall sensor stack -- tricking the sensor."""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "krakovna_wall_sensor_stack",
    "source_paper": (
        'Paine et al. 2019, "Making Efficient Use of Demonstrations'
        ' to Solve Hard Exploration Problems," R2D3'
    ),
    "paper_url": "https://arxiv.org/abs/1909.01387",
    "year": 2019,
    "domain": "manipulation",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-29",
    "brief_summary": (
        "Agent was supposed to stack blocks to press a wall sensor."
        " Instead it discovered an environment bug that activates"
        " the sensor directly, bypassing stacking."
    ),
    "documented_failure": (
        "Agent tricks the wall sensor without stacking blocks."
        " The reward (binary sensor activation) correctly specifies"
        " the goal, but the environment provides an unintended"
        " shortcut."
    ),
    "failure_mechanism": "environment_bug",
    "detection_type": "dynamic",
    "discovery_stage": "during_training",
    "fix_known": (
        "Fix the environment bug that allows direct sensor"
        " activation. The reward function itself is correct."
    ),
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "source_quality": (
            "Paper is known (Paine et al 2019, R2D3, arxiv:1909.01387)"
            " but reward details not read in depth. Encoding derived"
            " from Krakovna catalog description."
        ),
        "sensor_activation": (
            "Binary terminal reward. The reward is correctly specified"
            " -- the exploit is in the environment dynamics, not the"
            " reward function."
        ),
        "structural_limitation": (
            "This is an environment_bug, not a reward misspecification."
            " Structural reward analysis cannot detect this failure."
        ),
    },
    "paper_url": "https://arxiv.org/abs/1909.01387",
    "reward_location": "Paine et al. 2019 (R2D3): sparse reward for sensor activation. Sensor shortcut exploit.",
    "verification_date": "2026-04-30",
    "discovery_stage": "during_training",
    "source_code_url": None,
    "fix_known": None,
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Wall sensor stack (Le Paine et al 2019)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Wall Sensor Stack (Le Paine et al 2019)",
        max_steps=500,
        gamma=0.99,
        n_actions=6,
        action_type="continuous",
    )

    # Binary sensor activation — the intended goal.
    model.add_reward_source(RewardSource(
        name="sensor_activation",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        intentional=True,
        discovery_probability=0.05,
    ))

    # Distance shaping toward sensor position
    model.add_reward_source(RewardSource(
        name="distance_to_sensor",
        reward_type=RewardType.SHAPING,
        value=0.1,
        scales_with="distance",
        state_dependent=True,
        requires_action=True,
    ))

    engine.print_report(model)

    print()
    print("Advisory: environment_bug — the sensor can be activated without")
    print("stacking blocks, bypassing the intended solution path. The")
    print("reward function (binary sensor activation) correctly specifies")
    print("the goal. The problem is that the environment provides an")
    print("unintended shortcut to achieve that goal, which structural")
    print("reward analysis cannot detect.")


if __name__ == "__main__":
    run_example()
