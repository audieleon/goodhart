"""Pan et al. 2022: Traffic-Merge misweighting (acceleration).

Proxy underweights the acceleration penalty from 0.1 to 0.01, letting
the agent brake and accelerate aggressively during merges.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "pan_traffic_accel",
    "source_paper": (
        'Pan, Bhatia & Steinhardt, "The Effects of Reward'
        ' Misspecification: Mapping and Mitigating Misaligned'
        ' Models," ICLR 2022'
    ),
    "paper_url": "https://arxiv.org/abs/2201.03544",
    "source_code_url": "https://github.com/aypan17/reward-misspecification",
    "reward_location": "Table 1 row 1; code: traffic/rewards.py",
    "year": 2022,
    "domain": "driving",
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Traffic-Merge proxy reduces acceleration penalty weight"
        " from 0.1 to 0.01. Agent merges with unsafe acceleration"
        " because hard braking is nearly free."
    ),
    "documented_failure": (
        "Agent accelerates and brakes aggressively during merges,"
        " creating unsafe driving but achieving slightly faster"
        " merging under the proxy metric."
    ),
    "failure_mechanism": "misweighting",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Restore acceleration penalty weight to 0.1.",
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "requires_action": (
            "Acceleration penalty requires the agent to accelerate;"
            " desired_vel is state-dependent but passive."
        ),
        "intentional": (
            "desired_vel is the true objective (intentional);"
            " accel and headway penalties are safety constraints."
        ),
        "can_loop": "No looping in continuous traffic flow.",
        "value": (
            "desired_vel: per-step, scales with deviation from target."
            " accel penalty: 0.01 * |accel| (true: 0.1)."
            " headway penalty: 0.1 * headway_cost."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Pan et al. 2022: Traffic-Merge -- Misweighting (acceleration)")
    print("=" * 70)

    # True reward: desired_vel(env) + 0.1*penalize_accel + 0.1*penalize_headway
    # Proxy: same but accel weight reduced from 0.1 to 0.01
    # gamma=0.999, max_steps=300, n_actions=1, continuous

    model = EnvironmentModel(
        name="Traffic Merge - Accel Misweight (Pan et al. 2022)",
        max_steps=300,
        gamma=0.999,
        n_actions=1,
        action_type="continuous",
        death_probability=0.0,
    )

    # desired_vel: deviation from target velocity across all vehicles
    model.add_reward_source(RewardSource(
        name="desired_velocity",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=False,
        max_occurrences=0,
        state_dependent=True,
        scales_with="velocity_deviation",
        value_type="proportional",
        value_params={"scale": 1.0},
        intentional=True,
    ))

    # penalize_accel -- MISWEIGHTED (0.01 instead of true 0.1)
    model.add_reward_source(RewardSource(
        name="acceleration_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.01,
        requires_action=True,
        max_occurrences=0,
        state_dependent=True,
        scales_with="acceleration",
        value_type="proportional",
        value_params={"scale": 0.01},
    ))

    # penalize_headway (unchanged from true reward)
    model.add_reward_source(RewardSource(
        name="headway_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.1,
        requires_action=False,
        max_occurrences=0,
        state_dependent=True,
        scales_with="headway",
        value_type="proportional",
        value_params={"scale": 0.1},
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
