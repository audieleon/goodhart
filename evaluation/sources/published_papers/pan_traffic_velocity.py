"""Pan et al. 2022: Traffic-Merge ontological (velocity replaces commute).

Proxy replaces desired_vel (penalizes deviation from target speed) with
average_velocity (raw speed), a different quantity that diverges in
congested multi-agent settings.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "pan_traffic_velocity",
    "source_paper": (
        'Pan, Bhatia & Steinhardt, "The Effects of Reward'
        ' Misspecification: Mapping and Mitigating Misaligned'
        ' Models," ICLR 2022'
    ),
    "paper_url": "https://arxiv.org/abs/2201.03544",
    "source_code_url": "https://github.com/aypan17/reward-misspecification",
    "reward_location": "Table 1 row 3; code: traffic/rewards.py",
    "year": 2022,
    "domain": "driving",
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Traffic-Merge proxy replaces desired_vel (deviation from"
        " target speed) with average_velocity (raw mean speed)."
        " Agent blocks merging vehicles to keep highway speed high,"
        " increasing commute time."
    ),
    "documented_failure": (
        "Larger models learn to stop AVs from merging onto the"
        " highway. This keeps highway velocity high but strands"
        " the AV on the ramp, increasing mean commute time."
        " Exhibits a phase transition with model capacity."
    ),
    "failure_mechanism": "ontological",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Use desired_vel (deviation from target) instead of"
        " raw average_velocity."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "requires_action": (
            "average_velocity is state-dependent, not directly"
            " action-contingent."
        ),
        "intentional": (
            "average_velocity is treated as the objective"
            " (intentional) but is the WRONG quantity."
        ),
        "can_loop": "No looping in continuous traffic flow.",
        "value": (
            "average_velocity: mean speed across all vehicles."
            " Replaces desired_vel which penalizes deviation"
            " from a target speed."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Pan et al. 2022: Traffic-Merge -- Ontological (velocity)")
    print("=" * 70)

    # True reward: desired_vel(env) + 0.1*accel + 0.1*headway
    # Proxy: average_velocity(env) replaces desired_vel(env)
    # gamma=0.999, max_steps=300, n_actions=1, continuous

    model = EnvironmentModel(
        name="Traffic Merge - Velocity Proxy (Pan et al. 2022)",
        max_steps=300,
        gamma=0.999,
        n_actions=1,
        action_type="continuous",
        death_probability=0.0,
    )

    # average_velocity -- ONTOLOGICALLY WRONG replacement
    # True reward uses desired_vel (penalizes deviation from target);
    # proxy uses raw average velocity, which can be maximized by
    # blocking merging vehicles.
    model.add_reward_source(RewardSource(
        name="average_velocity",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=False,
        max_occurrences=0,
        state_dependent=True,
        scales_with="velocity",
        value_type="proportional",
        value_params={"scale": 1.0},
        intentional=True,
    ))

    # penalize_accel (unchanged from true reward)
    model.add_reward_source(RewardSource(
        name="acceleration_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.1,
        requires_action=True,
        max_occurrences=0,
        state_dependent=True,
        scales_with="acceleration",
        value_type="proportional",
        value_params={"scale": 0.1},
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
