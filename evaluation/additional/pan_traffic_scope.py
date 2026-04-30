"""Pan et al. 2022: Traffic-Merge scope (local velocity only).

Proxy monitors velocity only in the merge zone instead of across all
vehicles, so the agent freely externalizes congestion downstream.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "pan_traffic_scope",
    "source_paper": (
        'Pan, Bhatia & Steinhardt, "The Effects of Reward'
        ' Misspecification: Mapping and Mitigating Misaligned'
        ' Models," ICLR 2022'
    ),
    "paper_url": "https://arxiv.org/abs/2201.03544",
    "source_code_url": "https://github.com/aypan17/reward-misspecification",
    "reward_location": "Table 1 row 4; code: traffic/rewards.py",
    "year": 2022,
    "domain": "driving",
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Traffic-Merge proxy replaces desired_velocity (all vehicles)"
        " with local_desired_velocity (merge zone only). Agent"
        " optimizes merge-zone speed while ignoring downstream"
        " congestion. Exhibits a phase transition."
    ),
    "documented_failure": (
        "Agent pushes congestion outside the monitored merge zone."
        " Local velocity stays high but system-wide commute time"
        " increases. Shows phase transition with model capacity."
    ),
    "failure_mechanism": "scope",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Monitor velocity across all vehicles, not just the"
        " merge zone."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "requires_action": (
            "local_desired_velocity is state-dependent;"
            " acceleration penalty requires action."
        ),
        "intentional": (
            "local_desired_velocity is treated as the objective"
            " but covers the wrong spatial scope."
        ),
        "can_loop": "No looping in continuous traffic flow.",
        "value": (
            "local_desired_velocity: deviation from target speed"
            " for merge-zone vehicles only."
            " True reward uses desired_velocity over all vehicles."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Pan et al. 2022: Traffic-Merge -- Scope (local velocity)")
    print("=" * 70)

    # True reward: desired_vel(all vehicles) + 0.1*accel + 0.1*headway
    # Proxy: local_desired_velocity(merge zone only) replaces desired_vel
    # gamma=0.999, max_steps=300, n_actions=1, continuous

    model = EnvironmentModel(
        name="Traffic Merge - Local Scope (Pan et al. 2022)",
        max_steps=300,
        gamma=0.999,
        n_actions=1,
        action_type="continuous",
        death_probability=0.0,
    )

    # local_desired_velocity -- SCOPED proxy
    # Only measures velocity in the merge zone; true reward
    # measures desired_vel across all vehicles on the network.
    model.add_reward_source(RewardSource(
        name="local_desired_velocity",
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
