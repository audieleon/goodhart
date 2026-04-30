"""Pan et al. 2022: Traffic-Bottleneck misweighting (lane changes).

Proxy underweights the lane-change penalty, so the agent switches
lanes excessively near the bottleneck to improve local throughput.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "pan_traffic_bottleneck",
    "source_paper": (
        'Pan, Bhatia & Steinhardt, "The Effects of Reward'
        ' Misspecification: Mapping and Mitigating Misaligned'
        ' Models," ICLR 2022'
    ),
    "paper_url": "https://arxiv.org/abs/2201.03544",
    "source_code_url": "https://github.com/aypan17/reward-misspecification",
    "reward_location": "Table 1 row 2; code: traffic/rewards.py",
    "year": 2022,
    "domain": "driving",
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Traffic-Bottleneck proxy reduces lane-change penalty weight."
        " Agent switches lanes excessively near the bottleneck,"
        " treating lane changes as nearly free."
    ),
    "documented_failure": (
        "Agent exploits underweighted lane-change penalty to"
        " switch lanes aggressively. This produces a phase"
        " transition as model capacity increases."
    ),
    "failure_mechanism": "misweighting",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Restore lane-change penalty weight to true value.",
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "requires_action": (
            "Lane changes require explicit agent action;"
            " desired_vel and forward_progress are passive."
        ),
        "intentional": (
            "desired_vel + forward_progress are the true objective;"
            " lane-change penalty is a safety constraint."
        ),
        "can_loop": "No looping in continuous traffic flow.",
        "value": (
            "desired_vel: per-step velocity tracking."
            " forward_progress: per-step distance covered."
            " lane_change_penalty: eta * |lane_changes|"
            " (eta underweighted in proxy)."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Pan et al. 2022: Traffic-Bottleneck -- Misweighting (lanes)")
    print("=" * 70)

    # True reward: desired_vel + forward_progress - eta*lane_change_penalty
    # Proxy: same but eta (lane change weight) reduced
    # gamma=0.999, max_steps=300, n_actions=2 (accel + lane), continuous

    model = EnvironmentModel(
        name="Traffic Bottleneck - Lane Misweight (Pan et al. 2022)",
        max_steps=300,
        gamma=0.999,
        n_actions=2,
        action_type="continuous",
        death_probability=0.0,
    )

    # desired_vel: deviation from target velocity
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

    # forward_progress: distance covered per step
    model.add_reward_source(RewardSource(
        name="forward_progress",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=False,
        max_occurrences=0,
        state_dependent=True,
        scales_with="distance",
        value_type="proportional",
        value_params={"scale": 1.0},
        intentional=True,
    ))

    # lane_change_penalty -- MISWEIGHTED (eta reduced)
    model.add_reward_source(RewardSource(
        name="lane_change_penalty",
        reward_type=RewardType.ON_EVENT,
        value=-0.01,
        requires_action=True,
        max_occurrences=0,
        state_dependent=True,
        scales_with="lane_changes",
        value_type="proportional",
        value_params={"scale": 0.01},
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
