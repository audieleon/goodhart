"""Risk-aware driving reward (Abouelazm et al. 2025, IV 2025).

Without risk awareness, agent crashes into static obstacles because
accumulated progress penalty from waiting exceeds collision penalty.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "riskaware_driving",
    "source_paper": (
        'Abouelazm, Michel, Gremmelmaier, Joseph, Schorner & Zollner, '
        '"Balancing Progress and Safety: A Novel Risk-Aware Objective '
        'for RL in Autonomous Driving," IV 2025 (IEEE)'
    ),
    "paper_url": "https://arxiv.org/abs/2505.06737",
    "source_code_url": None,
    "reward_location": (
        "Section III: Eq. 1-2 (hierarchical structure, w_Li = beta^(i-1), "
        "beta=0.25). Eq. 3-5 (terminal: success/collision/offroad). "
        "Eq. 6 (traffic rules L0=-1). Eq. 17 (risk L1*). "
        "Eq. 18 (progress L1). Eq. 19 (driving style L2). "
        "Eq. 20 (comfort L3). Table II (all parameters)."
    ),
    "year": 2025,
    "domain": "driving",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Agent was supposed to navigate intersections safely. Without "
        "risk awareness, it crashes into static obstacles because the "
        "accumulated progress penalty from waiting exceeds the sparse "
        "collision penalty. Risk-aware reward reduces collisions 21%."
    ),
    "documented_failure": (
        "Without L1* risk awareness: collision rate 42.5-61.9% across "
        "traffic densities (L0-1 baseline). With full L_complete: "
        "19.6-38.8%. The failure is that progress shaping (L1) makes "
        "waiting costly, and the sparse collision penalty (-1) is "
        "insufficient to deter crashing when cumulative waiting cost "
        "exceeds crash cost. Fig. 1 illustrates: agent prefers crash "
        "over indefinite waiting behind static obstacle."
    ),
    "failure_mechanism": "penalty_dominance",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Add risk-awareness level L1* with ellipsoid risk field (Eq. 7) "
        "that provides dense safety signal before collision occurs. "
        "Reduces collision 21% while maintaining progress."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "hierarchy": (
            "Reward has 5 levels: Terminal (w=50), L0 traffic rules, "
            "L1 progress (w=1), L1* risk (w=beta=0.25), L2 style "
            "(w=beta^2=0.0625), L3 comfort (w=beta^3=0.0156). "
            "Encoding the L0-1 baseline (without risk) to show the "
            "failure mode. L_complete is the fix."
        ),
        "collision_sparse": (
            "R_collision = -1*(0.5 + 0.5*v/v_max). Terminal, sparse. "
            "At low speed (v~0), penalty is only -0.5. This is the "
            "structural trap: dense progress shaping outweighs sparse "
            "collision penalty during extended waiting."
        ),
        "progress_dense": (
            "R_L1 = d_traveled/(v_max*dt). Dense per-step. Active. "
            "This creates the progress pressure that makes waiting "
            "costly — exactly the penalty_dominance pattern."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Encoding the L0-1 BASELINE (without risk awareness)
    # to demonstrate the documented failure mode.
    # From Table II: beta=0.25, v_max=6.0, w_Terminal=50.0

    model = EnvironmentModel(
        name="Risk-Aware Driving L0-1 Baseline (Abouelazm et al. 2025)",
        max_steps=1000,  # intersection scenario
        gamma=0.99,
        n_actions=2,  # (v_f, d_f) = desired velocity + lateral offset
        action_type="discrete",  # DQN with discrete actions
        death_probability=0.01,
    )

    # Terminal: success reward (Eq. 5)
    # R_success = 1.0 if offset < threshold, else 0.5
    # Weighted by w_Terminal = 50.0
    model.add_reward_source(RewardSource(
        name="success_reward",
        reward_type=RewardType.TERMINAL,
        value=50.0,  # w_Terminal * R_success (1.0)
        requires_action=True,
        intentional=True,
        discovery_probability=0.5,
    ))

    # Terminal: collision penalty (Eq. 4)
    # R_collision = -1 * (0.5 + 0.5 * v/v_max)
    # At typical speed v~3 m/s, v_max=6: R = -1*(0.5+0.25) = -0.75
    # Weighted by w_Terminal = 50.0
    model.add_reward_source(RewardSource(
        name="collision_penalty",
        reward_type=RewardType.TERMINAL,
        value=-37.5,  # w_Terminal * R_collision at typical speed
        requires_action=True,
        state_dependent=True,
        scales_with="velocity",
    ))

    # L0: Traffic rule conformance (Eq. 6)
    # R_L0 = -1 per violation (speed limit, red light)
    # Not weighted by hierarchy (added directly)
    model.add_reward_source(RewardSource(
        name="traffic_rule_violation",
        reward_type=RewardType.ON_EVENT,
        value=-1.0,
        requires_action=True,
    ))

    # L1: Progress (Eq. 18)
    # R_L1 = d_traveled / (v_max * dt), normalized to [0,1]
    # Weight w_L1 = beta^0 = 1.0
    model.add_reward_source(RewardSource(
        name="progress",
        reward_type=RewardType.PER_STEP,
        value=1.0,  # max per step (normalized)
        requires_action=True,
        intentional=True,
        state_dependent=True,
        scales_with="velocity",
    ))

    # L2: Driving style (Eq. 19)
    # velocity deviation + lane offset penalties
    # Weight w_L2 = beta^1 = 0.25
    model.add_reward_source(RewardSource(
        name="velocity_deviation",
        reward_type=RewardType.PER_STEP,
        value=-0.125,  # w_L2 * penalty (half of 0.25)
        requires_action=True,
        state_dependent=True,
        scales_with="velocity",
    ))

    model.add_reward_source(RewardSource(
        name="lane_offset",
        reward_type=RewardType.PER_STEP,
        value=-0.125,  # w_L2 * penalty (other half)
        requires_action=True,
        state_dependent=True,
    ))

    # L3: Comfort (Eq. 20)
    # -(1/3)(a/a_max + theta_dot/(v*kappa_max) + j/(a_max/dt))
    # Weight w_L3 = beta^2 = 0.0625
    model.add_reward_source(RewardSource(
        name="comfort_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.0625,  # w_L3 * max comfort penalty
        requires_action=True,
        state_dependent=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
