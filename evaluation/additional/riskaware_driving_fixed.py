"""Risk-aware driving L_complete (Abouelazm et al. 2025, IV 2025).

The fixed version with risk awareness. Adds dense safety signal
via ellipsoid risk field before collision occurs. Reduces
collisions 21% vs the L0-1 baseline.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "riskaware_driving_fixed",
    "source_paper": (
        'Abouelazm, Michel, Gremmelmaier, Joseph, Schorner & Zollner, '
        '"Balancing Progress and Safety: A Novel Risk-Aware Objective '
        'for RL in Autonomous Driving," IV 2025 (IEEE)'
    ),
    "paper_url": "https://arxiv.org/abs/2505.06737",
    "source_code_url": None,
    "reward_location": (
        "Section III: L_complete = L0 + L1 + L1* + L2 + L3. "
        "L1* risk awareness: Eq. 7 (ellipsoid), Eq. 17 (combined). "
        "Table II: w_geom=0.5, w_dyn=0.5 for L1*. "
        "Table I results: L_complete collision 19.6-38.8% vs "
        "L0-1 baseline 42.5-61.9%."
    ),
    "year": 2025,
    "domain": "driving",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Fixed version of the risk-aware driving reward. Adds dense "
        "risk penalty via ellipsoid field (Eq. 7) that penalizes "
        "proximity to other vehicles BEFORE collision. Agent learns "
        "to maintain safe clearance rather than gambling on sparse "
        "collision penalty."
    ),
    "documented_failure": (
        "None in practice — L_complete achieves highest success rate "
        "(73.1% at density 0.5), lowest collision rate (19.6%), and "
        "highest cumulative reward (0.78). goodhart fires idle_exploit "
        "because it assumes worst-case for all state-dependent penalties "
        "(risk penalties near all vehicles simultaneously). In practice, "
        "the risk penalty is low when driving safely in open road. "
        "This is a known limitation: goodhart's worst-case model "
        "overapproximates state-dependent penalty accumulation."
    ),
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "medium",
    "is_negative_example": True,
    "encoding_rationale": {
        "pair_entry": (
            "This is the FIXED version of riskaware_driving.py. "
            "The dataset includes both the broken baseline (L0-1) "
            "and the fix (L_complete) to demonstrate the before/after "
            "pattern. goodhart should pass this clean or with minor "
            "warnings only."
        ),
        "risk_as_shaping": (
            "L1* risk penalty is per-step, state-dependent, "
            "requires_action=True (agent must act to maintain safe "
            "clearance). NOT passive — the penalty depends on the "
            "agent's position relative to other vehicles, which "
            "requires active driving decisions."
        ),
        "hierarchy_weights": (
            "From Table II: L1 progress w=1.0 (beta^0), "
            "L1* risk w=0.25 (beta^1, split w_geom=0.5 w_dyn=0.5), "
            "L2 style w=0.0625 (beta^2), L3 comfort w=0.0156 (beta^3). "
            "Risk is weighted equally to progress — the key design "
            "decision that makes this version work."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # L_complete: all levels including risk awareness
    # From Table II: beta=0.25, v_max=6.0, w_Terminal=50.0

    model = EnvironmentModel(
        name="Risk-Aware Driving L_complete (Abouelazm et al. 2025)",
        max_steps=1000,
        gamma=0.99,
        n_actions=2,
        action_type="discrete",
        death_probability=0.01,
    )

    # Terminal: success (Eq. 5), weighted by w_Terminal=50
    model.add_reward_source(RewardSource(
        name="success_reward",
        reward_type=RewardType.TERMINAL,
        value=50.0,
        requires_action=True,
        intentional=True,
        discovery_probability=0.5,
    ))

    # Terminal: collision (Eq. 4), weighted by w_Terminal=50
    model.add_reward_source(RewardSource(
        name="collision_penalty",
        reward_type=RewardType.TERMINAL,
        value=-37.5,
        requires_action=True,
        state_dependent=True,
        scales_with="velocity",
    ))

    # L0: Traffic rule violation (Eq. 6)
    model.add_reward_source(RewardSource(
        name="traffic_rule_violation",
        reward_type=RewardType.ON_EVENT,
        value=-1.0,
        requires_action=True,
    ))

    # L1: Progress (Eq. 18), w_L1 = 1.0
    model.add_reward_source(RewardSource(
        name="progress",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        requires_action=True,
        intentional=True,
        state_dependent=True,
        scales_with="velocity",
    ))

    # L1*: Risk awareness (Eq. 17) — THE FIX
    # R_L1* = -w_geom * P_risk^geom - w_dyn * P_risk^dyn
    # Weight w_L1* = beta = 0.25, split w_geom=0.5 w_dyn=0.5
    # Dense per-step penalty based on proximity to other vehicles
    # Uses ellipsoid risk field (Eq. 7) for geometric risk
    # and TTC-based log penalty (Eq. 16) for dynamic risk
    model.add_reward_source(RewardSource(
        name="geometric_risk_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.125,  # w_L1* * w_geom = 0.25 * 0.5
        requires_action=True,
        state_dependent=True,
        scales_with="distance",
    ))

    model.add_reward_source(RewardSource(
        name="dynamic_risk_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.125,  # w_L1* * w_dyn = 0.25 * 0.5
        requires_action=True,
        state_dependent=True,
        scales_with="velocity",
    ))

    # L2: Driving style (Eq. 19), w_L2 = 0.0625
    model.add_reward_source(RewardSource(
        name="velocity_deviation",
        reward_type=RewardType.PER_STEP,
        value=-0.03125,
        requires_action=True,
        state_dependent=True,
        scales_with="velocity",
    ))

    model.add_reward_source(RewardSource(
        name="lane_offset",
        reward_type=RewardType.PER_STEP,
        value=-0.03125,
        requires_action=True,
        state_dependent=True,
    ))

    # L3: Comfort (Eq. 20), w_L3 = 0.0156
    model.add_reward_source(RewardSource(
        name="comfort_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.0156,
        requires_action=True,
        state_dependent=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
