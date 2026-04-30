"""Nuclear power plant with Lagrangian constraints (Sun et al. 2024).

Fixed version: chance-constrained PPO adds temperature limit penalties
via trainable Lagrangian multipliers. Achieves 50% power deviation
reduction while respecting safety bounds.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "nuclear_power_plant_constrained",
    "source_paper": (
        'Sun, Khairy, Vilim, Hu & Dave, "A Safe Reinforcement Learning '
        'Algorithm for Supervisory Control of Power Plants," '
        'Knowledge-Based Systems / arXiv:2401.13020, 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2401.13020",
    "source_code_url": None,
    "reward_location": (
        "Section 4.2, Eq. 14: R_hat = R(s,a) - sum(lambda_k * C_k) + const. "
        "Two constraints (K=2): C_in (inlet temp), C_out (outlet temp). "
        "Lambda_k are trainable, updated on slower timescale. "
        "Fig. 3: lambda values increase to ~0.3-0.4 during training."
    ),
    "year": 2024,
    "domain": "energy",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Fixed version of nuclear power plant reward. Adds Lagrangian "
        "constraint penalties for inlet/outlet temperature limits. "
        "Agent learns to follow power demand while respecting safety "
        "bounds. Up to 50% reduction in total power deviation."
    ),
    "documented_failure": (
        "None — this is the well-designed constrained version. "
        "Fig. 4 shows both MLP and LSTM actors follow demand while "
        "keeping temperatures within bounds. Violation rate omega "
        "decreases to near-zero with trained Lagrangian multipliers."
    ),
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "high",
    "is_negative_example": True,
    "encoding_rationale": {
        "pair_entry": (
            "This is the FIXED version of nuclear_power_plant.py. "
            "Dataset includes both unconstrained (failure) and "
            "constrained (fix) to show before/after."
        ),
        "lagrangian_as_penalty": (
            "Constraint costs C_k are binary (0 or 1). In the "
            "Lagrangian formulation, they're weighted by learned "
            "lambda_k. We model them as ON_EVENT penalties with "
            "lambda ~ 0.35 (stabilized value from Fig. 3)."
        ),
        "still_negative": (
            "The reward is still all-negative (power tracking -||p-a||^2 "
            "plus constraint penalties). goodhart may still fire "
            "negative_only_reward. This is a known pattern in tracking "
            "control: the objective is to minimize error, so reward is "
            "always <= 0. The constrained version is correct despite "
            "having no positive signal."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # CONSTRAINED version — the fix
    # Adds Lagrangian temperature constraint penalties

    model = EnvironmentModel(
        name="Nuclear Power Plant Constrained (Sun et al. 2024)",
        max_steps=2250,
        gamma=0.997,
        n_actions=3,
        action_type="continuous",
        n_states=100000,
        death_probability=0.0,
    )

    # Primary reward: power tracking (Eq. 3)
    model.add_reward_source(RewardSource(
        name="power_tracking",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
        requires_action=True,
        intentional=True,
        state_dependent=True,
        scales_with="distance",
        value_type="proportional",
    ))

    # Constraint: inlet temperature (Eq. 7, 14)
    # C_in = indicator(T_in < C_in_lower_bound)
    # Penalty = -lambda_in * C_in, lambda_in ~0.35 (Fig. 3)
    model.add_reward_source(RewardSource(
        name="inlet_temp_constraint",
        reward_type=RewardType.ON_EVENT,
        value=-0.35,  # lambda_in from Fig. 3 stabilized value
        requires_action=True,
        state_dependent=True,
    ))

    # Constraint: outlet temperature (Eq. 7, 14)
    # C_out = indicator(T_out > C_out_upper_bound)
    # Penalty = -lambda_out * C_out, lambda_out ~0.35 (Fig. 3)
    model.add_reward_source(RewardSource(
        name="outlet_temp_constraint",
        reward_type=RewardType.ON_EVENT,
        value=-0.35,  # lambda_out from Fig. 3
        requires_action=True,
        state_dependent=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
