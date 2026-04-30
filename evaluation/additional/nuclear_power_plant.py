"""Nuclear power plant load-following (Sun et al. 2024, Argonne).

Unconstrained RL violates temperature safety limits. Chance-constrained
PPO with Lagrangian multipliers enforces inlet/outlet temperature bounds.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "nuclear_power_plant",
    "source_paper": (
        'Sun, Khairy, Vilim, Hu & Dave, "A Safe Reinforcement Learning '
        'Algorithm for Supervisory Control of Power Plants," '
        'Knowledge-Based Systems / arXiv:2401.13020, 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2401.13020",
    "source_code_url": None,
    "reward_location": (
        "Section 4.1, Eq. 3: R(s,a) = -||p_t - a_t||^2 (L2 power tracking). "
        "Eq. 7: C_k(s) = indicator(constraint k violated). K=2 constraints: "
        "core inlet temp >= C_in, core outlet temp <= C_out. "
        "Eq. 14: penalized reward R_hat = R - sum(lambda_k * C_k) + const. "
        "Eq. 20: evaluation metric R_bar = R - sum(C_k). "
        "Section 3.3: T=2250 steps (subsampled from 11250). "
        "320 MW pebble-bed fluoride-cooled high-temperature reactor."
    ),
    "year": 2024,
    "domain": "energy",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "RL agent controls a 320MW nuclear reactor during load-following "
        "transients. Without safety constraints, the agent violates "
        "core inlet/outlet temperature limits that could damage reactor "
        "components. Chance-constrained PPO with Lagrangian multipliers "
        "enforces temperature bounds while following power demand."
    ),
    "documented_failure": (
        "Unconstrained RL violates physics-hard temperature constraints. "
        "Section 4.1: 'standard RL methods did not consider the "
        "operational constraints of states.' Fig. 4 shows: simply "
        "following power demand causes inlet/outlet temperatures to "
        "exceed safety bounds. The Lagrangian multipliers lambda_in "
        "and lambda_out increase sharply during early training as the "
        "agent learns to respect constraints (Fig. 3)."
    ),
    "failure_mechanism": "missing_constraint",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Chance-constrained PPO with Lagrangian relaxation (Eq. 11-14). "
        "Trainable lambda_k multipliers enforce temperature constraints "
        "on a slower timescale than policy updates. Achieves up to 50% "
        "reduction in total power deviation while satisfying constraints."
    ),
    "compute_cost_class": "high",
    "is_negative_example": False,
    "encoding_rationale": {
        "power_tracking": (
            "R = -||p_t - a_t||^2. PER_STEP, requires_action=True "
            "(agent controls reactivity + pumps). intentional=True "
            "(load-following IS the objective). Negative-valued: "
            "perfect tracking gives 0, any deviation is penalized."
        ),
        "temperature_constraints": (
            "C_k are binary indicators: 1 if violated, 0 if safe. "
            "Modeled as ON_EVENT with value=-1 (penalty per violation). "
            "In the constrained version, these are weighted by learned "
            "Lagrangian multipliers. In the UNconstrained version "
            "(the failure), these penalties are absent entirely — "
            "the agent has no incentive to respect temperature limits."
        ),
        "encoding_unconstrained": (
            "We encode the UNCONSTRAINED version (the failure mode) "
            "to show goodhart detecting the missing constraint. "
            "advisory_missing_constraint should fire: all-positive-ish "
            "control with no safety penalties."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # UNCONSTRAINED version — the failure mode
    # Agent has power-tracking reward only, no temperature constraints
    # 320 MW pebble-bed fluoride-cooled reactor
    # 3 actions: external reactivity, primary pump, secondary pump
    # T=2250 steps (subsampled), gamma ~0.997

    model = EnvironmentModel(
        name="Nuclear Power Plant Unconstrained (Sun et al. 2024)",
        max_steps=2250,
        gamma=0.997,
        n_actions=3,  # reactivity, primary pump head, secondary pump head
        action_type="continuous",
        n_states=100000,  # high-dimensional physics state
        death_probability=0.0,  # no termination on constraint violation
    )

    # Primary reward: power tracking (Eq. 3)
    # R(s,a) = -||p_t - a_t||^2
    # Negative of squared error between demand and supply
    model.add_reward_source(RewardSource(
        name="power_tracking",
        reward_type=RewardType.PER_STEP,
        value=-1.0,  # normalized; actual is -||p-a||^2
        requires_action=True,
        intentional=True,
        state_dependent=True,
        scales_with="distance",
        value_type="proportional",
    ))

    # NOTE: In the unconstrained version, there are NO temperature
    # constraint penalties. The agent optimizes power tracking only,
    # and violates inlet/outlet temperature limits freely.
    # The fix adds: -lambda_in * C_in(s) - lambda_out * C_out(s)
    # where C_k = indicator(constraint k violated)

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
