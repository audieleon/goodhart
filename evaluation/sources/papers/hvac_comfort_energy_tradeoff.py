"""HVAC comfort-energy tradeoff — weight sensitivity (Zhang et al. 2019).

Standard HVAC reward: r = -w_e * energy - w_c * comfort_violation.
Small changes in w_e/w_c ratio cause the agent to flip between
"freeze the occupants to save energy" and "blast heat regardless
of cost." The tradeoff weight is not robust.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "hvac_comfort_energy",
    "source_paper": (
        'Zhang & Lam, "Practical Implementation and Evaluation of '
        'Deep Reinforcement Learning Control for a Radiant Heating '
        'System," ACM BuildSys 2019'
    ),
    "paper_url": "https://dl.acm.org/doi/10.1145/3360322.3360861",
    "source_code_url": None,
    "reward_location": (
        "Section 3.2: r_t = -alpha * E_t - (1-alpha) * max(0, "
        "|T_t - T_set| - delta). E_t = energy consumption, T_t = "
        "zone temperature, T_set = setpoint, delta = comfort band. "
        "alpha in [0,1] trades off energy vs comfort."
    ),
    "year": 2019,
    "domain": "energy",
    "encoding_basis": "primary_source",
    "verification_date": "2026-05-01",
    "brief_summary": (
        "HVAC agent minimizes weighted sum of energy cost and "
        "comfort violation. The weight alpha is extremely sensitive: "
        "small changes flip behavior between energy-wasteful and "
        "occupant-discomforting extremes."
    ),
    "documented_failure": (
        "Section 5: the comfort-energy tradeoff is highly sensitive "
        "to alpha. At alpha too high, agent lets temperature drift "
        "outside comfort bounds. At alpha too low, agent runs heating "
        "continuously. Finding the right balance requires extensive "
        "trial-and-error per building."
    ),
    "failure_mechanism": "ontological_proxy",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Use constrained RL with comfort as a hard constraint "
        "rather than a weighted penalty. Or use multi-objective RL "
        "to find the Pareto frontier."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "negative_only": (
            "Both terms are negative: -energy and -comfort_violation. "
            "negative_only_reward should fire."
        ),
        "weight_sensitivity": (
            "The alpha parameter creates reward_dominance_imbalance "
            "at extreme values. Energy and comfort have different "
            "scales, making the weighted sum fragile."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    model = EnvironmentModel(
        name="HVAC Comfort-Energy Tradeoff (Zhang 2019)",
        max_steps=288,
        gamma=0.99,
        n_states=10000,
        n_actions=5,
        action_type="discrete",
        death_probability=0.0,
    )

    # -alpha * energy (always negative)
    model.add_reward_source(RewardSource(
        name="energy_cost",
        reward_type=RewardType.PER_STEP,
        value=-0.5,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # -(1-alpha) * comfort violation (negative when uncomfortable)
    model.add_reward_source(RewardSource(
        name="comfort_violation",
        reward_type=RewardType.PER_STEP,
        value=-0.5,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
