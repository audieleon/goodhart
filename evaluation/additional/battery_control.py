"""Battery optimal control with degradation (Kwon & Zhu 2022, IEEE Smart Grid).

Reward = -(energy_cost + FR_penalty + degradation_cost). Battery
frequency regulation with cycle-based aging model via rainflow.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "battery_control",
    "source_paper": (
        'Kwon & Zhu, "Reinforcement Learning Based Optimal Battery '
        'Control Under Cycle-based Degradation Cost," '
        'IEEE Trans. Smart Grid, 2022'
    ),
    "paper_url": "https://arxiv.org/abs/2108.02374",
    "source_code_url": None,
    "reward_location": (
        "Section IV, Eq. 14: r_t = -h_e - h_f - h_d. "
        "Eq. 6: h_e = p_t*b_t (energy cost). "
        "Eq. 7: h_f = delta*|f_t - b_t| (FR penalty). "
        "Eq. 12: h_d = degradation via rainflow algorithm. "
        "gamma=1, T=8640 steps (7 days at 5-min), 11 actions, "
        "200kWh battery, alpha_d=4.5e-3, beta=1.3."
    ),
    "year": 2022,
    "domain": "energy",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "RL controls battery charging/discharging for grid frequency "
        "regulation while accounting for cycle-based degradation cost. "
        "Three-component negative reward: energy cost, FR penalty, and "
        "battery aging. Novel rainflow-based degradation model."
    ),
    "documented_failure": (
        "Documented failure of LINEAR degradation approximation "
        "(Eq. 10): linearized model misrepresents cycle costs, leading "
        "to suboptimal deep cycling. Cycle-based model (Eq. 12) "
        "correctly penalizes deep cycles, improving battery lifetime."
    ),
    "failure_mechanism": "proxy_reward",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Replace linearized degradation (Eq. 10) with cycle-based "
        "model (Eq. 12) using rainflow algorithm switching points. "
        "Guaranteed equivalent to true cycle-based cost (Proposition 1)."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "three_costs": (
            "Energy (p*b), FR penalty (delta*|f-b|), degradation "
            "(exponential cycle cost). All negative — minimize total. "
            "goodhart should fire negative_only as WARNING (tracking)."
        ),
        "degradation_novel": (
            "The key contribution is the degradation model. Linear "
            "approximation (prior work) is a proxy that diverges from "
            "true cycle cost. This IS a documented reward misspecification "
            "that goodhart could flag as proxy_reward_hackability."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Eq. 14: r = -h_e - h_f - h_d
    # 200kWh battery, 11 discrete actions, gamma=1

    model = EnvironmentModel(
        name="Battery Optimal Control (Kwon & Zhu 2022)",
        max_steps=8640,  # 7 days at 5-min intervals
        gamma=1.0,  # undiscounted
        n_actions=11,  # [-1, -0.8, ..., 0.8, 1]
        action_type="discrete",
        death_probability=0.0,
    )

    # Energy cost: h_e = p_t * b_t (price * power)
    model.add_reward_source(RewardSource(
        name="energy_cost",
        reward_type=RewardType.PER_STEP,
        value=-0.5,  # average normalized cost
        requires_action=True,
        intentional=True,
        state_dependent=True,
        scales_with="price",
    ))

    # FR penalty: h_f = delta * |f_t - b_t|
    model.add_reward_source(RewardSource(
        name="fr_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.3,  # delta * avg deviation
        requires_action=True,
        state_dependent=True,
    ))

    # Degradation cost: h_d (cycle-based, Eq. 12)
    model.add_reward_source(RewardSource(
        name="degradation_cost",
        reward_type=RewardType.PER_STEP,
        value=-0.1,  # exponential cycle cost, normalized
        requires_action=True,
        state_dependent=True,
        value_type="exponential",
        value_params={"alpha_d": 0.0045, "beta": 1.3},
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
