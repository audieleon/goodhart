"""WFCRL wind farm control benchmark (Bizon Monroc et al. 2024, NeurIPS D&B).

Multi-agent: each turbine adjusts yaw to maximize total farm power
while limiting structural fatigue loads. r = power - alpha*load.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "windfarm_wfcrl",
    "source_paper": (
        'Bizon Monroc, Busic, Dubuc & Zhu, "WFCRL: A Multi-Agent '
        'Reinforcement Learning Benchmark for Wind Farm Control," '
        'NeurIPS 2024 Datasets & Benchmarks'
    ),
    "paper_url": "https://arxiv.org/abs/2501.13592",
    "source_code_url": "https://github.com/ifpen/wfcrl-env",
    "reward_location": (
        "Section 3.1: r_k = r_k^P - alpha*r_k^L, alpha=1. "
        "Eq. 4: r_k^P = (1/M)*sum(P_k^i / u_inf^3) (normalized power). "
        "Eq. 5: r_k^L (FLORIS) = turbulence + velocity std proxy. "
        "Eq. 6: r_k^L (FAST.Farm) = bending moments. "
        "Actions: yaw delta in [-5,5] degrees, 10% actuation limit."
    ),
    "year": 2024,
    "domain": "energy",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Multi-agent wind farm control benchmark. Each turbine adjusts "
        "yaw to deflect wake, maximizing total farm power while limiting "
        "structural fatigue loads. Up to 14% power increase via wake "
        "steering. 5 real wind farm layouts, 7-91 turbines."
    ),
    "documented_failure": (
        "None — benchmark with well-designed reward. Power and load "
        "are balanced by alpha. Fig. 2 shows IPPO and MAPPO both "
        "increase power while load penalty stabilizes."
    ),
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "medium",
    "is_negative_example": True,
    "encoding_rationale": {
        "cooperative_marl": (
            "All M turbines share the same reward (cooperative). "
            "Modeled as single-agent with M*action_dim actions. "
            "The MARL structure doesn't affect reward analysis."
        ),
        "power_normalized": (
            "Power is normalized by u_inf^3 (wind speed cubed) "
            "to make reward comparable across wind conditions. "
            "value=1.0 represents normalized max power."
        ),
        "load_as_penalty": (
            "Load penalty is state-dependent: only accumulates "
            "when turbines are actively yawing (actuation causes "
            "structural stress). alpha=1 balances power vs load."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # r_k = r_k^P - alpha * r_k^L, alpha=1
    # Ablaincourt layout: 7 turbines, T=150, FLORIS

    model = EnvironmentModel(
        name="Wind Farm WFCRL (Bizon Monroc et al. 2024)",
        max_steps=150,
        gamma=0.99,
        n_actions=7,  # 1 yaw per turbine, 7 turbines
        action_type="continuous",
        death_probability=0.0,
    )

    # Power production (Eq. 4)
    # r_k^P = (1/M) * sum(P_k^i / u_inf^3)
    model.add_reward_source(RewardSource(
        name="farm_power",
        reward_type=RewardType.PER_STEP,
        value=1.0,  # normalized power production
        requires_action=True,
        intentional=True,
        state_dependent=True,
        scales_with="velocity",
    ))

    # Load penalty (Eq. 5 for FLORIS)
    # r_k^L = turbulence + velocity variation proxy
    # alpha=1, scaled to similar magnitude as power
    model.add_reward_source(RewardSource(
        name="fatigue_load_penalty",
        reward_type=RewardType.PER_STEP,
        value=-1.0,  # alpha=1, similar magnitude to power
        requires_action=True,
        state_dependent=True,
        scales_with="velocity",
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
