"""OCTOPUS smart building control (Ding et al. 2023, ACM ToSN).

Reward = -[energy + thermal_comfort + visual_comfort + air_quality].
Holistic control of HVAC, lighting, blinds, windows. 14.26% savings.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "building_octopus",
    "source_paper": (
        'Ding, Cerpa & Du, "Exploring Deep Reinforcement Learning for '
        'Holistic Smart Building Control," ACM Trans. Sensor Networks, 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2301.11510",
    "source_code_url": None,
    "reward_location": (
        "Section IV-D, Eq. 3: R = -[rho1*Norm(E) + rho2*Norm(Tc) "
        "+ rho3*Norm(Vc) + rho4*Norm(Ic)]. Eq. 7: E = Heating + "
        "Cooling + Fan + Lighting power. Eq. 8: Tc = |PMV-P| penalty. "
        "Eq. 9: Vc = illuminance out-of-range penalty. Eq. 10: "
        "Ic = CO2 out-of-range penalty. 4 subsystems, 2.37M actions."
    ),
    "year": 2024,
    "domain": "energy",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Holistic building control: HVAC, lighting, blinds, windows "
        "jointly optimized. Reward minimizes energy while keeping "
        "thermal comfort (PMV), visual comfort (illuminance), and "
        "air quality (CO2) in acceptable ranges. 14.26% energy savings."
    ),
    "documented_failure": (
        "None — well-designed multi-objective reward. Outperforms "
        "rule-based by 14.26% and prior DRL by 8.1%. Lagrangian "
        "multipliers balance energy vs comfort objectives."
    ),
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "high",
    "is_negative_example": True,
    "encoding_rationale": {
        "all_negative_tracking": (
            "All four components are penalties (negative). Minimize "
            "energy and comfort deviations. Same tracking pattern as "
            "nuclear/warfarin. goodhart should fire negative_only as "
            "WARNING (state-dependent, informative gradient)."
        ),
        "lagrangian_weights": (
            "rho1-rho4 are Lagrangian multipliers balancing energy "
            "vs three comfort constraints. Normalized so all "
            "components are on the same scale."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Eq. 3: R = -[rho1*E + rho2*Tc + rho3*Vc + rho4*Ic]
    # 4 subsystems, 15-min control intervals, EnergyPlus

    model = EnvironmentModel(
        name="OCTOPUS Building Control (Ding et al. 2024)",
        max_steps=96,  # 24 hours at 15-min intervals
        gamma=0.99,
        n_actions=4,  # HVAC temp + lighting + blind + window (simplified)
        action_type="discrete",
        n_states=10000,
        death_probability=0.0,
    )

    # Energy consumption penalty (Eq. 7)
    model.add_reward_source(RewardSource(
        name="energy_penalty",
        reward_type=RewardType.PER_STEP,
        value=-1.0,  # rho1 * Norm(E)
        requires_action=True,
        intentional=True,
        state_dependent=True,
        scales_with="energy",
    ))

    # Thermal comfort penalty (Eq. 8)
    model.add_reward_source(RewardSource(
        name="thermal_comfort_penalty",
        reward_type=RewardType.PER_STEP,
        value=-1.0,  # rho2 * Norm(Tc), PMV deviation
        requires_action=True,
        state_dependent=True,
    ))

    # Visual comfort penalty (Eq. 9)
    model.add_reward_source(RewardSource(
        name="visual_comfort_penalty",
        reward_type=RewardType.PER_STEP,
        value=-1.0,  # rho3 * Norm(Vc), illuminance deviation
        requires_action=True,
        state_dependent=True,
    ))

    # Indoor air quality penalty (Eq. 10)
    model.add_reward_source(RewardSource(
        name="air_quality_penalty",
        reward_type=RewardType.PER_STEP,
        value=-1.0,  # rho4 * Norm(Ic), CO2 deviation
        requires_action=True,
        state_dependent=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
