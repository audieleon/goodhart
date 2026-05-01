"""Google Circuit Training proxy gap (Mirhoseini 2021 / Chiang 2023).

RL optimizes proxy cost (HPWL + density + congestion) for chip macro
placement. Chiang et al. showed proxy has near-zero correlation with
true metrics: area (r=0.00), power (r=0.05), timing TNS (r=0.05).
The proxy doesn't even compute timing analysis.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "mirhoseini_proxy_gap",
    "source_paper": (
        'Chiang, "The False Dawn: Reevaluating Google\'s RL for '
        'Chip Macro Placement," arXiv 2023 / CACM 2024. Original: '
        'Mirhoseini et al., "A Graph Placement Methodology for Fast '
        'Chip Design," Nature 2021'
    ),
    "paper_url": "https://arxiv.org/abs/2306.09633",
    "source_code_url": "https://github.com/google-research/circuit_training",
    "reward_location": (
        "Chiang Section 3 + Table 2: proxy cost = weighted blend of "
        "HPWL (half-perimeter wirelength), component density, and "
        "routing congestion. Proxy-true rank correlations: area=0.00, "
        "routed wirelength=0.28, power=0.05, WNS timing=0.20, TNS "
        "timing=0.05. Proxy does NOT perform circuit timing analysis."
    ),
    "year": 2023,
    "domain": "chip_design",
    "encoding_basis": "primary_source",
    "verification_date": "2026-05-01",
    "brief_summary": (
        "RL chip placement optimizes proxy cost (HPWL + density + "
        "congestion) but proxy has near-zero correlation with true "
        "metrics: area (0.00), power (0.05), timing (0.05). Nature "
        "added editor's note questioning the performance claims."
    ),
    "documented_failure": (
        "Chiang et al. Table 2: rank correlations between proxy cost "
        "and true chip metrics are 0.00-0.28. The RL agent optimizes "
        "a proxy that doesn't predict real performance. Simulated "
        "annealing and human experts produce better placements. "
        "Nature added editor's note to original paper."
    ),
    "failure_mechanism": "ontological_proxy",
    "detection_type": "structural",
    "discovery_stage": "post_deployment",
    "fix_known": (
        "Include timing analysis in the cost function. Use "
        "post-placement evaluation metrics as the true reward, "
        "not pre-routing estimates."
    ),
    "compute_cost_class": "extreme",
    "is_negative_example": False,
    "encoding_rationale": {
        "proxy_hackability": (
            "The proxy cost (HPWL + density + congestion) is "
            "structurally different from true metrics (area, power, "
            "timing). proxy_reward_hackability should fire."
        ),
        "missing_timing": (
            "The proxy doesn't compute timing at all. TNS and WNS "
            "are critical chip metrics with r=0.05 and r=0.20 "
            "correlation to the proxy."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Mirhoseini 2021: Circuit Training
    # RL places macro blocks on chip canvas
    model = EnvironmentModel(
        name="Circuit Training Proxy Gap (Mirhoseini/Chiang)",
        max_steps=100,
        gamma=1.0,
        n_states=100000,
        n_actions=1000,
        action_type="discrete",
        death_probability=0.0,
    )

    # Proxy cost: HPWL component (wirelength estimate)
    model.add_reward_source(RewardSource(
        name="hpwl_proxy",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Proxy cost: density component
    model.add_reward_source(RewardSource(
        name="density_proxy",
        reward_type=RewardType.PER_STEP,
        value=-0.5,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Proxy cost: congestion component
    model.add_reward_source(RewardSource(
        name="congestion_proxy",
        reward_type=RewardType.PER_STEP,
        value=-0.5,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
