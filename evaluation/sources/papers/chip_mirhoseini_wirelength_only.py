"""Mirhoseini et al. 2021 — wirelength-only placement (Nature, pre-fix).

Before the proxy gap was identified, the original Circuit Training
reward focused heavily on HPWL (half-perimeter wirelength) with
lighter density and congestion terms. This encoding shows the
wirelength-dominated version that ignores timing entirely.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "mirhoseini_wirelength_only",
    "source_paper": (
        'Mirhoseini, Goldie et al., "A Graph Placement Methodology '
        'for Fast Chip Design," Nature 2021 (Google Brain)'
    ),
    "paper_url": "https://www.nature.com/articles/s41586-021-03544-w",
    "source_code_url": "https://github.com/google-research/circuit_training",
    "reward_location": (
        "Methods: reward = -wirelength_cost - lambda_c * congestion_cost "
        "- lambda_d * density_cost. Wirelength = HPWL (half-perimeter). "
        "The proxy cost function does not include timing analysis "
        "(TNS/WNS), area, or power. Chiang 2023 later showed this "
        "proxy has near-zero correlation with real chip metrics."
    ),
    "year": 2021,
    "domain": "chip_design",
    "encoding_basis": "primary_source",
    "verification_date": "2026-05-01",
    "brief_summary": (
        "Chip macro placement RL optimizes wirelength + congestion + "
        "density but entirely omits timing, area, and power from the "
        "reward. The agent finds placements with low wirelength that "
        "fail timing closure — requiring expensive manual fixes."
    ),
    "documented_failure": (
        "Nature editor's note added: 'performance claims have been "
        "called into question.' Chiang 2023: proxy-true correlations "
        "of 0.00 (area), 0.05 (power), 0.05 (TNS), 0.20 (WNS). "
        "RL placements lag human experts and simulated annealing "
        "on true metrics."
    ),
    "failure_mechanism": "ontological_proxy",
    "detection_type": "structural",
    "discovery_stage": "post_deployment",
    "fix_known": (
        "Include timing analysis in the placement reward. Use "
        "post-routing metrics rather than pre-routing estimates."
    ),
    "compute_cost_class": "extreme",
    "is_negative_example": False,
    "encoding_rationale": {
        "proxy_without_timing": (
            "Wirelength is a proxy for routing quality. Timing, "
            "area, and power are the true objectives. The proxy "
            "misses the most critical chip design constraint."
        ),
        "negative_only": (
            "All three cost terms are negative. The agent minimizes "
            "cost but has no positive signal for good placement."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    model = EnvironmentModel(
        name="Circuit Training Wirelength-Only (Mirhoseini 2021)",
        max_steps=100,
        gamma=1.0,
        n_states=100000,
        n_actions=1000,
        action_type="discrete",
        death_probability=0.0,
    )

    # Wirelength cost (HPWL, dominant term)
    model.add_reward_source(RewardSource(
        name="wirelength_cost",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Congestion cost (weighted)
    model.add_reward_source(RewardSource(
        name="congestion_cost",
        reward_type=RewardType.PER_STEP,
        value=-0.1,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Density cost (weighted)
    model.add_reward_source(RewardSource(
        name="density_cost",
        reward_type=RewardType.PER_STEP,
        value=-0.1,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
