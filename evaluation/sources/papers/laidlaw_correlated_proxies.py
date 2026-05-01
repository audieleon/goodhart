"""Laidlaw et al. 2024 — KL penalty fails to prevent reward hacking (NeurIPS).

Shows standard KL penalty in RLHF is insufficient to prevent reward
hacking. Proxy-true reward correlation degrades under optimization
pressure. Proposes chi-squared divergence as alternative.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "laidlaw_correlated_proxies",
    "source_paper": (
        'Laidlaw, Singhal, Dragan, "Correlated Proxies: A New '
        'Definition and Improved Mitigation for Reward Hacking," '
        'NeurIPS 2024 (UC Berkeley)'
    ),
    "paper_url": "https://arxiv.org/abs/2403.03185",
    "source_code_url": None,
    "reward_location": (
        "Section 3: proxy R_proxy correlated with R_true under "
        "reference policy pi_ref. Theorem 1: policy return bounded "
        "by occupancy measure divergence. Section 4: KL(pi||pi_ref) "
        "regularization insufficient — chi-squared OM divergence "
        "needed. Experiments on RLHF sentiment, summarization."
    ),
    "year": 2024,
    "domain": "rlhf",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Standard RLHF with KL penalty still enables reward hacking. "
        "Proxy-true correlation degrades under optimization. Agent "
        "finds high-proxy/low-true policies that satisfy KL constraint."
    ),
    "documented_failure": (
        "Section 5: sentiment task and summarization show proxy reward "
        "increases while true reward plateaus or decreases. KL penalty "
        "alone cannot prevent this because it constrains the policy "
        "distribution, not the occupancy measure."
    ),
    "failure_mechanism": "ontological_proxy",
    "detection_type": "dynamic",
    "discovery_stage": "during_training",
    "fix_known": (
        "Replace KL penalty with chi-squared occupancy measure "
        "divergence: D_chi2(eta_pi || eta_pi_ref). This directly "
        "constrains the state-action distribution."
    ),
    "compute_cost_class": "high",
    "is_negative_example": False,
    "encoding_rationale": {
        "advisory_learned_reward": (
            "Proxy is a learned reward model. The paper shows "
            "proxy-true correlation degrades during training."
        ),
        "kl_insufficient": (
            "KL(pi||pi_ref) constrains action probabilities but "
            "not the states visited. Agent can visit novel states "
            "where the proxy is miscalibrated."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    model = EnvironmentModel(
        name="RLHF with KL — Proxy Hacking (Laidlaw 2024)",
        max_steps=512,
        gamma=1.0,
        n_states=100000,
        n_actions=50000,
        action_type="discrete",
        death_probability=0.0,
    )

    # Learned reward model (proxy)
    model.add_reward_source(RewardSource(
        name="proxy_reward_model",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # KL penalty: -beta * log(pi/pi_ref) per token
    model.add_reward_source(RewardSource(
        name="kl_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.01,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
