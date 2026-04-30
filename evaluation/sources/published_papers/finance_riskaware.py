"""Risk-aware financial trading reward (Srivastava et al. 2025).

Composite reward prevents single-metric reward hacking by combining
annualized return, downside risk, differential return, and Treynor ratio.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "finance_riskaware",
    "source_paper": (
        'Srivastava, Aryan & Singh, "Risk-Aware Reinforcement Learning '
        'Reward for Financial Trading," arXiv:2506.04358, 2025'
    ),
    "paper_url": "https://arxiv.org/abs/2506.04358",
    "source_code_url": None,
    "reward_location": (
        "Section 3.1, Eq. 6: R = w1*R_ann - w2*sigma_down + w3*D_ret + w4*T_ry. "
        "Eq. 2: R_ann = (prod(1+R_p,t))^(252/T) - 1. "
        "Eq. 3: sigma_down = sqrt(1/T * sum(max(0,-R_p,t)^2)). "
        "Eq. 4: D_ret = (mu_p - mu_b) / (beta_p * T). "
        "Eq. 5: T_ry = (R_ann - R_f) / beta_p. "
        "Section 3.3: weights w_i tuned via grid search on simplex."
    ),
    "year": 2025,
    "domain": "finance",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Composite reward for financial trading that prevents single-metric "
        "reward hacking. Combines return, downside risk penalty, benchmark "
        "outperformance, and Treynor ratio. Documented failure: Sharpe-only "
        "rewards encourage over-optimization of one aspect of trading."
    ),
    "documented_failure": (
        "Single-metric rewards (Sharpe ratio alone, cumulative return alone) "
        "encourage reward hacking: agent over-optimizes one aspect of "
        "trading at the expense of others. The composite reward with "
        "tunable weights prevents this by balancing return vs risk."
    ),
    "failure_mechanism": "proxy_reward",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Use composite reward R = w1*R_ann - w2*sigma_down + w3*D_ret + w4*T_ry "
        "with tunable weights. Grid search over simplex to match risk profile."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": True,
    "encoding_rationale": {
        "composite_well_designed": (
            "The composite reward is the FIX, not the failure. "
            "Encoding the well-designed version as a negative example. "
            "All four components require action (trading decisions), "
            "are state-dependent (market-driven), and intentional."
        ),
        "aggregation_note": (
            "R_ann and T_ry are episode-level aggregations (annualized "
            "over T periods), not per-step values. goodhart's per-step "
            "model approximates these as per-step averages. "
            "advisory_aggregation_trap may fire since the real "
            "objective is a ratio, not a sum."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Composite reward: R = w1*R_ann - w2*sigma_down + w3*D_ret + w4*T_ry
    # Weights sum to 1 (simplex), tuned via grid search
    # Using balanced weights: w1=w2=w3=w4=0.25

    model = EnvironmentModel(
        name="Risk-Aware Trading (Srivastava et al. 2025)",
        max_steps=252,  # 1 year of trading days
        gamma=0.99,
        n_actions=10,  # portfolio weights for N assets
        action_type="continuous",
        death_probability=0.0,
    )

    # w1 * R_ann: annualized return (Eq. 2)
    model.add_reward_source(RewardSource(
        name="annualized_return",
        reward_type=RewardType.PER_STEP,
        value=0.25,  # w1=0.25, normalized return per step
        requires_action=True,
        intentional=True,
        state_dependent=True,
        scales_with="return",
    ))

    # -w2 * sigma_down: downside risk penalty (Eq. 3)
    model.add_reward_source(RewardSource(
        name="downside_risk_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.25,  # w2=0.25, penalizes negative returns
        requires_action=True,
        state_dependent=True,
        value_type="proportional",
    ))

    # w3 * D_ret: differential return (Eq. 4)
    model.add_reward_source(RewardSource(
        name="differential_return",
        reward_type=RewardType.PER_STEP,
        value=0.25,  # w3=0.25, benchmark outperformance
        requires_action=True,
        intentional=True,
        state_dependent=True,
    ))

    # w4 * T_ry: Treynor ratio (Eq. 5)
    model.add_reward_source(RewardSource(
        name="treynor_ratio",
        reward_type=RewardType.PER_STEP,
        value=0.25,  # w4=0.25, return per unit systematic risk
        requires_action=True,
        intentional=True,
        state_dependent=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
