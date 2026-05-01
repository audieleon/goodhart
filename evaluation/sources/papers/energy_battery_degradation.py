"""Battery storage RL — degradation cost missing from reward.

Battery energy storage agents maximize arbitrage (buy low, sell
high) but the reward typically ignores battery degradation costs.
Deep cycling (0-100% charge) degrades lithium-ion batteries 10x
faster than shallow cycling (40-60%). Agent learns to deep cycle.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "battery_degradation_missing",
    "source_paper": (
        'Kwon & Zhu, "Reinforcement Learning for Battery Energy '
        'Storage with Degradation," 2022. Degradation omission '
        'also documented in: Cao, Wan, Li, Li & Wu, "RL for '
        'Battery Energy Storage Dispatch," IJCAI 2020'
    ),
    "paper_url": "https://arxiv.org/abs/2205.13583",
    "source_code_url": None,
    "reward_location": (
        "Kwon Section 2: r_t = p_t * (d_t - c_t) - lambda * "
        "degradation_cost(SoC, DoD). Without degradation term "
        "(lambda=0): r_t = p_t * (d_t - c_t) = price × net "
        "discharge. Agent maximizes arbitrage by deep cycling. "
        "With degradation: lambda > 0 penalizes cycle depth."
    ),
    "year": 2022,
    "domain": "energy",
    "encoding_basis": "primary_source",
    "verification_date": "2026-05-01",
    "brief_summary": (
        "Battery storage RL without degradation cost learns to "
        "deep cycle (0-100% SoC) for maximum arbitrage. This "
        "destroys the battery 10x faster than shallow cycling. "
        "The degradation cost is often omitted because it's hard "
        "to model accurately."
    ),
    "documented_failure": (
        "Kwon Section 1: 'most existing RL approaches for battery "
        "storage do not account for degradation.' Without the "
        "degradation term, the agent treats the battery as having "
        "infinite cycle life and deep-cycles on every price swing."
    ),
    "failure_mechanism": "ontological_proxy",
    "detection_type": "structural",
    "discovery_stage": "post_training",
    "fix_known": (
        "Kwon: add degradation cost lambda * deg(SoC, DoD) to "
        "reward. Use rainflow counting or equivalent full cycle "
        "degradation model. The lambda parameter calibrates "
        "against battery replacement cost."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "missing_degradation": (
            "Without degradation cost, the agent sees only "
            "arbitrage profit. advisory_missing_constraint should "
            "fire — continuous action with no penalty for extreme "
            "cycling behavior."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    model = EnvironmentModel(
        name="Battery Storage — No Degradation Cost",
        max_steps=8760,
        gamma=0.99,
        n_states=10000,
        n_actions=1,
        action_type="continuous",
        death_probability=0.0,
    )

    # Arbitrage profit: price × net discharge
    model.add_reward_source(RewardSource(
        name="arbitrage_profit",
        reward_type=RewardType.PER_STEP,
        value=0.1,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
