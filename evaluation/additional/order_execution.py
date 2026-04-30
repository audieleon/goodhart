"""Optimal order execution (Hafsi & Vittori 2024, ABIDES).

Reward = implementation shortfall - depth penalty - terminal inventory
penalty. Agent splits large order to minimize market impact.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "order_execution",
    "source_paper": (
        'Hafsi & Vittori, "Optimal Execution with Reinforcement Learning '
        'in a Multi-Agent Market Simulator," arXiv:2411.06389, 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2411.06389",
    "source_code_url": None,
    "reward_location": (
        "Section IV-A, Eq. 4: r_t = Q_t^k*(P_0-P_t) - alpha*d_t "
        "- beta*I_T*1(t=T). alpha=2 (depth penalty), beta=5/share "
        "(terminal inventory). Section V-A: Q_min=20 shares, "
        "X_0=20000 shares, 30-min window, 1-sec steps, gamma=0.9999."
    ),
    "year": 2024,
    "domain": "finance",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "RL agent executes a large buy order (20,000 shares) over 30 "
        "minutes by splitting into smaller orders to minimize market "
        "impact. Reward balances execution quality (implementation "
        "shortfall) against LOB depth consumption and deadline risk."
    ),
    "documented_failure": (
        "None — well-designed reward. Agent outperforms TWAP, passive, "
        "and random baselines. Front-loads execution then tapers, "
        "minimizing market disruption (Table I: lowest IS variance)."
    ),
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "medium",
    "is_negative_example": True,
    "encoding_rationale": {
        "three_components": (
            "IS (can be positive or negative), depth penalty (always "
            "negative), terminal penalty (negative at deadline only). "
            "The mix of positive and negative per-step values means "
            "goodhart should not fire negative_only_reward."
        ),
        "terminal_deadline": (
            "beta*I_T at t=T is a deadline penalty for unfinished "
            "inventory. This creates urgency — doing nothing accumulates "
            "terminal penalty. Well-designed incentive structure."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Eq. 4: r_t = Q*IS - alpha*depth - beta*I_T*1(t=T)
    # 20,000 shares, 30 min, 1-sec steps

    model = EnvironmentModel(
        name="Order Execution (Hafsi & Vittori 2024)",
        max_steps=1800,  # 30 min * 60 sec
        gamma=0.9999,
        n_actions=5,  # do nothing + 4 order sizes
        action_type="discrete",
        death_probability=0.0,
    )

    # Implementation shortfall: Q*(P0-Pt)
    # Positive when buying below arrival price
    model.add_reward_source(RewardSource(
        name="implementation_shortfall",
        reward_type=RewardType.PER_STEP,
        value=0.5,  # normalized average IS per step
        requires_action=True,
        intentional=True,
        state_dependent=True,
        value_range=(-1.0, 1.0),
    ))

    # Depth consumption penalty: -alpha*d_t, alpha=2
    model.add_reward_source(RewardSource(
        name="depth_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.1,  # -alpha*d_t normalized
        requires_action=True,
        state_dependent=True,
    ))

    # Terminal inventory penalty: -beta*I_T, beta=5/share
    model.add_reward_source(RewardSource(
        name="terminal_inventory_penalty",
        reward_type=RewardType.TERMINAL,
        value=-5.0,  # beta * remaining shares
        requires_action=True,
        state_dependent=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
