"""Supply chain inventory control (Kotecha & del Rio Chanona 2024).

Multi-agent: each supply chain node orders inventory to maximize
total profit. Revenue - ordering cost - storage cost - backlog cost.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "supplychain_inventory",
    "source_paper": (
        'Kotecha & del Rio Chanona, "Leveraging Graph Neural Networks '
        'and Multi-Agent Reinforcement Learning for Inventory Control '
        'in Supply Chains," Computers & Chemical Engineering, 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2410.18631",
    "source_code_url": None,
    "reward_location": (
        "Section 3.1, Eq. 7: max sum_i sum_t [P*g - C*o_r - V*v - B*b]. "
        "P=price, g=shipment, C=ordering cost, V=storage cost, B=backlog cost. "
        "Table 1: variable definitions. Shared reward (cooperative MARL). "
        "4 supply chain configs tested. Poisson demand (Eq. 17)."
    ),
    "year": 2024,
    "domain": "multi_agent",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Multi-echelon supply chain inventory control. Each node orders "
        "to maximize total network profit: revenue minus ordering, storage, "
        "and backlog costs. Documented finding: independent rewards cause "
        "coordination failure; shared reward fixes it."
    ),
    "documented_failure": (
        "Independent reward training leads to unstable training and "
        "significantly poorer coordination. Shared cooperative reward "
        "is the correct design. Not a reward structure failure — "
        "the reward function itself (Eq. 7) is well-designed."
    ),
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Use shared cooperative reward across all agents.",
    "compute_cost_class": "medium",
    "is_negative_example": True,
    "encoding_rationale": {
        "four_component_profit": (
            "Revenue (positive), ordering cost (negative), storage "
            "cost (negative), backlog cost (negative). All per-step, "
            "all require action (ordering decisions). Revenue is "
            "intentional (the objective)."
        ),
        "cooperative_as_single": (
            "Cooperative MARL with shared reward modeled as single-agent. "
            "The reward structure is the same regardless of whether "
            "agents share it or optimize independently."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Eq. 7: profit = P*g - C*o_r - V*v - B*b per node per step
    # Using normalized values from the paper's test configurations

    model = EnvironmentModel(
        name="Supply Chain Inventory (Kotecha et al. 2024)",
        max_steps=52,  # weekly decisions over 1 year
        gamma=0.99,
        n_actions=4,  # order quantity per node (4 nodes)
        action_type="continuous",
        death_probability=0.0,
    )

    # Revenue: P * g (price × goods sold)
    model.add_reward_source(RewardSource(
        name="revenue",
        reward_type=RewardType.PER_STEP,
        value=1.0,  # normalized price × shipment
        requires_action=True,
        intentional=True,
        state_dependent=True,
    ))

    # Ordering cost: -C * o_r
    model.add_reward_source(RewardSource(
        name="ordering_cost",
        reward_type=RewardType.PER_STEP,
        value=-0.3,  # normalized ordering cost
        requires_action=True,
        state_dependent=True,
    ))

    # Storage/holding cost: -V * v
    model.add_reward_source(RewardSource(
        name="storage_cost",
        reward_type=RewardType.PER_STEP,
        value=-0.1,  # normalized per-unit holding cost
        requires_action=True,
        state_dependent=True,
    ))

    # Backlog cost: -B * b
    model.add_reward_source(RewardSource(
        name="backlog_cost",
        reward_type=RewardType.PER_STEP,
        value=-0.5,  # normalized backlog penalty (typically > storage)
        requires_action=True,
        state_dependent=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
