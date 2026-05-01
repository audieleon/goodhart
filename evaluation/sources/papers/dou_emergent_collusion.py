"""Dou et al. 2024 — emergent collusion in algorithmic trading.

Independent RL trading agents learn to collude on pricing without
explicit communication. Each agent maximizes its own profit, but
the multi-agent dynamics produce cartel-like outcomes. The reward
is correct per-agent but the multi-agent equilibrium is harmful.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "dou_emergent_collusion",
    "source_paper": (
        'Dou, Goldstein & Ji, "Algorithmic Collusion and Price '
        'Discrimination: The Over-Usage of Data," AFA 2024 / '
        'SSRN 2024 (U. Penn Wharton)'
    ),
    "paper_url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4680978",
    "source_code_url": None,
    "reward_location": (
        "Section 3: each agent i maximizes E[sum_t pi_i(t)] where "
        "pi_i(t) = (p_i(t) - c) * q_i(p, t) is per-period profit. "
        "p_i = price set by agent, c = marginal cost, q_i = demand "
        "as function of all prices. Agents independently learn "
        "pricing policies via Q-learning / policy gradient."
    ),
    "year": 2024,
    "domain": "finance",
    "encoding_basis": "primary_source",
    "verification_date": "2026-05-01",
    "brief_summary": (
        "Independent RL pricing agents learn to collude without "
        "communication — achieving near-cartel profits by implicitly "
        "coordinating on high prices. Each agent's reward (own profit) "
        "is individually correct but produces harmful equilibrium."
    ),
    "documented_failure": (
        "Dou et al.: RL agents achieve near-monopoly profits in "
        "simulated markets. The collusion emerges from repeated "
        "interaction, not explicit programming. This is a multi-agent "
        "failure where individually correct rewards produce socially "
        "harmful outcomes."
    ),
    "failure_mechanism": "ontological_proxy",
    "detection_type": "dynamic",
    "discovery_stage": "post_training",
    "fix_known": (
        "Regulatory constraints on pricing algorithms. Add social "
        "welfare term to reward. Monitor for collusive equilibria "
        "via price pattern analysis."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "multi_agent_failure": (
            "Each agent's reward is correct (maximize own profit). "
            "The failure is in the multi-agent equilibrium, not the "
            "individual reward structure. advisory_nonstationarity "
            "should fire due to other agents' changing policies."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Dou et al. 2024: pricing agents in repeated game
    model = EnvironmentModel(
        name="Emergent Algorithmic Collusion (Dou 2024)",
        max_steps=1000,
        gamma=0.99,
        n_states=10000,
        n_actions=20,
        action_type="discrete",
        death_probability=0.0,
    )

    # Per-period profit: (price - cost) * quantity
    model.add_reward_source(RewardSource(
        name="profit",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
