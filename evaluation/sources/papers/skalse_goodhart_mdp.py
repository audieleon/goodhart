"""Skalse et al. 2024 — Goodhart's Law in RL (ICLR 2024).

Geometric explanation for Goodharting in MDPs: optimizing a proxy
reward leads to sudden phase transitions where true reward
collapses. Uses occupancy measure decomposition J_R(pi) = eta_pi·R
to bound Goodharting magnitude.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "skalse_goodhart_mdp",
    "source_paper": (
        'Skalse, Farrugia-Roberts, Russell, Abercrombie & Carey, '
        '"Goodhart\'s Law in Reinforcement Learning," ICLR 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2310.09144",
    "source_code_url": None,
    "reward_location": (
        "Section 3: proxy reward R_proxy and true reward R_true. "
        "J_R(pi) = eta_pi · R where eta_pi is the occupancy measure. "
        "Theorem 3.1: Goodharting magnitude bounded by "
        "||R_proxy - R_true|| * diam(M_Pi). Section 4: phase "
        "transition experiments on gridworlds and Atari."
    ),
    "year": 2024,
    "domain": "rlhf",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Optimizing proxy reward creates phase transition where "
        "true reward suddenly collapses. Small proxy-true gap "
        "amplifies under optimization. Demonstrated on gridworlds "
        "and Atari environments."
    ),
    "documented_failure": (
        "Section 4: experiments show sharp phase transitions — proxy "
        "reward increases smoothly while true reward crashes. The "
        "transition is sudden and hard to predict, occurring when the "
        "optimal policy for the proxy first diverges from the optimal "
        "policy for the true reward."
    ),
    "failure_mechanism": "ontological_proxy",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Bound proxy-true reward gap. Use reward uncertainty "
        "estimation. Conservative optimization (beta * KL penalty). "
        "The paper shows these are necessary but may not be sufficient."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "proxy_hackability": (
            "The paper formalizes proxy reward hackability. The "
            "proxy reward is optimized, but the true reward is "
            "what matters. proxy_reward_hackability should fire."
        ),
        "advisory_learned_reward": (
            "In RLHF contexts, the proxy is a learned reward model. "
            "advisory_learned_reward should fire."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Skalse et al. 2024: Goodharting in gridworld
    # Proxy reward = true reward + noise
    model = EnvironmentModel(
        name="Goodhart MDP Phase Transition (Skalse 2024)",
        max_steps=100,
        gamma=0.99,
        n_states=100,
        n_actions=4,
        action_type="discrete",
        death_probability=0.0,
    )

    # Proxy reward (what the agent optimizes)
    model.add_reward_source(RewardSource(
        name="proxy_reward",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Small noise term (proxy-true gap)
    model.add_reward_source(RewardSource(
        name="proxy_noise",
        reward_type=RewardType.PER_STEP,
        value=0.1,
        state_dependent=True,
        requires_action=False,
        intentional=False,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
