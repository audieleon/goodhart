"""Adaptive network routing via DRL (Abrol et al. 2024, A*STAR/SIT).

R = throughput_ratio + 1/delay. Agent selects routing paths for
traffic flows using graph convolutional Q-network (DGCNN).
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "network_routing",
    "source_paper": (
        'Abrol, Murali Mohan & Truong-Huu, "A Deep Reinforcement '
        'Learning Approach for Adaptive Traffic Routing in Next-gen '
        'Networks," arXiv:2402.04515, 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2402.04515",
    "source_code_url": None,
    "reward_location": (
        "Section III-A.3, Eq. 2: R = T^act/T^req + 1/D. "
        "T^act = actual throughput, T^req = requested rate, "
        "D = delay. Eq. 4: congestion flag c_t. Eq. 6: Q-target "
        "discounts by (1-c_t). Table I: gamma=0.99, lr=0.00025. "
        "9-node random + 14-node NSFNET topologies, 100 Mbps links."
    ),
    "year": 2024,
    "domain": "multi_agent",
    "encoding_basis": "primary_source",
    "verification_date": "2026-05-01",
    "brief_summary": (
        "RL agent routes network traffic flows by selecting among "
        "K shortest paths. Reward balances throughput efficiency "
        "(actual/requested bandwidth) and low delay (1/D). Graph "
        "convolutional network learns topology-aware routing. "
        "7.8% throughput increase, 16.1% delay reduction vs OSPF."
    ),
    "documented_failure": (
        "None — well-designed reward. DRL-GCNN outperforms both "
        "OSPF and MLP-based DRL on throughput and delay. Adapts "
        "to changing traffic patterns within ~500 episodes."
    ),
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "medium",
    "is_negative_example": True,
    "encoding_rationale": {
        "two_component_qos": (
            "Throughput ratio [0,1] + inverse delay (positive). "
            "Both positive, both require action (path selection), "
            "both state-dependent (network load determines throughput "
            "and delay). Well-balanced QoS objectives."
        ),
        "congestion_flag": (
            "c_t is not a reward component but modifies the Q-target "
            "(Eq. 6): y = R + (1-c)*gamma*max Q'. When congested, "
            "future value is zeroed — similar to episode termination. "
            "This is a clever implicit penalty."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # R = T^act/T^req + 1/D
    # K shortest paths, Deep Q-Learning, DGCNN

    model = EnvironmentModel(
        name="Network Routing DRL-GCNN (Abrol et al. 2024)",
        max_steps=2000,  # episodes of 100 flows each
        gamma=0.99,
        n_actions=3,  # K=3 shortest paths per flow
        action_type="discrete",
        death_probability=0.0,
    )

    # Throughput ratio: T^act / T^req
    # [0, 1] — fraction of requested bandwidth achieved
    model.add_reward_source(RewardSource(
        name="throughput_ratio",
        reward_type=RewardType.PER_STEP,
        value=0.7,  # average throughput efficiency
        requires_action=True,
        intentional=True,
        state_dependent=True,
        value_range=(0.0, 1.0),
    ))

    # Inverse delay: 1/D
    # Higher reward for lower-latency paths
    model.add_reward_source(RewardSource(
        name="inverse_delay",
        reward_type=RewardType.PER_STEP,
        value=0.3,  # normalized 1/D
        requires_action=True,
        intentional=True,
        state_dependent=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
