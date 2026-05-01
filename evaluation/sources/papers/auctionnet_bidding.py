"""AuctionNet: Ad auction auto-bidding (Su et al. 2024, NeurIPS D&B).

Maximize total conversion value subject to budget constraint.
Reward is sum of won impression values; budget is a hard episode
constraint. Multi-agent POSG with 48 competing bidders.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "auctionnet_bidding",
    "source_paper": (
        'Su, Huo, Zhang, Dou, Yu, Xu, Lu & Zheng, "AuctionNet: A Novel '
        'Benchmark for Decision-Making in Large-Scale Games," NeurIPS 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2412.10798",
    "source_code_url": "https://github.com/alimama-tech/AuctionNet",
    "reward_location": (
        "Section 2, Eq. 1: maximize sum_t <x_t, v_t> s.t. sum_t <x_t, "
        "c_t> <= omega_i. r_i(s,a) = <x_i, v_i> (value of won "
        "impressions). Budget omega_i is a hard constraint. GSP auction "
        "mechanism. 48 competing agents, T=48 decision steps per episode. "
        "Section 5: CPA constraint adds penalty for exceeding target CPA."
    ),
    "year": 2024,
    "domain": "multi_agent",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Auto-bidding agents maximize total ad conversion value "
        "subject to budget constraints in large-scale GSP auctions. "
        "Budget is a hard episode constraint, not in the reward."
    ),
    "documented_failure": None,
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": None,
    "fix_known": None,
    "compute_cost_class": "high",
    "is_negative_example": True,
    "encoding_rationale": {
        "advisory_constrained_rl": (
            "Budget constraint (omega_i) is a hard episode-level "
            "constraint not reflected in the per-step reward. The "
            "agent receives r_i = <x_i, v_i> (won value) with no "
            "penalty for spending. Budget enforcement is external. "
            "advisory_constrained_rl should fire."
        ),
        "multi_agent": (
            "48 competing bidders in POSG. Each agent's reward "
            "depends on all agents' bids. Non-stationary from each "
            "agent's perspective."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Su et al. 2024: Auto-bidding in GSP auctions
    # 48 agents, T=48 steps, action = bid coefficient alpha_i
    model = EnvironmentModel(
        name="AuctionNet Auto-Bidding (Su et al. 2024)",
        max_steps=48,
        gamma=0.99,
        n_states=100000,
        n_actions=1,
        action_type="continuous",
        death_probability=0.0,
    )

    # r_i(s,a) = <x_i, v_i> — value of won impressions
    model.add_reward_source(RewardSource(
        name="won_impression_value",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
