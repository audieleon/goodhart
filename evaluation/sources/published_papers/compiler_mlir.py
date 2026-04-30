"""MLIR RL compiler optimization (Tirichine et al. 2024, NYU Abu Dhabi).

Terminal reward = log(speedup). Agent selects loop-level
transformations (tiling, fusion, interchange, vectorization).
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "compiler_mlir",
    "source_paper": (
        'Tirichine, Ameur, Bendib, Aouadj, Bouchama, Bouloudene & '
        'Baghdadi, "A Reinforcement Learning Environment for Automatic '
        'Code Optimization in the MLIR Compiler," arXiv:2409.11068, 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2409.11068",
    "source_code_url": "https://github.com/Modern-Compilers-Lab/MLIR-RL",
    "reward_location": (
        "Section IV-C, page 6: reward = log(speedup) = log(T_old/T_new). "
        "Terminal only — intermediate steps get 0. Log for additive "
        "accumulation. Tried intermediate rewards but slowed training "
        "without improving policy. 6 transformations, PPO, Linalg."
    ),
    "year": 2024,
    "domain": "chip_design",
    "encoding_basis": "primary_source",
    "verification_date": "2026-05-01",
    "brief_summary": (
        "RL agent optimizes MLIR compiler loop nests by selecting "
        "transformations (tiling, fusion, interchange, vectorization). "
        "Terminal reward = log(speedup). Tested on PyTorch deep learning "
        "and LQCD (lattice quantum chromodynamics) code."
    ),
    "documented_failure": (
        "Intermediate rewards (executing code after each step) don't "
        "improve the policy and slow training significantly. Terminal-only "
        "reward works better despite the sparse signal."
    ),
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "high",
    "is_negative_example": True,
    "encoding_rationale": {
        "terminal_log_speedup": (
            "log(T_old/T_new) at final step only. Same pattern as "
            "MLGO — terminal measurement of actual objective. Log "
            "transforms multiplicative speedup to additive reward."
        ),
        "intermediate_not_helpful": (
            "Paper tested intermediate rewards but found them unhelpful. "
            "The cost of executing code at each step outweighs the "
            "benefit of denser signal. Interesting finding for reward "
            "design: sparser can be better."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Terminal reward: log(T_old / T_new) = log(speedup)
    # 6 transformation types, multi-discrete, PPO

    model = EnvironmentModel(
        name="MLIR Compiler Optimization (Tirichine et al. 2024)",
        max_steps=50,  # max transformation sequence length
        gamma=1.0,  # undiscounted (terminal only)
        n_actions=100,  # multi-discrete: 6 transforms × params
        action_type="discrete",
        death_probability=0.0,
    )

    # Terminal speedup reward
    # Positive when optimization improves performance,
    # negative when it degrades (log ratio)
    model.add_reward_source(RewardSource(
        name="log_speedup",
        reward_type=RewardType.TERMINAL,
        value=1.0,  # log(speedup), positive = faster
        requires_action=True,
        intentional=True,
        state_dependent=True,
        value_range=(-2.0, 3.0),  # can be negative (slowdown)
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
