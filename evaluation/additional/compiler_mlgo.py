"""MLGO compiler inlining-for-size (Trofin et al. 2021, Google/LLVM).

RL decides whether to inline each call site. Reward = change in
native code size. Deployed in LLVM production — 6.3% size reduction.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "compiler_mlgo",
    "source_paper": (
        'Trofin, Qian, Brevdo, Lin, Choromanski & Li, "MLGO: A Machine '
        'Learning Guided Compiler Optimizations Framework," '
        'arXiv:2101.04808, 2021. Deployed in LLVM.'
    ),
    "paper_url": "https://arxiv.org/abs/2101.04808",
    "source_code_url": "https://github.com/google/ml-compiler-opt",
    "reward_location": (
        "Section 4.1.1, Eq. 1: r = S(Caller_before) - S(Caller_after) "
        "+ S(Callee) if callee deleted, else 0. Where S(f) = native "
        "code size. If a=0 (don't inline), r=0. gamma=1 (undiscounted). "
        "Total reward R = sum of per-decision size reductions."
    ),
    "year": 2021,
    "domain": "chip_design",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "RL agent replaces LLVM's inlining heuristic for code size "
        "optimization. Binary decision per call site: inline or not. "
        "Reward = native size reduction. Deployed in production LLVM, "
        "achieving 6.3% code size reduction on Fuchsia."
    ),
    "documented_failure": (
        "None — well-designed reward, deployed in production. The "
        "reward directly measures the optimization objective (code "
        "size change). No proxy, no shaping, no approximation."
    ),
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "post_deployment",
    "fix_known": None,
    "compute_cost_class": "high",
    "is_negative_example": True,
    "encoding_rationale": {
        "direct_measurement": (
            "Reward IS the objective — delta code size. No proxy, "
            "no shaping. This is the cleanest possible reward design: "
            "the thing you measure is the thing you optimize."
        ),
        "binary_action": (
            "Actions are binary (inline=1, don't=0). Deterministic "
            "transitions. Sequential decisions over call graph."
        ),
        "negative_values_possible": (
            "Reward can be negative (inlining increases size) or "
            "positive (inlining reduces size). The zero-reward "
            "baseline (don't inline) means goodhart should not "
            "fire negative_only_reward."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Per-decision reward: delta native code size
    # Eq. 1: r = S(before) - S(after) + S(callee) if deleted
    # Binary action, deterministic, gamma=1

    model = EnvironmentModel(
        name="MLGO Compiler Inlining (Trofin et al. 2021)",
        max_steps=1000,  # call sites per module
        gamma=1.0,  # undiscounted
        n_actions=2,  # inline or not
        action_type="discrete",
        death_probability=0.0,
    )

    # Size reduction reward (Eq. 1)
    # Positive when inlining reduces size, negative when it increases
    # Zero when not inlining
    model.add_reward_source(RewardSource(
        name="code_size_reduction",
        reward_type=RewardType.PER_STEP,
        value=0.5,  # average: some inlines reduce, some increase
        requires_action=True,
        intentional=True,
        state_dependent=True,
        value_range=(-1.0, 1.0),  # can be positive or negative
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
