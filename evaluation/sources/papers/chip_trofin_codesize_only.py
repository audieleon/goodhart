"""MLGO compiler inlining — code size proxy ignores runtime (Trofin 2021).

Google's MLGO uses RL to decide function inlining in LLVM. Reward
is -delta(code_size). But code size is a proxy for runtime
performance — inlining can increase code size while improving
runtime via cache effects. The agent learns to minimize code size
at the expense of execution speed.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "trofin_codesize_proxy",
    "source_paper": (
        'Trofin, Qian, Brevdo, Lin, Choromanski & Li, "MLGO: A '
        'Machine Learning Guided Compiler Optimizations Framework," '
        'arXiv 2021 (Google Brain)'
    ),
    "paper_url": "https://arxiv.org/abs/2101.04808",
    "source_code_url": "https://github.com/nicktehrany/MLGO",
    "reward_location": (
        "Section 4.1: reward r = -(native_size_after - "
        "native_size_before) for each inlining decision. Positive "
        "reward when inlining reduces code size, negative when it "
        "increases. No runtime performance term — code size is "
        "used as a proxy because it's cheaper to measure."
    ),
    "year": 2021,
    "domain": "chip_design",
    "encoding_basis": "primary_source",
    "verification_date": "2026-05-01",
    "brief_summary": (
        "Compiler inlining RL minimizes code size as proxy for "
        "performance. But inlining can increase code size while "
        "improving runtime (better cache locality, eliminated call "
        "overhead). The proxy-true relationship is non-monotonic."
    ),
    "documented_failure": (
        "Section 5: authors acknowledge code size is a proxy — "
        "'we chose to optimize for code size reduction as a "
        "starting point because it is easier to measure and more "
        "deterministic than execution time.' The proxy was chosen "
        "for convenience, not accuracy."
    ),
    "failure_mechanism": "ontological_proxy",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Use execution time as reward (expensive but accurate). "
        "Or use multi-objective optimization with both code size "
        "and runtime. Authors note this is future work."
    ),
    "compute_cost_class": "high",
    "is_negative_example": False,
    "encoding_rationale": {
        "proxy_for_convenience": (
            "Code size is used BECAUSE it's easy to measure, not "
            "because it's the true objective. The authors explicitly "
            "say this. proxy_reward_hackability could fire."
        ),
        "negative_only": (
            "Reward is negative code size change. All rewards are "
            "zero or negative. negative_only_reward should fire."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    model = EnvironmentModel(
        name="MLGO Inlining — Code Size Proxy (Trofin 2021)",
        max_steps=200,
        gamma=1.0,
        n_states=100000,
        n_actions=2,
        action_type="discrete",
        death_probability=0.0,
    )

    # -delta(code_size): negative when inlining increases size
    model.add_reward_source(RewardSource(
        name="code_size_delta",
        reward_type=RewardType.PER_STEP,
        value=-0.01,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
