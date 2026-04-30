"""Practical tokamak RL control (Tracey et al. 2024, DeepMind/EPFL).

Extends Degrave 2022 with reward shaping: SmoothMax combiner +
SoftPlus components with good/bad reference points. Narrow reward
parameters achieve 65% shape accuracy improvement on TCV.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "tokamak_practical",
    "source_paper": (
        'Tracey, Michi, Chervonyi, Davies, Paduraru, Lazic, Felici, '
        'Ewalds, Donner, Galperti, Buchli, Neunert, Huber, Evens, '
        'Kurylowicz, Mankowitz, Riedmiller & TCV Team, "Towards '
        'Practical Reinforcement Learning for Tokamak Magnetic Control," '
        'Fusion Engineering and Design 200, 114171, 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2307.11546",
    "source_code_url": None,
    "reward_location": (
        "Reward Shaping section pages 6-8. Eq. 2: SmoothMax combiner "
        "with α<0 (emphasizes worst component). Eq. 3: SoftPlus "
        "component scaling with good/bad reference points. "
        "Table 1: Baseline (good=0.005, bad=0.05) vs Narrow "
        "(good=0, bad=0.025) vs Schedule (bad annealing). "
        "Narrow achieves 65% LCFS RMSE improvement (0.201 vs 0.567cm)."
    ),
    "year": 2024,
    "domain": "fusion",
    "encoding_basis": "primary_source",
    "verification_date": "2026-05-01",
    "brief_summary": (
        "Extends Degrave 2022 tokamak control with reward shaping "
        "methodology. SmoothMax combines components emphasizing the "
        "worst-performing objective. Tighter reward parameters (narrow "
        "good/bad) improve shape accuracy 65% but make exploration harder. "
        "Deployed on TCV with 3x faster training via transfer learning."
    ),
    "documented_failure": (
        "Reward Schedule (annealing bad from 0.1 to 0.025) performs "
        "WORSE than static Narrow reward on the simple stabilization "
        "task (LCFS 0.490 vs 0.201cm). The curriculum doesn't help when "
        "exploration isn't the bottleneck. Key insight: reward shaping "
        "affects final accuracy, not just learning speed."
    ),
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Use tight reward parameters (good=0, bad=0.025) when "
        "exploration is easy. Use annealing when exploration is hard. "
        "Transfer learning from existing policies reduces training 3x."
    ),
    "compute_cost_class": "extreme",
    "is_negative_example": True,
    "encoding_rationale": {
        "extends_degrave": (
            "Same modular architecture as tokamak_degrave.py but with "
            "improved SmoothMax combiner and SoftPlus component scaling. "
            "Encoding the Narrow Reward configuration (best result)."
        ),
        "smoothmax_combiner": (
            "SmoothMax(α<0) emphasizes the worst component — the agent "
            "must satisfy ALL objectives, not just the easy ones. This "
            "is a design choice that trades off easy convergence for "
            "uniform quality across all objectives."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Same architecture as Degrave 2022, Narrow Reward config
    # SmoothMax combiner, SoftPlus components, good=0, bad=0.025

    model = EnvironmentModel(
        name="Tokamak TCV Practical (Tracey et al. 2024)",
        max_steps=10000,
        gamma=0.99,
        n_actions=19,
        action_type="continuous",
        n_states=100000,
        death_probability=0.01,
    )

    # Plasma current tracking (SoftPlus scaled, good=0, bad=0.025)
    model.add_reward_source(RewardSource(
        name="current_tracking",
        reward_type=RewardType.PER_STEP,
        value=30.0,
        requires_action=True,
        intentional=True,
        state_dependent=True,
        scales_with="distance",
    ))

    # Plasma shape tracking (32 LCFS control points)
    model.add_reward_source(RewardSource(
        name="shape_tracking",
        reward_type=RewardType.PER_STEP,
        value=30.0,
        requires_action=True,
        intentional=True,
        state_dependent=True,
        scales_with="distance",
    ))

    # Position tracking (R, Z)
    model.add_reward_source(RewardSource(
        name="position_tracking",
        reward_type=RewardType.PER_STEP,
        value=30.0,
        requires_action=True,
        intentional=True,
        state_dependent=True,
        scales_with="distance",
    ))

    # Termination penalty
    model.add_reward_source(RewardSource(
        name="termination_penalty",
        reward_type=RewardType.TERMINAL,
        value=-100.0,
        requires_action=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
