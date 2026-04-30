"""COSY synchrotron injection optimization (Awal et al. 2025).

r = precision - penalty - 1. Beam alignment × transmission
efficiency, with penalty for unnecessary magnet changes.
Deployed on real COSY accelerator at FZ Jülich.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "synchrotron_cosy",
    "source_paper": (
        'Awal, Hetzel, Gebel & Pretz, "Injection Optimization at '
        'Particle Accelerators via Reinforcement Learning: From '
        'Simulation to Real-World Application," '
        'Phys. Rev. Accel. Beams 28, 034601, 2025'
    ),
    "paper_url": "https://arxiv.org/abs/2406.12735",
    "source_code_url": None,
    "reward_location": (
        "Section 4.5, Eq. 16-20: transverse = softplus(alignment error). "
        "transmission = fraction particles transmitted (Eq. 17). "
        "precision = (1-transverse)*transmission (Eq. 18). "
        "penalty = k*precision*1(changed)*1(worse) (Eq. 19). "
        "r = precision - penalty - 1 (Eq. 20). "
        "11 actions (4 quad + 7 steer), SAC, gamma=0.95, 32 steps."
    ),
    "year": 2025,
    "domain": "fusion",
    "encoding_basis": "primary_source",
    "verification_date": "2026-05-01",
    "brief_summary": (
        "RL optimizes particle beam injection into COSY synchrotron "
        "by adjusting 11 magnets. Reward combines beam alignment "
        "precision with transmission efficiency, minus penalty for "
        "unnecessary magnet changes. Sim-to-real on live accelerator, "
        "matches human operator beam cross-section quality."
    ),
    "documented_failure": (
        "Early experiments without transmission term (Eq. 17): agent "
        "learned to match beam position/spread but lost particles — "
        "undesired behavior where beam matches target shape but most "
        "particles don't reach the injection point. Adding transmission "
        "term fixed this by requiring both alignment AND efficiency."
    ),
    "failure_mechanism": "proxy_reward",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Add transmission term (Eq. 17-18): precision = (1-transverse) "
        "* transmission. The multiplicative structure ensures both "
        "alignment AND efficiency must be high."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "all_negative_tracking": (
            "r = precision - penalty - 1. Range [-inf, 0]. The -1 "
            "offset makes reward always negative (tracking controller "
            "pattern). goodhart should fire negative_only as WARNING."
        ),
        "multiplicative_precision": (
            "precision = (1-transverse) * transmission. Both factors "
            "in [0,1]. Multiplicative structure ensures the agent "
            "can't optimize one at the expense of the other."
        ),
        "documented_proxy_failure": (
            "Without transmission, alignment alone is a proxy that "
            "the agent hacks by losing particles. The fix adds the "
            "missing constraint. This is a documented proxy_reward "
            "failure with a clear structural fix."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Eq. 20: r = precision - penalty - 1
    # 11 magnets, SAC, gamma=0.95, 32 steps

    model = EnvironmentModel(
        name="COSY Synchrotron Injection (Awal et al. 2025)",
        max_steps=32,
        gamma=0.95,
        n_actions=11,  # 4 quadrupoles + 7 steerers
        action_type="continuous",
        death_probability=0.0,
    )

    # Precision: (1-transverse) * transmission
    # Range [0, 1], all-negative after -1 offset
    model.add_reward_source(RewardSource(
        name="beam_precision",
        reward_type=RewardType.PER_STEP,
        value=-0.5,  # precision - 1, typically around -0.5
        requires_action=True,
        intentional=True,
        state_dependent=True,
        scales_with="distance",
    ))

    # Penalty for unnecessary magnet changes (Eq. 19)
    # Only fires when change worsens performance
    model.add_reward_source(RewardSource(
        name="change_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.1,  # k * precision when triggered
        requires_action=True,
        state_dependent=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
