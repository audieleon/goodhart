"""Tearing instability avoidance (Seo et al. 2024, Nature).

R = beta_N if safe, else k-T. Two-component conditional reward
balancing plasma pressure against tearing instability risk.
Deployed on DIII-D tokamak.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "tearing_avoidance",
    "source_paper": (
        'Seo, Kim, Jalalvand, Conlin, Rothstein, Abbate, Erickson, '
        'Wai, Shousha & Kolemen, "Avoiding fusion plasma tearing '
        'instability with deep reinforcement learning," '
        'Nature 626, 746-751, 2024'
    ),
    "paper_url": "https://www.nature.com/articles/s41586-024-07024-9",
    "source_code_url": None,
    "reward_location": (
        "Section RL design, Eq. 1: R(beta_N, T; k) = beta_N if T < k, "
        "else k - T. beta_N = normalized plasma pressure, T = tearability, "
        "k = threshold {0.2, 0.5, 0.7}. Actions: beam power + "
        "triangularity. 25ms control. DIII-D tokamak."
    ),
    "year": 2024,
    "domain": "fusion",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "RL avoids tearing instability in DIII-D tokamak by actively "
        "tracking the stable path between high pressure and instability "
        "onset. Elegant 2-component conditional reward: pursue pressure "
        "when safe, retreat when tearability exceeds threshold."
    ),
    "documented_failure": (
        "Without AI control (discharge 193273): traditional controller "
        "maintains beta_N=2.3, but tearing instability occurs at t=2.6s "
        "leading to disruption at t=3.1s. With AI (discharge 193280): "
        "tearability maintained below threshold for entire discharge. "
        "Low threshold k=0.2 is too conservative (hits beam power "
        "lower bound). k=0.5 is the sweet spot."
    ),
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "post_deployment",
    "fix_known": None,
    "compute_cost_class": "extreme",
    "is_negative_example": True,
    "encoding_rationale": {
        "conditional_reward": (
            "R is conditional: positive (beta_N) when safe (T<k), "
            "negative (k-T) when unstable (T>=k). This is a piecewise "
            "function, not two independent sources. Encoded as two "
            "sources: positive pressure tracking + negative instability "
            "penalty. The conditional structure means they don't "
            "both contribute simultaneously."
        ),
        "threshold_tuning": (
            "k=0.2 too conservative (paper Fig. 4b: beam power hits "
            "lower bound, disruption still occurs). k=0.5 works "
            "(Fig. 4c: stable for entire discharge). k=0.7 too "
            "aggressive (pushes pressure too high). The threshold "
            "IS the reward design decision."
        ),
        "elegance": (
            "Possibly the most elegant reward in our dataset: "
            "2 components, 1 threshold, deployed on real fusion "
            "reactor. Maximum information with minimum specification."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Eq. 1: R = beta_N if T < k, else k - T
    # DIII-D tokamak, k=0.5, 25ms control

    model = EnvironmentModel(
        name="Tearing Avoidance DIII-D (Seo et al. 2024)",
        max_steps=240,  # ~6 seconds at 25ms
        gamma=0.99,
        n_actions=2,  # beam power + triangularity
        action_type="continuous",
        n_states=100000,  # plasma profiles
        death_probability=0.05,  # disruption terminates
    )

    # Positive: plasma pressure (beta_N) when safe
    # Earned when T < k (tearability below threshold)
    model.add_reward_source(RewardSource(
        name="plasma_pressure",
        reward_type=RewardType.PER_STEP,
        value=2.0,  # typical beta_N target ~2.0
        requires_action=True,
        intentional=True,
        state_dependent=True,
        scales_with="pressure",
    ))

    # Negative: instability penalty when T >= k
    # R = k - T, which is negative when T > k
    model.add_reward_source(RewardSource(
        name="instability_penalty",
        reward_type=RewardType.PER_STEP,
        value=-1.0,  # k - T when T > k (negative)
        requires_action=True,
        state_dependent=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
