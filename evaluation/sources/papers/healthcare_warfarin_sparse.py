"""Warfarin dosing — sparse delayed reward failure.

Warfarin (blood thinner) has a narrow therapeutic range (INR 2-3).
RL agent gets reward only when INR is measured (every few days),
creating extreme reward delay. Agent can't learn the dose-response
relationship from such sparse feedback.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "warfarin_sparse_delay",
    "source_paper": (
        'Nemati, Ghassemi & Clifford, "Optimal Medication Dosing '
        'from Suboptimal Clinical Examples: A Deep RL Approach," '
        'IEEE EMBC 2016. Also: Zadeh, Frick & Bhatt, "Reinforcement '
        'Learning for Warfarin Dosing," 2023'
    ),
    "paper_url": "https://ieeexplore.ieee.org/document/7591339",
    "source_code_url": None,
    "reward_location": (
        "Nemati Section III: reward r_t = 0 for most steps (no INR "
        "measurement), r_t = -|INR_t - INR_target|^2 when INR is "
        "measured (every 3-7 days). Target INR = 2.5 (range 2-3). "
        "Actions: dose adjustment in mg. Warfarin has 2-3 day lag "
        "between dose change and INR response."
    ),
    "year": 2016,
    "domain": "healthcare",
    "encoding_basis": "primary_source",
    "verification_date": "2026-05-01",
    "brief_summary": (
        "Warfarin dosing RL gets reward only when INR is measured "
        "(every 3-7 days) with 2-3 day pharmacological lag. Most "
        "timesteps have zero reward. Agent struggles with extreme "
        "reward sparsity and delayed drug response."
    ),
    "documented_failure": (
        "The combination of sparse measurement (reward only every "
        "3-7 steps) and pharmacological delay (2-3 day lag) means "
        "the agent receives feedback 5-10 steps after its action. "
        "Credit assignment is nearly impossible without a model of "
        "warfarin pharmacokinetics."
    ),
    "failure_mechanism": "credit_assignment",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Use pharmacokinetic model to provide intermediate reward "
        "estimates. Or use model-based RL with learned warfarin "
        "response dynamics."
    ),
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "sparse_reward": (
            "Reward is zero on most steps. Only non-zero when INR "
            "measured. exploration_threshold should fire due to "
            "sparse signal."
        ),
        "credit_assignment": (
            "Drug effect lags 2-3 days. Reward measured 3-7 days "
            "after dose. Total delay: 5-10 steps. "
            "reward_delay_horizon should fire."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    model = EnvironmentModel(
        name="Warfarin Dosing — Sparse + Delayed (Nemati 2016)",
        max_steps=90,
        gamma=0.99,
        n_states=1000,
        n_actions=5,
        action_type="discrete",
        death_probability=0.01,
    )

    # INR tracking: only measured every ~5 steps
    # Most steps: 0. Measurement steps: -|INR - 2.5|^2 ≈ -0.5
    model.add_reward_source(RewardSource(
        name="inr_tracking",
        reward_type=RewardType.PER_STEP,
        value=-0.1,
        state_dependent=True,
        requires_action=True,
        intentional=True,
        explore_fraction=0.2,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
