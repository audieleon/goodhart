"""Example: Mountain Car — the canonical reward desert.

Every step costs -1 and the goal gives 0. No positive reward exists
anywhere, so the agent has no gradient toward the hilltop.

Source: Moore 1990, "Efficient Memory-based Learning for Robot
  Control." Also Sutton & Barto 2018, Section 10.1.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "mountain_car_desert",
    "source_paper": "Moore 1990, Efficient Memory-based Learning for Robot Control; Sutton & Barto 2018, Section 10.1",
    "paper_url": "https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.pdf",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 1990,
    "domain": "control",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to reach the hilltop. Instead every step costs -1 and the goal gives 0 -- no positive signal anywhere.",
    "documented_failure": "Every step costs -1 and the goal gives 0; no positive reward exists anywhere; the agent gets -1 whether building momentum or standing still, making learning extremely slow",
    "failure_mechanism": "reward_desert",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Add positive reward at goal or potential-based shaping toward hilltop",
    "compute_cost_class": "low",
    "is_negative_example": True,
    "encoding_rationale": {
        "negative_only_reward": "No positive reward exists; optimal strategy is least total penalty",
        "canonical_example": "The textbook example of reward deserts from Sutton & Barto",
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Mountain Car — reward desert (all negative)")
    print("=" * 70)
    print()
    print("Source: Moore 1990, Sutton & Barto 2018")
    print("-1 per step, 0 at goal. No positive signal anywhere.")
    print()

    model = EnvironmentModel(
        name="Mountain Car (reward desert)",
        max_steps=200,
        gamma=1.0,
        n_actions=3,
        action_type="discrete",
        death_probability=0.0,
    )
    model.add_reward_source(
        RewardSource(
            name="step_penalty",
            reward_type=RewardType.PER_STEP,
            value=-1.0,
            requires_action=False,
        )
    )
    model.add_reward_source(
        RewardSource(
            name="goal",
            reward_type=RewardType.TERMINAL,
            value=0.0,
            requires_action=True,
            intentional=True,
            discovery_probability=0.01,
        )
    )

    engine.print_report(model)

    print()
    print("The agent earns -1 per step regardless of behavior.")
    print("No positive reward exists. The optimal strategy is")
    print("whichever accumulates the least total penalty.")


if __name__ == "__main__":
    run_example()
