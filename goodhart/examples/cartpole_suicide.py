"""Example: CartPole suicidal agent (Classic RL).

Constant step penalty with no positive reward makes dying at step 1
optimal, so the agent learns to fall over immediately.

Source: Barto, Sutton & Anderson 1983; Sutton & Barto 2018 (Ch. 3.4)
"""

from goodhart.presets import PRESETS
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "cartpole_suicide",
    "source_paper": "Barto, Sutton & Anderson 1983 (CartPole); Sutton & Barto 2018 (Ch. 3.4)",
    "paper_url": None,
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 1983,
    "domain": "control",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to balance the pole. Instead it falls immediately to minimize accumulated step penalty.",
    "documented_failure": "With a constant step penalty and no positive per-step reward, dying at step 1 accumulates less penalty than surviving. The agent learns to fall over immediately.",
    "failure_mechanism": "death_beats_survival",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Add positive alive reward or use +1 per step (Gymnasium default)",
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "step_penalty_only": "No positive per-step reward makes death optimal",
    },
}


def run_example():
    model, config = PRESETS["mountain-car"]
    engine = TrainingAnalysisEngine().add_all_rules()

    print("Mountain Car / CartPole Suicide Analysis")
    print("=" * 50)
    print("Source: Moore 1990 / Gymnasium classic control")
    print(f"Step penalty: {model.total_step_penalty}/step")
    print(f"Goal: {model.max_goal_reward}")
    print(f"Max steps: {model.max_steps}")
    print()

    engine.print_report(model, config)

    print("This is a 'reward desert' — all non-goal strategies")
    print("score equally. The agent has no gradient signal to")
    print("learn from until it accidentally discovers the goal.")


if __name__ == "__main__":
    run_example()
