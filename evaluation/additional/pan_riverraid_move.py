"""Pan et al. 2022: Atari Riverraid misweighting (movement penalty).

Proxy adds a per-step movement penalty that downweights the value of
moving. Agent learns to stay stationary to avoid the penalty, halting
the plane at the start of the level.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "pan_riverraid_move",
    "source_paper": (
        'Pan, Bhatia & Steinhardt, "The Effects of Reward'
        ' Misspecification: Mapping and Mitigating Misaligned'
        ' Models," ICLR 2022'
    ),
    "paper_url": "https://arxiv.org/abs/2201.03544",
    "source_code_url": "https://github.com/aypan17/reward-misspecification",
    "reward_location": "Table 1 row 7; code: atari/wrappers.py (MoveEnv)",
    "year": 2022,
    "domain": "game",
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Atari Riverraid proxy: R = 10 + move_penalty*(0 if"
        " stationary else -1). Penalizes movement, so the agent"
        " learns to halt the plane and avoid moving entirely."
    ),
    "documented_failure": (
        "Agent exploits a simulator quirk to halt the plane at"
        " the start of the level. The plane stays stationary,"
        " collecting the +10 per-step survival bonus while"
        " avoiding the movement penalty."
    ),
    "failure_mechanism": "misweighting",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Use raw game score without movement penalty.",
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "requires_action": (
            "Movement penalty triggers on any non-stationary"
            " action. Survival bonus is passive."
        ),
        "intentional": (
            "The +10 survival bonus is intentional (alive reward)."
            " Movement penalty is the misspecified term."
        ),
        "can_loop": (
            "Agent loops by staying stationary indefinitely,"
            " collecting +10 per step."
        ),
        "value": (
            "R = 10 + move_penalty * (0 if stationary else -1)."
            " True reward is the raw game score."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Pan et al. 2022: Atari Riverraid -- Misweighting (movement)")
    print("=" * 70)

    # True reward: game score (points from shooting, fuel, checkpoints)
    # Proxy (MoveEnv): R = 10 + move_penalty * (0 if stationary else -1)
    # gamma=0.99, max_steps=5000, n_actions=18, discrete

    model = EnvironmentModel(
        name="Riverraid Movement Penalty (Pan et al. 2022)",
        max_steps=5000,
        gamma=0.99,
        n_actions=18,
        action_type="discrete",
        death_probability=0.01,
    )

    # Survival bonus: +10 per step (always received)
    model.add_reward_source(RewardSource(
        name="survival_bonus",
        reward_type=RewardType.PER_STEP,
        value=10.0,
        requires_action=False,
        max_occurrences=0,
        intentional=True,
    ))

    # Movement penalty: -move_penalty if agent takes any
    # non-stationary action
    model.add_reward_source(RewardSource(
        name="movement_penalty",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
        requires_action=True,
        max_occurrences=0,
        state_dependent=True,
        scales_with="is_moving",
        value_type="constant",
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
