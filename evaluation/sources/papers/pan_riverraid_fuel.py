"""Pan et al. 2022: Atari Riverraid ontological (shooting penalty).

Proxy adds a penalty when a bullet is on screen, creating a "pacifist"
incentive. Agent avoids shooting and maneuvers to survive instead of
scoring points by destroying enemies.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "pan_riverraid_fuel",
    "source_paper": (
        'Pan, Bhatia & Steinhardt, "The Effects of Reward'
        ' Misspecification: Mapping and Mitigating Misaligned'
        ' Models," ICLR 2022'
    ),
    "paper_url": "https://arxiv.org/abs/2201.03544",
    "source_code_url": "https://github.com/aypan17/reward-misspecification",
    "reward_location": "Table 1 row 8; code: atari/wrappers.py (FuelEnv)",
    "year": 2022,
    "domain": "game",
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Atari Riverraid proxy: R = 10 normally; R = 10 -"
        " |fuel_multiplier| when a bullet is on screen. Penalizes"
        " shooting, creating a pacifist agent that avoids combat"
        " and survives by maneuvering."
    ),
    "documented_failure": (
        "Agent shoots less frequently and maneuvers to survive."
        " Larger models survive longer but score fewer true-reward"
        " points. Some agents exploit a simulator bug to halt"
        " the plane entirely."
    ),
    "failure_mechanism": "ontological",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Use raw game score without shooting penalty.",
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "requires_action": (
            "Shooting penalty triggers when agent fires (bullet"
            " on screen). Survival bonus is passive."
        ),
        "intentional": (
            "The +10 survival bonus is intentional. The shooting"
            " penalty is the ontological addition -- the true"
            " reward (game score) does not penalize shooting."
        ),
        "can_loop": (
            "Agent can loop by surviving without shooting,"
            " collecting +10 per step indefinitely."
        ),
        "value": (
            "R = 10 normally. R = 10 - |fuel_multiplier| when"
            " bullet on screen. True reward is raw game score."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Pan et al. 2022: Atari Riverraid -- Ontological (pacifist)")
    print("=" * 70)

    # True reward: game score
    # Proxy (FuelEnv): R = 10 normally; R = 10 - |fuel_multiplier|
    #   when bullet on screen
    # gamma=0.99, max_steps=5000, n_actions=18, discrete

    model = EnvironmentModel(
        name="Riverraid Pacifist (Pan et al. 2022)",
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

    # Shooting penalty: -|fuel_multiplier| when bullet on screen
    # This is ontologically wrong -- the true game score rewards
    # shooting enemies, it does not penalize it.
    model.add_reward_source(RewardSource(
        name="shooting_penalty",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
        requires_action=True,
        max_occurrences=0,
        state_dependent=True,
        scales_with="bullet_on_screen",
        value_type="proportional",
        value_params={"scale": 1.0},
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
