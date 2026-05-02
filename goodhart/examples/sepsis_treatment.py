"""Example: Sepsis Treatment RL -- SOFA score gaming.

Shaping rewards on SOFA/lactate are loopable via drug cycling, creating clinical risk.
Source: Raghu et al. 2017 (NeurIPS ML4H), Peng et al. 2023 (npj Digital Medicine)
"""

from goodhart import reward_function, RewardSource, RewardType

METADATA = {
    "id": "sepsis_treatment",
    "source_paper": "Raghu et al. 2017 (NeurIPS ML4H), Peng et al. 2023 (npj Digital Medicine)",
    "paper_url": "https://arxiv.org/abs/1711.09602",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2017,
    "domain": "healthcare",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to optimize sepsis treatment. Instead it could game SOFA/lactate shaping by cycling drug combinations that reduce scores short-term but cause fluid overload long-term.",
    "documented_failure": "SOFA/lactate shaping rewards are action-dependent and loopable. Aggressive IV fluids reduce SOFA short-term but cause fluid overload long-term. Agent could learn to oscillate: bolus -> SOFA drops -> reabsorbed -> SOFA rises -> bolus.",
    "failure_mechanism": "shaping_loop",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Use potential-based shaping or terminal-only reward to avoid short-term gaming",
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "clinical_risk": "Shaping reward on SOFA/lactate creates actionable clinical risk",
        "loopable_shaping": "Drug combinations can cycle SOFA scores over ~4h periods",
    },
}


# =========================================================================
# Reward config — single source of truth
# =========================================================================

SURVIVAL_REWARD = 15.0
DEATH_PENALTY = -15.0
SOFA_SHAPING_SCALE = 1.0
LACTATE_SHAPING_SCALE = 0.5

# Clinical parameters
SOFA_OSCILLATION_PERIOD = 4   # hours — SOFA can be gamed over ~4h cycles
LACTATE_OSCILLATION_PERIOD = 6  # hours — lactate responds slower
MAX_HOURS = 48                # typical ICU observation window
BASELINE_MORTALITY = 0.15     # ~15% ICU mortality for severe sepsis
SURVIVAL_RATE = 1 - BASELINE_MORTALITY

SOURCES = [
    RewardSource(
        name="survival",
        reward_type=RewardType.TERMINAL,
        value=SURVIVAL_REWARD,
        requires_action=True,
        discovery_probability=SURVIVAL_RATE,
    ),
    RewardSource(
        name="death",
        reward_type=RewardType.TERMINAL,
        value=DEATH_PENALTY,
        requires_action=False,
    ),
    RewardSource(
        name="sofa_decrease_shaping",
        reward_type=RewardType.SHAPING,
        value=SOFA_SHAPING_SCALE,
        state_dependent=True,
        requires_action=True,      # depends on drug choice
        can_loop=True,             # SOFA can be gamed
        loop_period=SOFA_OSCILLATION_PERIOD,
    ),
    RewardSource(
        name="lactate_decrease_shaping",
        reward_type=RewardType.SHAPING,
        value=LACTATE_SHAPING_SCALE,
        state_dependent=True,
        requires_action=True,
        can_loop=True,
        loop_period=LACTATE_OSCILLATION_PERIOD,
    ),
]


# =========================================================================
# The reward function
# =========================================================================

@reward_function(
    name="Sepsis Treatment RL",
    max_steps=MAX_HOURS,
    gamma=0.99,
    n_states=50000,
    n_actions=25,       # 5 IV fluid levels x 5 vasopressor levels
    action_type="discrete",
    death_probability=BASELINE_MORTALITY,
    sources=SOURCES,
    lr=3e-4,
    entropy_coeff=0.01,
    num_envs=16,
    total_steps=1_000_000,
)
def compute_reward(obs, action, info):
    """Sepsis treatment reward (simplified model of Raghu et al.)."""
    if info.get("died", False):
        return DEATH_PENALTY
    if info.get("discharged", False):
        return SURVIVAL_REWARD

    # Intermediate shaping: SOFA and lactate improvement
    sofa_delta = info.get("sofa_prev", 0) - info.get("sofa_current", 0)
    lactate_delta = info.get("lactate_prev", 0) - info.get("lactate_current", 0)

    return (SOFA_SHAPING_SCALE * sofa_delta
            + LACTATE_SHAPING_SCALE * lactate_delta)


def run_example():
    print("=" * 70)
    print("Sepsis Treatment RL — SOFA score gaming risk")
    print("=" * 70)
    print()
    print("Source: Raghu et al. 2017, Peng et al. 2023")
    print(f"Terminal: +{SURVIVAL_REWARD} survive, {DEATH_PENALTY} die")
    print(f"Shaping: SOFA (scale {SOFA_SHAPING_SCALE},"
          f" period {SOFA_OSCILLATION_PERIOD}h),"
          f" lactate (scale {LACTATE_SHAPING_SCALE},"
          f" period {LACTATE_OSCILLATION_PERIOD}h)")
    print()

    # The function works normally
    test_survive = compute_reward({}, 0, {"discharged": True})
    test_shaping = compute_reward(
        {}, 0, {"sofa_prev": 8, "sofa_current": 6,
                "lactate_prev": 3.0, "lactate_current": 2.5},
    )
    print(f"Test (discharged): {test_survive}")
    print(f"Test (SOFA 8→6, lactate 3→2.5): {test_shaping}")
    print()

    compute_reward.goodhart_check()

    print()
    print("The tool correctly identifies the clinical risks:")
    print()
    print("  1. shaping_not_potential_based: SOFA/lactate shaping")
    print("     depends on which drugs are given — the shaping can")
    print("     change the optimal treatment policy.")
    print()
    print("  2. Aggressive IV fluids reduce SOFA short-term (organs")
    print("     perfuse better) but cause fluid overload long-term.")
    print(f"     The {SOFA_OSCILLATION_PERIOD}-hour loop period means the")
    print("     agent could learn to oscillate: bolus → SOFA drops →")
    print("     reabsorbed → SOFA rises → bolus.")
    print()
    print("This is the highest-stakes domain for reward design.")


if __name__ == "__main__":
    run_example()
