"""Example: LunarLander-v2 — well-designed reward (Gymnasium).

Potential-based shaping (distance + velocity decrease) preserves optimal
policy by Ng 1999. Should pass clean with no structural reward traps.

Source: Brockman et al. 2016 (OpenAI Gym), Ng et al. 1999 (PBRS theory)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "lunar_lander",
    "source_paper": "Brockman et al. 2016 (OpenAI Gym); Ng et al. 1999 (PBRS theory)",
    "paper_url": "https://arxiv.org/abs/1606.01540",
    "source_code_url": None,
    "reward_location": "Gymnasium LunarLander-v2 source code",
    "year": 2016,
    "domain": "control",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Well-designed reward with potential-based shaping. Should pass clean with no structural traps.",
    "documented_failure": "N/A -- well-designed reward. Potential-based shaping (Ng 1999) preserves optimal policy and components are well-balanced",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "positive_example": "Demonstrates that the tool does not false-positive on good reward design",
        "pbrs_sound": "Distance and velocity shaping are potential-based (cycles net zero)",
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("LunarLander-v2 — potential-based shaping (should pass clean)")
    print("=" * 70)
    print()
    print("Source: Brockman et al. 2016, Ng et al. 1999")
    print("LunarLander uses distance-based shaping that is potential-based")
    print("(Phi = -distance - velocity), so it preserves optimal policy.")
    print()

    model = EnvironmentModel(
        name="LunarLander-v2",
        max_steps=1000,
        gamma=0.99,
        n_states=10000,
        n_actions=4,
        death_probability=0.1,
    )
    model.add_reward_source(RewardSource(
        name="distance_shaping",
        reward_type=RewardType.SHAPING,
        value=1.0,
        value_range=(-1.0, 1.0),
        state_dependent=True,
        scales_with="distance",
        requires_action=True,
        can_loop=False,  # potential-based: cycles net zero
    ))
    model.add_reward_source(RewardSource(
        name="velocity_shaping",
        reward_type=RewardType.SHAPING,
        value=0.5,
        state_dependent=True,
        scales_with="velocity",
        requires_action=True,
        can_loop=False,  # potential-based
    ))
    model.add_reward_source(RewardSource(
        name="landing_bonus",
        reward_type=RewardType.TERMINAL,
        value=100.0,
        requires_action=True,
        discovery_probability=0.3,
    ))
    model.add_reward_source(RewardSource(
        name="crash_penalty",
        reward_type=RewardType.ON_EVENT,
        value=-100.0,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="leg_contact",
        reward_type=RewardType.ON_EVENT,
        value=10.0,
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="fuel_cost",
        reward_type=RewardType.PER_STEP,
        value=-0.03,
        requires_action=True,
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.01,
        num_envs=16,
        n_actors=16,
        total_steps=1_000_000,
    )
    result = engine.print_report(model, config)

    print()
    if result.passed:
        print("As expected: LunarLander passes clean.")
        print("Potential-based shaping (Ng 1999) preserves optimal policy,")
        print("and the reward components are well-balanced.")
    else:
        print("Unexpected findings — investigate whether the model")
        print("accurately captures LunarLander's reward structure.")


if __name__ == "__main__":
    run_example()
