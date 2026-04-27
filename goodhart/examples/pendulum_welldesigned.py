"""Example: Pendulum-v1 — a well-designed all-penalty reward.

Pendulum uses r = -(theta^2 + 0.1*theta_dot^2 + 0.001*action^2).
The best possible reward is 0 (upright, zero velocity, zero torque).
The coefficient ratios encode priorities: angle matters 10x more
than velocity, velocity matters 100x more than energy.

This demonstrates the @reward_function decorator with the
single-source-of-truth pattern: constants defined once, shared
between the annotation and the reward computation.

This is a POSITIVE example — the tool gives a nearly-clean bill,
which is correct.

Source: Gymnasium documentation, Brockman et al. 2016
Tool result: PASS with minor dominance warning (correct)
"""

from goodhart import reward_function, RewardSource, RewardType, TrainingConfig


# =========================================================================
# Reward config — single source of truth
# =========================================================================

ANGLE_WEIGHT = -1.0       # -theta^2
VELOCITY_WEIGHT = -0.1    # -0.1 * theta_dot^2
TORQUE_WEIGHT = -0.001    # -0.001 * action^2

MAX_STEPS = 200
GAMMA = 0.99

SOURCES = [
    RewardSource(
        name="angle_penalty",
        reward_type=RewardType.PER_STEP,
        value=ANGLE_WEIGHT,
        value_range=(-3.14, 0.0),
        state_dependent=True,
        requires_action=False,
    ),
    RewardSource(
        name="velocity_penalty",
        reward_type=RewardType.PER_STEP,
        value=VELOCITY_WEIGHT,
        state_dependent=True,
        requires_action=False,
    ),
    RewardSource(
        name="torque_penalty",
        reward_type=RewardType.PER_STEP,
        value=TORQUE_WEIGHT,
        requires_action=True,
    ),
]


# =========================================================================
# The reward function — uses the same constants
# =========================================================================

@reward_function(
    name="Pendulum-v1",
    max_steps=MAX_STEPS,
    gamma=GAMMA,
    n_states=1000,
    n_actions=1,
    action_type="continuous",
    death_probability=0.0,
    sources=SOURCES,
    lr=3e-4,
    entropy_coeff=0.01,
    num_envs=16,
    total_steps=500_000,
)
def compute_reward(obs, action, _info):
    """Gymnasium Pendulum-v1 reward (exact implementation)."""
    theta, theta_dot = obs["theta"], obs["theta_dot"]
    torque = action[0]
    return (ANGLE_WEIGHT * theta ** 2
            + VELOCITY_WEIGHT * theta_dot ** 2
            + TORQUE_WEIGHT * torque ** 2)


def run_example():
    print("=" * 70)
    print("Pendulum-v1 — well-designed all-penalty reward (POSITIVE)")
    print("=" * 70)
    print()
    print("Source: Gymnasium, Brockman et al. 2016")
    print(f"r = {ANGLE_WEIGHT}*theta^2 + {VELOCITY_WEIGHT}*theta_dot^2"
          f" + {TORQUE_WEIGHT}*action^2")
    print("Best reward: 0 (upright, still, no torque)")
    print()
    print("Constants defined once, shared by decorator and function:")
    print(f"  ANGLE_WEIGHT    = {ANGLE_WEIGHT}")
    print(f"  VELOCITY_WEIGHT = {VELOCITY_WEIGHT}")
    print(f"  TORQUE_WEIGHT   = {TORQUE_WEIGHT}")
    print()

    # Function still works normally
    test = compute_reward({"theta": 0.5, "theta_dot": 1.0}, [0.5], {})
    print(f"Test call: compute_reward(theta=0.5, dot=1.0, torque=0.5) = {test:.4f}")
    print()

    # Run analysis
    result = compute_reward.goodhart_check()

    print()
    print("Clean bill (with a minor dominance warning). This is")
    print("correct — the reward is well-designed.")
    print()
    print("Why it works:")
    print("  1. All-penalty, zero is best — no idle exploit possible")
    print("     (standing still is NOT zero, the pendulum hangs down)")
    print("  2. Coefficient ratios encode priorities correctly:")
    print(f"     angle ({abs(ANGLE_WEIGHT)}) >> velocity ({abs(VELOCITY_WEIGHT)})"
          f" >> torque ({abs(TORQUE_WEIGHT)})")
    print("  3. No respawning, no looping, no terminal goals to miss")
    print()
    print("DESIGN PATTERN: When the task is 'maintain state X',")
    print("penalize deviation from X. When all penalties are negative")
    print("and the target state achieves zero penalty, there is no")
    print("degenerate strategy — every non-target state is strictly")
    print("worse. This is the safest reward design pattern.")


if __name__ == "__main__":
    run_example()
