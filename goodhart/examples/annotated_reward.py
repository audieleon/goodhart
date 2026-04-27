"""Example: Using the @reward_function decorator.

The recommended pattern: define reward constants ONCE, share them
between the decorator (structural analysis) and the function body
(runtime computation). No duplication, no drift.

Usage:
    # Python API
    from goodhart.examples.annotated_reward import compute_reward
    compute_reward.goodhart_check()          # print report
    assert compute_reward.goodhart_passed()  # CI gate

    # CLI
    goodhart --check goodhart.examples.annotated_reward:compute_reward
"""

from goodhart import reward_function, RewardSource, RewardType, RespawnBehavior


# =========================================================================
# Reward config — single source of truth
# =========================================================================
# Define constants once. Both the decorator and the function body use them.

VELOCITY_SCALE = 1.0
ALIVE_BONUS = 0.2
TORQUE_COST = -0.002

MAX_STEPS = 1000
GAMMA = 0.99
N_ACTIONS = 12

# Semantic metadata (can't be derived from the numbers alone —
# these describe WHAT the reward means, not just what it pays)
SOURCES = [
    RewardSource(
        name="velocity_tracking",
        reward_type=RewardType.PER_STEP,
        value=VELOCITY_SCALE,
        state_dependent=True,
        scales_with="velocity",
        requires_action=True,
        intentional=True,       # velocity IS the goal
    ),
    RewardSource(
        name="alive_bonus",
        reward_type=RewardType.PER_STEP,
        value=ALIVE_BONUS,
        respawn=RespawnBehavior.INFINITE,
        requires_action=False,
        intentional=False,      # staying alive is a means, not the end
    ),
    RewardSource(
        name="torque_penalty",
        reward_type=RewardType.PER_STEP,
        value=TORQUE_COST,
        requires_action=True,
    ),
]


# =========================================================================
# The reward function — uses the same constants
# =========================================================================

@reward_function(
    max_steps=MAX_STEPS,
    gamma=GAMMA,
    n_states=50000,
    n_actions=N_ACTIONS,
    action_type="continuous",
    death_probability=0.02,
    sources=SOURCES,
    lr=1e-3,
    entropy_coeff=0.01,
    num_envs=4096,
    total_steps=100_000_000,
)
def compute_reward(obs, action, info):
    """ANYmal-style quadruped locomotion reward."""
    velocity_error = abs(obs["velocity"] - obs["command_velocity"])
    tracking = VELOCITY_SCALE * (1.0 - velocity_error)
    alive = ALIVE_BONUS
    torque = TORQUE_COST * sum(a ** 2 for a in action)
    return tracking + alive + torque


def run_example():
    print("=" * 70)
    print("@reward_function decorator — single source of truth")
    print("=" * 70)
    print()
    print("Constants defined once, shared by decorator and function:")
    print(f"  VELOCITY_SCALE = {VELOCITY_SCALE}")
    print(f"  ALIVE_BONUS    = {ALIVE_BONUS}")
    print(f"  TORQUE_COST    = {TORQUE_COST}")
    print()
    print("The function works normally:")
    result = compute_reward(
        {"velocity": 1.0, "command_velocity": 1.5},
        [0.1, 0.2, 0.3],
        {},
    )
    print(f"  compute_reward(...) = {result:.4f}")
    print()
    print("And carries analysis metadata:")
    print(f"  compute_reward.goodhart_passed() = {compute_reward.goodhart_passed()}")
    print()
    print("Full report:")
    print()
    compute_reward.goodhart_check()


if __name__ == "__main__":
    run_example()
