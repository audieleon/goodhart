"""Example: MuJoCo Humanoid-v4 idle exploit (Gymnasium).

Healthy_reward=5.0/step dwarfs velocity_reward~1.25/step, so the
rational agent stands still and earns 5000 vs ~6250 with fall risk.

Source: Todorov et al. 2012 (MuJoCo), Brockman et al. 2016 (Gymnasium)
"""

from goodhart import reward_function, RewardSource, RewardType, RespawnBehavior

METADATA = {
    "id": "humanoid_idle",
    "source_paper": "Todorov et al. 2012 (MuJoCo); Brockman et al. 2016 (Gymnasium)",
    "paper_url": "https://arxiv.org/abs/1606.01540",
    "source_code_url": None,
    "reward_location": "Gymnasium Humanoid-v4 default constants",
    "year": 2016,
    "domain": "locomotion",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to walk. Instead it stands still because healthy_reward (5.0/step) dwarfs velocity_reward (~1.25/step).",
    "documented_failure": "healthy_reward=5.0 per step with velocity reward averaging ~1.25; standing still earns 5000 over 1000 steps vs ~6250 with fall risk, making idle the rational strategy",
    "failure_mechanism": "idle_exploit",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Set healthy_reward=1.0 (Humanoid-v3 default) or use terminate_when_unhealthy=False with no healthy_reward",
    "compute_cost_class": "medium",
    "is_negative_example": True,
    "encoding_rationale": {
        "idle_exploit": "Passive healthy_reward dominates active velocity_reward by 4x",
        "gymnasium_defaults": "Constants match Gymnasium Humanoid-v4 defaults exactly",
    },
}


# =========================================================================
# Reward config — matches Gymnasium Humanoid-v4 defaults
# =========================================================================

HEALTHY_REWARD = 5.0      # per step, passive (Gymnasium default)
VELOCITY_SCALE = 1.25     # forward_reward_weight
CTRL_COST_WEIGHT = 0.1    # ctrl_cost_weight (applied to ||action||^2)
CONTACT_COST_WEIGHT = 5e-7  # contact_cost_weight
CONTACT_COST_RANGE = (-1.0, 0.0)

MAX_STEPS = 1000
GAMMA = 0.99

SOURCES = [
    RewardSource(
        name="healthy_reward",
        reward_type=RewardType.PER_STEP,
        value=HEALTHY_REWARD,
        respawn=RespawnBehavior.INFINITE,
        requires_action=False,
        intentional=False,  # NOT the goal — locomotion is
    ),
    RewardSource(
        name="forward_velocity",
        reward_type=RewardType.PER_STEP,
        value=VELOCITY_SCALE,
        value_range=(-1.0, 3.0),
        state_dependent=True,
        scales_with="velocity",
        requires_action=True,
        intentional=True,
    ),
    RewardSource(
        name="ctrl_cost",
        reward_type=RewardType.PER_STEP,
        value=-CTRL_COST_WEIGHT,
        requires_action=True,
    ),
    RewardSource(
        name="contact_cost",
        reward_type=RewardType.PER_STEP,
        value=-CONTACT_COST_WEIGHT,
        requires_action=False,
    ),
]


# =========================================================================
# The reward function — uses the same constants
# =========================================================================

@reward_function(
    name="Humanoid-v4",
    max_steps=MAX_STEPS,
    gamma=GAMMA,
    n_states=100000,
    n_actions=17,
    action_type="continuous",
    death_probability=0.03,
    sources=SOURCES,
    lr=3e-4,
    entropy_coeff=0.0,
    num_envs=1,
    total_steps=1_000_000,
    num_epochs=10,
    rollout_length=2048,
    minibatch_size=64,
)
def compute_reward(obs, action, info):
    """Gymnasium Humanoid-v4 reward (matches source)."""
    healthy = HEALTHY_REWARD if info.get("is_healthy", True) else 0.0
    velocity = VELOCITY_SCALE * obs.get("x_velocity", 0.0)
    ctrl = -CTRL_COST_WEIGHT * sum(a ** 2 for a in action)
    contact = max(CONTACT_COST_RANGE[0],
                  -CONTACT_COST_WEIGHT * obs.get("contact_force", 0.0))
    return healthy + velocity + ctrl + contact


def run_example():
    print("=" * 70)
    print("MuJoCo Humanoid-v4 — healthy_reward idle exploit")
    print("=" * 70)
    print()
    print("Source: Todorov et al. 2012, Brockman et al. 2016")
    print("Known issue: agent stands still because")
    print(f"  healthy_reward ({HEALTHY_REWARD}/step)"
          f" >> velocity_reward (~{VELOCITY_SCALE}/step)")
    print()
    print("Constants (from Gymnasium defaults):")
    print(f"  HEALTHY_REWARD     = {HEALTHY_REWARD}")
    print(f"  VELOCITY_SCALE     = {VELOCITY_SCALE}")
    print(f"  CTRL_COST_WEIGHT   = {CTRL_COST_WEIGHT}")
    print(f"  CONTACT_COST_WEIGHT = {CONTACT_COST_WEIGHT}")
    print()

    # Function works normally
    test = compute_reward(
        {"x_velocity": 1.0, "contact_force": 0.0},
        [0.1] * 17,
        {"is_healthy": True},
    )
    print(f"Test call (walking): {test:.4f}")
    idle = compute_reward(
        {"x_velocity": 0.0, "contact_force": 0.0},
        [0.0] * 17,
        {"is_healthy": True},
    )
    print(f"Test call (standing still): {idle:.4f}")
    print(f"Standing still earns {idle/test:.1f}x more than walking.")
    print()

    compute_reward.goodhart_check()

    print()
    print("The fix: set HEALTHY_REWARD = 1.0 (Humanoid-v3 default)")
    print("or use terminate_when_unhealthy=False with no healthy_reward.")


if __name__ == "__main__":
    run_example()
