"""Example: Bicycle agent riding in circles.

Agent exploits one-directional distance-decrease shaping by orbiting
the goal, harvesting 25x more reward than actually reaching it.

Source: Randlov & Alstrom 1998 (ICML)
"""

from goodhart.models import *
from goodhart.engine import *
from goodhart.rules.reward import *
from goodhart.rules.training import *

METADATA = {
    "id": "bicycle_circles",
    "source_paper": "Randlov & Alstrom 1998, 'Learning to Drive a Bicycle using Reinforcement Learning and Shaping' (ICML)",
    "paper_url": None,
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 1998,
    "domain": "control",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to ride to the goal. Instead it rides in circles to farm distance-decrease shaping reward.",
    "documented_failure": "One-directional distance-decrease shaping reward (no penalty for increasing distance) lets the agent orbit the goal with a 4-step loop period, earning 25x more than the terminal goal reward.",
    "failure_mechanism": "shaping_loop",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Use potential-based shaping (Ng et al. 1999) where loops cancel out",
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "can_loop": "Distance decrease can be cycled: approach then pass then approach again",
        "loop_period_4": "4 steps per orbit cycle around the goal",
    },
}


def run_example():
    model = EnvironmentModel(
        name="Bicycle navigation (Weng 2024)",
        max_steps=1000,
        n_states=10000,
        n_actions=4,
    )

    # Terminal goal reward
    model.add_reward_source(RewardSource(
        name="reach goal",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        discovery_probability=0.05,
    ))

    # Distance-decrease shaping reward — THIS IS THE TRAP
    # Agent can orbit the goal, repeatedly decreasing distance
    # on one side and increasing on the other. The shaping reward
    # only fires on decrease, creating an exploitable cycle.
    model.add_reward_source(RewardSource(
        name="distance decrease",
        reward_type=RewardType.SHAPING,
        value=0.1,
        can_loop=True,       # can cycle: approach → pass → approach
        loop_period=4,        # 4 steps per orbit cycle
    ))

    engine = TrainingAnalysisEngine().add_all_rules()
    engine.print_report(model)

    print("The framework catches this because:")
    print("  - Cycling 'distance decrease' gives EV = 0.1 * 250 = +25")
    print("  - Reaching goal gives EV = +1.0")
    print("  - Orbiting is 25x more rewarding than completing the task")
    print()
    print("Fix: Use POTENTIAL-BASED shaping (Ng et al. 1999).")
    print("  Instead of: reward = (old_distance - new_distance)")
    print("  Use:        reward = gamma * potential(new) - potential(old)")
    print("  where potential(s) = -distance(s, goal)")
    print()
    print("Potential-based shaping is provably policy-invariant:")
    print("the optimal policy under shaped reward is the same as")
    print("under the original sparse reward. Loops cancel out")
    print("because the potential difference over a cycle is zero.")


if __name__ == "__main__":
    run_example()
