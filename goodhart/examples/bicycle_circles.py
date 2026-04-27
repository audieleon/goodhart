"""Example: Bicycle agent riding in circles (Lilian Weng 2024).

Reward for decreasing distance to goal, but no penalty for
increasing distance. Agent rides tight circles to repeatedly
harvest the distance-decrease shaping reward.

Shows the shaping_loop_exploit rule and illustrates why
potential-based shaping (Ng et al. 1999) is the only safe
form of reward shaping.
"""

from goodhart.models import *
from goodhart.engine import *
from goodhart.rules.reward import *
from goodhart.rules.training import *


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
