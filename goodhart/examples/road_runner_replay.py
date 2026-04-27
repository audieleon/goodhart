"""Example: Atari Road Runner level replay exploit.

The agent learned to deliberately die near the end of level 1
to replay it and re-collect rewards, instead of advancing to
harder levels with the same or lower reward density.

Shows the death_reset_exploit rule.
"""

from goodhart.models import *
from goodhart.engine import *
from goodhart.rules.reward import *
from goodhart.rules.training import *


def run_example():
    model = EnvironmentModel(
        name="Atari Road Runner",
        max_steps=10000,
        n_states=50000,
        n_actions=18,
        death_probability=0.001,
    )

    # Level completion reward (advances to harder level)
    model.add_reward_source(RewardSource(
        name="level completion",
        reward_type=RewardType.TERMINAL,
        value=100.0,
        discovery_probability=0.5,
    ))

    # Level 1 collectibles — THIS IS THE TRAP
    # These reset when the agent dies, allowing re-collection
    model.add_reward_source(RewardSource(
        name="level 1 collectibles",
        reward_type=RewardType.ON_EVENT,
        value=80.0,
        respawn=RespawnBehavior.ON_DEATH,  # resets on death
        discovery_probability=0.9,
    ))

    engine = TrainingAnalysisEngine().add_all_rules()
    engine.print_report(model)

    print("The framework catches this because:")
    print("  - Die-and-replay EV = 80 * 200 lives = +16,000")
    print("  - Level completion EV = +100")
    print("  - Replaying level 1 is 160x more rewarding")
    print()
    print("Fix: Don't reset collectible rewards on death.")
    print("Or: Scale level completion reward to exceed")
    print("     total collectible value (100 → 5000+).")


if __name__ == "__main__":
    run_example()
