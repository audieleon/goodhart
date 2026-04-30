"""Examples: MuJoCo locomotion reward design.

The standard MuJoCo locomotion reward is a multi-component
design: forward_velocity - energy_cost + alive_bonus.

This example shows how the framework analyzes the interaction
between these components and predicts which degenerate
strategies are viable.

Covers:
- Walker2d alive bonus exploitation
- Hopper early termination incentive
- Humanoid energy minimization
- The general pattern: alive_bonus vs velocity tradeoff

Source: Todorov et al. 2012 (MuJoCo, IROS); Brockman et al. 2016
  (OpenAI Gym); Towers et al. 2023 (Gymnasium, arXiv:2407.17032)
"""

from goodhart.models import *
from goodhart.engine import *
from goodhart.rules.reward import *
from goodhart.rules.training import *


def run_example():
    print("=" * 70)
    print("MuJoCo LOCOMOTION — Reward component interactions")
    print("=" * 70)
    print()

    engine = TrainingAnalysisEngine().add_all_rules()

    # --- Standard MuJoCo reward decomposition ---
    print("Standard MuJoCo locomotion reward:")
    print("  R = velocity_reward - energy_penalty + alive_bonus")
    print()
    print("Three components, three potential exploits:")
    print("  1. Maximize alive_bonus by standing still")
    print("  2. Minimize energy_penalty by not moving")
    print("  3. Maximize velocity by exploiting physics")
    print()

    # Walker2d typical config
    print("--- Walker2d (typical configuration) ---")
    model = EnvironmentModel(name="Walker2d", max_steps=1000,
                             death_probability=0.02)
    model.add_reward_source(RewardSource(
        name="alive bonus", reward_type=RewardType.PER_STEP,
        value=1.0, respawn=RespawnBehavior.INFINITE,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="velocity reward", reward_type=RewardType.SHAPING,
        value=1.0, can_loop=False,
    ))
    model.add_reward_source(RewardSource(
        name="energy penalty", reward_type=RewardType.PER_STEP,
        value=-0.001,
    ))
    engine.print_report(model)

    # Hopper with termination penalty
    print("--- Hopper (with early termination) ---")
    print("Hopper falls easily. If alive_bonus is too high relative")
    print("to velocity reward, the agent learns to barely hop in")
    print("place rather than risk falling for forward progress.")
    print()

    model2 = EnvironmentModel(name="Hopper", max_steps=1000,
                              death_probability=0.05)
    model2.add_reward_source(RewardSource(
        name="alive bonus", reward_type=RewardType.PER_STEP,
        value=1.0, respawn=RespawnBehavior.INFINITE,
        requires_action=False,
    ))
    model2.add_reward_source(RewardSource(
        name="velocity", reward_type=RewardType.SHAPING,
        value=0.5,  # lower than alive bonus
        can_loop=False,
    ))
    engine.print_report(model2)

    print("When alive_bonus > velocity_reward, standing still")
    print("gives EV=+1000 (alive) while attempting forward motion")
    print("gives EV=+500 (velocity) - risk(falling)*1000 (lost alive).")
    print("The rational strategy depends on fall probability.")
    print()

    # The fix
    print("--- The Fix: Scale alive bonus below velocity ---")
    model3 = EnvironmentModel(name="Walker2d (fixed)", max_steps=1000,
                              death_probability=0.02)
    model3.add_reward_source(RewardSource(
        name="alive bonus", reward_type=RewardType.PER_STEP,
        value=0.1,  # reduced from 1.0
        respawn=RespawnBehavior.INFINITE,
        requires_action=False,
    ))
    model3.add_reward_source(RewardSource(
        name="velocity", reward_type=RewardType.SHAPING,
        value=1.0,
        can_loop=False,
    ))
    engine.print_report(model3)

    print("With alive_bonus << velocity_reward, standing still")
    print("gives only +100 while forward motion gives +1000.")
    print("The incentive to move is now dominant.")
    print()
    print("=== The general principle ===")
    print("In multi-component rewards, check that no single component")
    print("can be maximized by a degenerate strategy. The framework's")
    print("idle_exploit and respawning_exploit rules catch this.")


if __name__ == "__main__":
    run_example()
