"""Examples: Sparse reward environments across the literature.

Sparse reward is the most common source of degenerate policies.
The pattern is always the same: the goal reward is too far away
for random exploration to reach, so ANY step penalty creates an
incentive to die fast or stand still.

Covers:
- Montezuma's Revenge (Atari, 2016) — the task that spawned RND
- MiniHack MultiRoom (2021) — our failures
- Minecraft/Crafter exploration (2022)
- Sparse robotics manipulation (2020)

All use only built-in rules, no custom rules needed.

Source: Bellemare et al. 2013 (ALE/Montezuma, JAIR); Samvelyan et al.
  2021 (MiniHack, NeurIPS); Guss et al. 2019 (MineRL, NeurIPS)
"""

from goodhart.models import *
from goodhart.engine import *
from goodhart.rules.reward import *
from goodhart.rules.training import *


def run_example():
    print("=" * 70)
    print("SPARSE REWARD TRAPS — The universal failure mode")
    print("=" * 70)
    print()

    engine = AnalysisEngine().add_all_rules()

    # --- Montezuma's Revenge ---
    print("--- Montezuma's Revenge (Atari, 2016) ---")
    print("The game that exposed deep RL's exploration weakness.")
    print("Requires navigating multiple rooms, collecting keys,")
    print("avoiding enemies. Reward only for reaching new rooms.")
    print()

    model = EnvironmentModel(name="Montezuma's Revenge", max_steps=18000,
                             n_states=500000, death_probability=0.002)
    model.add_reward_source(RewardSource(
        name="room reward", reward_type=RewardType.ON_EVENT,
        value=100.0, max_occurrences=24,  # 24 rooms
        discovery_probability=0.001,  # very hard to find
    ))
    model.add_reward_source(RewardSource(
        name="key pickup", reward_type=RewardType.ON_EVENT,
        value=50.0, max_occurrences=6,
        discovery_probability=0.0005,
    ))
    engine.print_report(model)

    print("No step penalty means no degenerate equilibrium — but")
    print("the BudgetSufficiency rule catches the core issue:")
    print("random exploration won't discover goals in time.")
    print("This is why RND was invented for this specific game.")
    print()

    # --- Crafter / Minecraft-like ---
    print("--- Crafter (Hafner, 2022) ---")
    print("Minecraft-like survival with 22 achievement milestones.")
    print("Achievements are sparse but INCREMENTALLY discoverable.")
    print()

    model2 = EnvironmentModel(name="Crafter", max_steps=10000,
                              n_states=100000)
    model2.add_reward_source(RewardSource(
        name="achievements", reward_type=RewardType.ON_EVENT,
        value=1.0, max_occurrences=22,
        discovery_probability=0.1,  # easier than Montezuma
    ))
    engine.print_report(model2)

    print("Crafter avoids the sparse reward trap by having many")
    print("achievable milestones. Discovery probability is high")
    print("enough for the BudgetSufficiency rule to pass.")
    print()

    # --- Sparse robotics manipulation ---
    print("--- Sparse Robotics: Pick-and-Place (2020) ---")
    print("Robot must pick up object and place at target. Reward")
    print("only for successful placement. With a step penalty,")
    print("the robot learns to do nothing or knock object off table.")
    print()

    model3 = EnvironmentModel(name="Sparse Pick-and-Place",
                              max_steps=200, n_states=50000)
    model3.add_reward_source(RewardSource(
        name="successful placement", reward_type=RewardType.TERMINAL,
        value=1.0, discovery_probability=0.005,
    ))
    model3.add_reward_source(RewardSource(
        name="step penalty", reward_type=RewardType.PER_STEP,
        value=-0.01,
    ))
    engine.print_report(model3)

    print("Caught: penalty dominates goal, death beats survival,")
    print("exploration threshold impossible. The standard fix is")
    print("Hindsight Experience Replay (HER) which relabels failed")
    print("attempts as successes for nearby goals.")
    print()

    # --- The general principle ---
    print("=" * 70)
    print("THE UNIVERSAL SPARSE REWARD CHECK")
    print("=" * 70)
    print()
    print("For ANY sparse reward environment, compute:")
    print("  1. p(goal) — probability of reaching goal randomly")
    print("  2. penalty × max_steps — cost of a full episode")
    print("  3. If penalty × max_steps > goal × p(goal):")
    print("     → exploration is irrational")
    print("     → agent will learn degenerate policy")
    print("     → need: intrinsic motivation, HER, or curriculum")
    print()
    print("This check takes <1ms. Training takes hours to days.")
    print("Always run this check first.")


if __name__ == "__main__":
    run_example()
