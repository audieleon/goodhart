"""Example: ChopperCommand — RND scores 4.7x worse than epsilon-greedy.

RND tuned for Montezuma's Revenge (hard exploration) actively hurts
on ChopperCommand (easy exploration). The exploration bonus adds
noise to a game where the task reward signal is already sufficient.
RND scored 2,675 vs epsilon-greedy's 12,578.

Source: Taiga et al. 2021, "On Bonus-Based Exploration Methods
  in the Arcade Learning Environment" (arXiv:2109.11052)
Tool should catch: the intrinsic coefficient is very small here
  (beta=0.00005), so the structural ratio is below threshold.
  This is a coefficient-mismatch problem (tuned on wrong game),
  not a magnitude-dominance problem. The tool correctly does NOT
  flag this — the failure requires knowing the coefficient was
  tuned elsewhere.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("ChopperCommand — RND hurts (4.7x worse than epsilon-greedy)")
    print("=" * 70)
    print()
    print("Source: Taiga et al. 2021")
    print("RND beta=0.00005, tuned for Montezuma's Revenge.")
    print("Scores 2,675 vs epsilon-greedy 12,578.")
    print()

    model = EnvironmentModel(
        name="ChopperCommand RND",
        max_steps=18000,
        gamma=0.99,
        n_actions=18,
        action_type="discrete",
        death_probability=0.05,
    )
    model.add_reward_source(RewardSource(
        name="score",
        reward_type=RewardType.ON_EVENT,
        value=100.0,  # typical per-enemy score
        requires_action=True,
        intentional=True,
        discovery_probability=0.3,  # easy to find enemies
    ))
    model.add_reward_source(RewardSource(
        name="rnd_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.005,  # beta=0.00005, typical RND ~0.1
        respawn=RespawnBehavior.INFINITE,
        requires_action=True,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("The RND coefficient is tiny (beta=0.00005), so goodhart")
    print("does not flag structural dominance. The failure is from")
    print("coefficient mismatch: tuned on Montezuma's (hard exploration)")
    print("and applied to ChopperCommand (easy exploration). Adding")
    print("ANY exploration bonus hurts when exploration is not the")
    print("bottleneck. This is beyond structural analysis.")


if __name__ == "__main__":
    run_example()
