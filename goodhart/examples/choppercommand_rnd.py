"""Example: ChopperCommand — RND scores 4.7x worse than epsilon-greedy.

RND tuned for Montezuma's Revenge hurts on easy-exploration games;
the tool correctly does not flag this coefficient-mismatch problem.

Source: Taiga et al. 2021 (arXiv:2109.11052)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "choppercommand_rnd",
    "source_paper": "Taiga et al. 2021, 'On Bonus-Based Exploration Methods in the Arcade Learning Environment'",
    "paper_url": "https://arxiv.org/abs/2109.11052",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2021,
    "domain": "game_ai",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to shoot enemies. RND exploration bonus (tuned for Montezuma) added noise that hurt performance 4.7x vs epsilon-greedy.",
    "documented_failure": "RND coefficient tuned on Montezuma's Revenge (hard exploration) applied to ChopperCommand (easy exploration). The exploration bonus adds noise where task reward is already sufficient, scoring 2,675 vs epsilon-greedy's 12,578.",
    "failure_mechanism": None,
    "detection_type": "dynamic",
    "discovery_stage": "post_training",
    "fix_known": "Do not apply exploration bonuses to easy-exploration games; tune coefficients per environment",
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "coefficient_mismatch": "Structural ratio is below threshold; failure is from wrong-game tuning, not magnitude dominance",
    },
}


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
    model.add_reward_source(
        RewardSource(
            name="score",
            reward_type=RewardType.ON_EVENT,
            value=100.0,  # typical per-enemy score
            requires_action=True,
            intentional=True,
            discovery_probability=0.3,  # easy to find enemies
        )
    )
    model.add_reward_source(
        RewardSource(
            name="rnd_bonus",
            reward_type=RewardType.PER_STEP,
            value=0.005,  # beta=0.00005, typical RND ~0.1
            respawn=RespawnBehavior.INFINITE,
            requires_action=True,
            intentional=False,
        )
    )

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
