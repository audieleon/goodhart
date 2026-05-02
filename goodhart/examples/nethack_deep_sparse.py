"""Example: NetHack — deep hierarchical credit assignment.

50+ levels with hierarchically deep prerequisites (food, weapons, stairs).
Not just sparse -- random exploration will never discover the full chain.

Source: Kuttler et al. 2020 (NeurIPS, NetHack Learning Environment)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "nethack_deep_sparse",
    "source_paper": "Kuttler et al. 2020 (NeurIPS, NetHack Learning Environment)",
    "paper_url": "https://arxiv.org/abs/2006.13760",
    "source_code_url": "https://github.com/facebookresearch/nle",
    "reward_location": "Reward structure from paper description",
    "year": 2020,
    "domain": "game_ai",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to retrieve the Amulet of Yendor. 50+ levels with hierarchically deep prerequisites make random exploration unable to discover the full task structure.",
    "documented_failure": "Extremely sparse reward across 50+ levels with hundreds of item types; hierarchically deep prerequisite chain (food -> weapons -> monsters -> stairs -> ... -> amulet); random exploration will never discover the full task structure",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Curriculum learning or hierarchical RL; RND alone is insufficient for this depth",
    "compute_cost_class": "high",
    "is_negative_example": True,
    "encoding_rationale": {
        "hierarchical_depth": "Not just sparse but hierarchically deep -- each step is prerequisite for the next",
        "advisory_distinction": "advisory_credit_assignment fires alongside exploration_threshold to flag depth",
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("NetHack — deep hierarchical credit assignment")
    print("=" * 70)
    print()
    print("Source: Kuttler et al. 2020 (NeurIPS)")
    print("50+ levels, hundreds of items, extremely sparse reward.")
    print("The task is hierarchically deep, not just sparse.")
    print()

    model = EnvironmentModel(
        name="NetHack (full game)",
        max_steps=100000,
        gamma=0.999,
        n_states=10000000,
        n_actions=77,
        death_probability=0.01,
    )
    model.add_reward_source(
        RewardSource(
            name="score",
            reward_type=RewardType.ON_EVENT,
            value=1.0,
            max_occurrences=0,
            requires_action=True,
            requires_exploration=True,
            discovery_probability=0.001,
        )
    )
    model.add_reward_source(
        RewardSource(
            name="amulet",
            reward_type=RewardType.TERMINAL,
            value=10000.0,
            requires_action=True,
            requires_exploration=True,
            discovery_probability=0.0000001,  # essentially zero for random
        )
    )

    config = TrainingConfig(
        algorithm="APPO",
        lr=2e-4,
        entropy_coeff=0.001,
        num_epochs=1,
        num_envs=256,
        num_workers=32,
        n_actors=8192,
        total_steps=10_000_000_000,
        use_rnn=True,
        rnn_type="lstm",
        rnn_size=512,
    )
    engine.print_report(model, config)

    print()
    print("The advisory_credit_assignment fires here alongside")
    print("exploration_threshold. The key difference: exploration_threshold")
    print("says 'add intrinsic motivation.' The advisory says 'this might")
    print("be too deep for intrinsic motivation alone — consider curriculum")
    print("or hierarchical RL.' That distinction matters: RND helps with")
    print("Montezuma's Revenge but not with NetHack's full game.")


if __name__ == "__main__":
    run_example()
