"""Example: Unity 3D maze with noisy TV -- canonical curiosity trap.

Agent watches TV instead of solving maze because each channel change is maximally novel.
Source: Burda et al. 2019, "Large-Scale Study of Curiosity-Driven Learning" (ICLR)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "unity_noisy_tv",
    "source_paper": "Burda et al. 2019, 'Large-Scale Study of Curiosity-Driven Learning' (ICLR, arXiv:1808.04355)",
    "paper_url": "https://arxiv.org/abs/1808.04355",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2019,
    "domain": "navigation",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to solve a 9-room maze. Instead it parked at an in-world TV and changed channels, generating constant high intrinsic reward from novel images.",
    "documented_failure": "In-world TV generates maximally novel observations on every channel change. Curiosity bonus never decays for the TV because each image is unique. Agent parks at TV instead of exploring maze.",
    "failure_mechanism": "respawning_exploit",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "infinite_novelty": "TV generates maximally novel observations that never decay",
        "intrinsic_dominance": "Curiosity bonus competes with and dominates sparse goal reward",
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Unity 3D maze with noisy TV — the canonical curiosity trap")
    print("=" * 70)
    print()
    print("Source: Burda et al. 2019 (ICLR)")
    print("Sparse goal +1.0, curiosity coeff 0.01.")
    print("Agent watches TV instead of solving the maze.")
    print()

    model = EnvironmentModel(
        name="Unity Maze + Noisy TV",
        max_steps=1000,
        gamma=0.99,
        n_actions=5,  # forward, left, right, no-op, change_channel
        action_type="discrete",
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="maze_goal",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        requires_action=True,
        intentional=True,
        discovery_probability=0.05,
    ))
    model.add_reward_source(RewardSource(
        name="curiosity_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.01,
        respawn=RespawnBehavior.INFINITE,
        requires_action=True,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("The TV generates maximally novel observations on every")
    print("channel change. The curiosity bonus never decays for the TV")
    print("because each image is unique. The agent parks at the TV")
    print("and changes channels instead of exploring the maze.")


if __name__ == "__main__":
    run_example()
