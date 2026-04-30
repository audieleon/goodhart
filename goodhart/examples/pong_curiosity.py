"""Example: Pong curiosity-only — agent maximizes bounces, not score.

Pure curiosity reward makes the agent keep rallies going (each bounce
is novel) instead of scoring, which ends the rally and resets novelty.

Source: Burda et al. 2019, "Large-Scale Study of Curiosity-Driven
  Learning" (ICLR, arXiv:1808.04355)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "pong_curiosity",
    "source_paper": "Burda et al. 2019, Large-Scale Study of Curiosity-Driven Learning (ICLR)",
    "paper_url": "https://arxiv.org/abs/1808.04355",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2019,
    "domain": "game_ai",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to score points. Instead it keeps rallies going because each bounce is novel while scoring ends the rally.",
    "documented_failure": "Pure curiosity reward makes the agent keep rallies going indefinitely; each bounce produces moderate novelty while scoring ends the rally and resets to a low-novelty state; curiosity reward for sustained play exceeds the implicit value of winning",
    "failure_mechanism": "curiosity_trap",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Add extrinsic reward signal; decay intrinsic reward over time",
    "compute_cost_class": "medium",
    "is_negative_example": True,
    "encoding_rationale": {
        "intrinsic_dominance": "Curiosity bonus (0.1/step infinite) dominates sparse score events",
        "rally_as_novelty": "Sustained play generates more novelty than scoring and resetting",
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Pong — curiosity-only agent maximizes bounces, not score")
    print("=" * 70)
    print()
    print("Source: Burda et al. 2019 (ICLR)")
    print("Pure curiosity reward, no extrinsic signal.")
    print()

    model = EnvironmentModel(
        name="Pong Curiosity-Only",
        max_steps=18000,
        gamma=0.99,
        n_actions=6,
        action_type="discrete",
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="score",
        reward_type=RewardType.ON_EVENT,
        value=1.0,
        requires_action=True,
        intentional=True,
        discovery_probability=0.1,
    ))
    model.add_reward_source(RewardSource(
        name="curiosity_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.1,
        respawn=RespawnBehavior.INFINITE,
        requires_action=True,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("The agent keeps rallies going because each bounce is novel.")
    print("Scoring ends the rally (low novelty reset). Curiosity reward")
    print("for sustained play exceeds the value of winning points.")


if __name__ == "__main__":
    run_example()
