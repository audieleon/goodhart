"""Example: Montezuma's Revenge RND — skull dancing instead of progressing.

RND agent farms novelty from dangerous states (skulls, lasers, bridges)
instead of progressing, because rare deadly states are perpetually novel.

Source: Burda et al. 2019, "Exploration by Random Network
  Distillation" (ICLR, arXiv:1810.12894)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "montezuma_dancing",
    "source_paper": "Burda et al. 2019, Exploration by Random Network Distillation (ICLR)",
    "paper_url": "https://arxiv.org/abs/1810.12894",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2019,
    "domain": "game_ai",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to explore rooms. Instead it dances with skulls and lasers because dangerous states are perpetually novel.",
    "documented_failure": "RND agent farms novelty from dangerous states (skulls, laser gates, disappearing bridges) instead of coordinated key collection and room progression; dangerous states are rarely represented in past experience and generate sustained high intrinsic reward",
    "failure_mechanism": "curiosity_trap",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Separate intrinsic and extrinsic value heads with different discount factors (gamma_I=0.99, gamma_E=0.999)",
    "compute_cost_class": "high",
    "is_negative_example": True,
    "encoding_rationale": {
        "intrinsic_dominance": "RND intrinsic reward competes with sparse extrinsic score",
        "local_novelty_trap": "Dangerous states are locally novel but globally unproductive",
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Montezuma's Revenge — RND skull dancing")
    print("=" * 70)
    print()
    print("Source: Burda et al. 2019 (ICLR)")
    print("Agent dances with skulls for novelty instead of progressing.")
    print()

    # RND paper config: extrinsic coeff=2, intrinsic coeff=1
    model = EnvironmentModel(
        name="Montezuma's Revenge RND",
        max_steps=4500,  # ~18000 frames at 4 frame skip
        gamma=0.999,  # gamma_E from RND paper
        n_actions=18,
        action_type="discrete",
        death_probability=0.1,
    )
    model.add_reward_source(RewardSource(
        name="score",
        reward_type=RewardType.ON_EVENT,
        value=100.0,  # typical room clear reward
        requires_action=True,
        intentional=True,
        discovery_probability=0.01,  # very sparse
    ))
    model.add_reward_source(RewardSource(
        name="rnd_intrinsic",
        reward_type=RewardType.PER_STEP,
        value=0.5,  # intrinsic coeff=1, typical RND output ~0.5
        respawn=RespawnBehavior.INFINITE,
        requires_action=True,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("After collecting easy rewards, the agent farms novelty from")
    print("dangerous states (skulls, lasers, bridges) because they are")
    print("rarely represented in past experience. Global exploration")
    print("(coordinated key collection) generates less intrinsic reward")
    print("than locally available danger.")


if __name__ == "__main__":
    run_example()
