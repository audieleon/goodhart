"""Example: Bank Heist — IDF curiosity farms dynamite explosions.

IDF curiosity agent farms explosion novelty from dynamite instead of
robbing banks, because stochastic visuals sustain intrinsic reward.

Source: Mavor-Parker et al. 2022 (ICML, arXiv:2102.04399)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "bank_heist_explosions",
    "source_paper": "Mavor-Parker et al. 2022, 'How to Stay Curious while avoiding Noisy TVs' (ICML)",
    "paper_url": "https://arxiv.org/abs/2102.04399",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2022,
    "domain": "game_ai",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to rob banks. Instead it farms dynamite explosion novelty for sustained intrinsic reward.",
    "documented_failure": "IDF curiosity agent repeatedly drops dynamite for visually unpredictable explosions, generating sustained intrinsic reward from natural stochasticity instead of pursuing the task objective of robbing banks.",
    "failure_mechanism": "idle_exploit",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Use aleatoric-uncertainty-aware curiosity to filter stochastic novelty",
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "idf_curiosity_per_step": "Intrinsic reward is sustained by natural stochasticity of explosions",
        "low_discovery_probability": "Bank robbery events are rare compared to easy explosion farming",
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Bank Heist — IDF curiosity farms dynamite explosions")
    print("=" * 70)
    print()
    print("Source: Mavor-Parker et al. 2022 (ICML)")
    print("Agent drops dynamite for explosion novelty, ignores banks.")
    print()

    model = EnvironmentModel(
        name="Bank Heist IDF Curiosity",
        max_steps=18000,
        gamma=0.99,
        n_actions=18,
        action_type="discrete",
        death_probability=0.05,
    )
    model.add_reward_source(
        RewardSource(
            name="bank_robbery",
            reward_type=RewardType.ON_EVENT,
            value=10.0,
            requires_action=True,
            intentional=True,
            discovery_probability=0.05,
        )
    )
    model.add_reward_source(
        RewardSource(
            name="idf_curiosity",
            reward_type=RewardType.PER_STEP,
            value=0.05,
            respawn=RespawnBehavior.INFINITE,
            requires_action=True,
            intentional=False,
        )
    )

    engine.print_report(model)

    print()
    print("Dynamite explosions are visually unpredictable (natural")
    print("stochasticity). IDF prediction error stays high for")
    print("explosions, generating sustained intrinsic reward.")
    print("The agent farms explosions instead of robbing banks.")


if __name__ == "__main__":
    run_example()
