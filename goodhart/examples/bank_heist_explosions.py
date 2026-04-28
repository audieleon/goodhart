"""Example: Bank Heist — IDF curiosity farms dynamite explosions.

Bank Heist has naturally stochastic dynamics: dynamite explodes
unpredictably and banks regenerate randomly. The IDF curiosity
agent repeatedly enters and exits the maze while dropping dynamite.
The unpredictable visual novelty of explosions generates sustained
intrinsic reward. The agent is not trying to rob banks. It is
farming explosion novelty.

Source: Mavor-Parker et al. 2022, "How to Stay Curious while
  avoiding Noisy TVs" (ICML, arXiv:2102.04399)
Tool should catch: intrinsic_dominance (WARNING, curiosity competes)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


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
    model.add_reward_source(RewardSource(
        name="bank_robbery",
        reward_type=RewardType.ON_EVENT,
        value=10.0,
        requires_action=True,
        intentional=True,
        discovery_probability=0.05,
    ))
    model.add_reward_source(RewardSource(
        name="idf_curiosity",
        reward_type=RewardType.PER_STEP,
        value=0.05,
        respawn=RespawnBehavior.INFINITE,
        requires_action=True,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("Dynamite explosions are visually unpredictable (natural")
    print("stochasticity). IDF prediction error stays high for")
    print("explosions, generating sustained intrinsic reward.")
    print("The agent farms explosions instead of robbing banks.")


if __name__ == "__main__":
    run_example()
