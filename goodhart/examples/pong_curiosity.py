"""Example: Pong curiosity-only — agent maximizes bounces, not score.

With pure curiosity reward (no extrinsic), the agent learns to keep
rallies going indefinitely. Each bounce produces moderate novelty
(opponent moves, ball angle changes) while scoring ends the rally
and resets to a low-novelty state. The curiosity reward for sustained
play exceeds the implicit value of scoring.

Source: Burda et al. 2019, "Large-Scale Study of Curiosity-Driven
  Learning" (ICLR, arXiv:1808.04355)
Tool should catch: intrinsic_dominance (curiosity 10x goal)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


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
