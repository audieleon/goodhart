"""Example: Montezuma's Revenge RND — skull dancing instead of progressing.

After collecting accessible rewards on the first level, the RND agent
"dances" with moving skulls, positions itself between laser gates,
and repeatedly mounts disappearing bridges. Dangerous states are
perpetually novel (rarely seen in past experience) and generate
sustained high intrinsic reward. The agent farms local danger
instead of coordinated key collection and room progression.

Source: Burda et al. 2019, "Exploration by Random Network
  Distillation" (ICLR, arXiv:1810.12894)
Tool should catch: intrinsic_dominance (WARNING, intrinsic competes)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


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
