"""Example: RND intrinsic motivation — non-episodic design insight.

RND (Random Network Distillation) adds intrinsic reward for visiting
novel states. A critical design choice: intrinsic reward is NON-EPISODIC
(doesn't reset on death). If it reset, the agent would learn to die
deliberately to re-explore already-visited states.

The tool flags RND's intrinsic reward as a respawning exploit (it's
infinite and always available). This is technically correct but
misleading — intrinsic rewards are DESIGNED to be infinite. The tool
can't yet distinguish "infinite reward by design" from "infinite
reward by accident."

Source: Burda et al. 2019 (ICLR), "Exploration by Random Network
Distillation"
Tool result: FAIL — respawning_exploit on intrinsic (correct math,
  but misleading interpretation)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("RND Intrinsic Motivation — non-episodic design")
    print("=" * 70)
    print()
    print("Source: Burda et al. 2019 (ICLR)")
    print("Key insight: intrinsic reward must NOT reset on death")
    print()

    model = EnvironmentModel(
        name="Montezuma + RND",
        max_steps=18000,
        gamma=0.999,
        n_states=1000000,
        n_actions=18,
        death_probability=0.1,
    )
    model.add_reward_source(RewardSource(
        name="game_score",
        reward_type=RewardType.ON_EVENT,
        value=100.0,
        max_occurrences=0,
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.001,
    ))
    model.add_reward_source(RewardSource(
        name="rnd_intrinsic",
        reward_type=RewardType.PER_STEP,
        value=0.01,
        respawn=RespawnBehavior.INFINITE,
        requires_action=True,
        intentional=False,
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=1e-4,
        entropy_coeff=0.001,
        num_envs=128,
        n_actors=128,
        total_steps=10_000_000_000,
    )
    result = engine.print_report(model, config)

    print()
    print("The respawning_exploit fires on rnd_intrinsic. The math")
    print("is correct: 0.01 * 18000 = 180 intrinsic reward per episode")
    print("vs 100 * 0.001 = 0.1 expected game score. But this is")
    print("INTENTIONAL — intrinsic rewards are supposed to be the")
    print("dominant signal early in training.")
    print()
    print("The real design insight is about EPISODIC vs NON-EPISODIC:")
    print()
    print("  EPISODIC intrinsic (bad): intrinsic reward resets on death.")
    print("  Agent learns to die → re-explore → earn intrinsic again.")
    print("  This is a death_reset_exploit on the intrinsic reward.")
    print()
    print("  NON-EPISODIC intrinsic (good): intrinsic reward persists")
    print("  across episodes. Already-visited states give no bonus.")
    print("  Dying wastes time without recovering intrinsic reward.")
    print()
    print("RND uses non-episodic intrinsic with a SEPARATE value head")
    print("and a shorter discount factor (gamma_int=0.99 vs gamma_ext=0.999).")
    print("This prevents the intrinsic signal from dominating long-term.")
    print()
    print("LIMITATION: goodhart can't yet distinguish designed-infinite")
    print("rewards from accidental-infinite rewards. A future rule could")
    print("check for intentional=True on infinite sources.")


if __name__ == "__main__":
    run_example()
