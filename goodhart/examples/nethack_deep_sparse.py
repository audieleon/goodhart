"""Example: NetHack — deep hierarchical credit assignment.

NetHack has 50+ levels, hundreds of item types, and extremely sparse
reward. The agent must learn a long chain: find food → eat to survive →
find weapons → fight monsters → find stairs → descend → ... → retrieve
the Amulet of Yendor. Each step is prerequisite for the next.

The exploration_threshold rule fires, but the advisory_credit_assignment
rule adds crucial context: this isn't just sparse, it's hierarchically
deep. Random exploration won't just be slow — it will never discover
the full task structure.

Source: Kuttler et al. 2020 (NeurIPS, NetHack Learning Environment)
Advisory fires: advisory_credit_assignment (deep sparse, no shaping)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


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
    model.add_reward_source(RewardSource(
        name="score",
        reward_type=RewardType.ON_EVENT,
        value=1.0,
        max_occurrences=0,
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.001,
    ))
    model.add_reward_source(RewardSource(
        name="amulet",
        reward_type=RewardType.TERMINAL,
        value=10000.0,
        requires_action=True,
        requires_exploration=True,
        discovery_probability=0.0000001,  # essentially zero for random
    ))

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
    result = engine.print_report(model, config)

    print()
    print("The advisory_credit_assignment fires here alongside")
    print("exploration_threshold. The key difference: exploration_threshold")
    print("says 'add intrinsic motivation.' The advisory says 'this might")
    print("be too deep for intrinsic motivation alone — consider curriculum")
    print("or hierarchical RL.' That distinction matters: RND helps with")
    print("Montezuma's Revenge but not with NetHack's full game.")


if __name__ == "__main__":
    run_example()
