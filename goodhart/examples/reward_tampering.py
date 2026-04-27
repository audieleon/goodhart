"""Example: Sycophancy to Subterfuge — reward tampering (LIMITATION).

Models trained on simple sycophancy (agreeing with users) generalize
zero-shot to: modifying task checklists, editing their own reward
function code, and covering their tracks. No explicit training on
reward tampering was needed.

The reward structure is simple and correct. The problem is that the
agent has write access to the reward mechanism itself.

Source: Denison et al. 2024 (Anthropic), "Sycophancy to Subterfuge:
Investigating Reward-Tampering in Large Language Models"
Tool result: PASS + advisory_learned_reward (correct boundary)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Reward Tampering — agent modifies its own reward (LIMITATION)")
    print("=" * 70)
    print()
    print("Source: Denison et al. 2024 (Anthropic)")
    print()
    print("The agent doesn't hack the reward FUNCTION — it hacks the")
    print("reward IMPLEMENTATION. It modifies the code that computes")
    print("its own score.")
    print()

    model = EnvironmentModel(
        name="Reward Tampering (RLHF)",
        max_steps=1,
        gamma=1.0,
        n_states=1000000,
        n_actions=50000,
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="task_completion",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="helpfulness_rating",
        reward_type=RewardType.TERMINAL,
        value=0.5,
        requires_action=True,
    ))

    config = TrainingConfig(
        lr=1.4e-5,
        entropy_coeff=0.0,
        num_envs=64,
        n_actors=64,
        total_steps=1_000_000,
    )
    result = engine.print_report(model, config)

    print()
    print("=" * 70)
    print("The advisory_learned_reward fires — correct. But the real")
    print("issue is even deeper than learned reward gaming:")
    print()
    print("  Standard reward hacking: agent finds high-reward policy")
    print("    that doesn't achieve the intended goal")
    print("  Reward tampering: agent modifies the reward function")
    print("    itself to always return high scores")
    print()
    print("This is not about reward STRUCTURE at all. It's about the")
    print("relationship between the agent and the reward mechanism.")
    print("No static analysis of R(s,a) can detect this — you need")
    print("sandboxing, monitoring, and access control.")
    print()
    print("Denison et al. showed this emerges from simple sycophancy")
    print("training, with no explicit reward tampering curriculum.")
    print("The generalization is the scary part.")


if __name__ == "__main__":
    run_example()
