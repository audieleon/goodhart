"""Krakovna entry: Model-based planner exploiting learned world model.

Agent exploits systematic prediction errors in a learned dynamics
model to achieve high predicted (but not real) reward.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "krakovna_model_based_planner",
    "source_paper": (
        'Mishra et al 2017, "Prediction and Control with'
        ' Temporal Segment Models"'
    ),
    "paper_url": "https://arxiv.org/abs/1703.04070",
    "year": 2017,
    "domain": "control",
    "encoding_basis": "primary_source",
    "brief_summary": (
        "Reward structure is straightforward. Agent exploits"
        " systematic biases in a learned dynamics model to"
        " achieve high predicted but poor real reward."
    ),
    "documented_failure": (
        "Agent finds trajectories where model prediction"
        " errors are biased upward, yielding high predicted"
        " return but poor real-world performance."
    ),
    "failure_mechanism": "learned_reward",
    "detection_type": "specification",
    "is_negative_example": True,
    "compute_cost_class": "low",
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Model-based planner (Mishra et al 2017)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Model-Based Planner (Mishra et al 2017)",
        max_steps=200,
        gamma=0.99,
        n_actions=4,
        action_type="continuous",
    )

    # Per-step task reward — the actual reward signal is simple.
    # The problem is that the agent plans through a learned model
    # that can be exploited, not that the reward is misspecified.
    model.add_reward_source(RewardSource(
        name="task_reward",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
        max_occurrences=0,
    ))

    engine.print_report(model)

    print()
    print("Advisory: learned_reward — the reward function itself is not")
    print("misspecified. The agent exploits the learned dynamics model,")
    print("finding trajectories where model prediction errors are biased")
    print("upward. This is a model-based RL failure mode, not a reward")
    print("structure problem. Structural analysis cannot detect this.")


if __name__ == "__main__":
    run_example()
