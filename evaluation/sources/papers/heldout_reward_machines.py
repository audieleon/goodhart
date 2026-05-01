"""Reward Machines — held-out evaluation (negative example).

Li et al. 2024 non-Markovian reward specification. Issues are
outside structural analysis scope — tool should pass.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "heldout_reward_machines",
    "source_paper": (
        'Li et al. 2024, "Reward Machines: Non-Markovian Reward'
        ' Specification," NeurIPS 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2406.00043",
    "source_code_url": None,
    "reward_location": "N/A",
    "year": 2024,
    "domain": "navigation",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Non-Markovian reward specification via automaton. Issues"
        " are about temporal dependencies, outside the scope of"
        " static structural analysis."
    ),
    "documented_failure": "None — reward is well-designed.",
    "failure_mechanism": None,
    "detection_type": "specification",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "low",
    "is_negative_example": True,
    "encoding_rationale": {
        "out_of_scope": (
            "Non-Markovian reward dependencies cannot be"
            " detected by static structural analysis."
        ),
        "expected_false_positive": (
            "Tool may flag exploration threshold as a warning"
            " since it cannot model the automaton's guided"
            " structure."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Reward Machines — held-out evaluation")
    print("=" * 70)
    print()
    print("Source: Reward Machines, NeurIPS 2024")
    print("Non-Markovian reward specification.")
    print("Issues are outside structural scope — tool should pass.")
    print()

    model = EnvironmentModel(
        name="Reward Machines (Held-Out)",
        max_steps=200,
        gamma=0.99,
        n_actions=4,
        action_type="discrete",
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="subtask_completion",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        requires_action=True,
        intentional=True,
        discovery_probability=0.5,
    ))
    model.add_reward_source(RewardSource(
        name="step_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.001,
        requires_action=False,
    ))

    engine.print_report(model)

    print()
    print("The tool passes (no criticals) but flags an exploration")
    print("threshold warning — it cannot model the reward machine's")
    print("automaton structure that guides the agent through subtasks.")
    print("This is a reasonable false positive: static analysis sees a")
    print("sparse goal but cannot know about the non-Markovian guidance.")
    print("The paper's actual issues involve non-Markovian reward")
    print("dependencies, which are outside what static structural")
    print("analysis can detect.")


if __name__ == "__main__":
    run_example()
