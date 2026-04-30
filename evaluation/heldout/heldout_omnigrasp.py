"""Omnigrasp — held-out evaluation.

Luo et al. 2024 grasping reward where dense shaping is 30x the
sparse goal. Agent optimizes shaping instead of reaching the goal.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "heldout_omnigrasp",
    "source_paper": (
        'Luo et al. 2024, "Omnigrasp: Grasping Diverse Objects'
        ' with Simulated Humanoids," NeurIPS 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2407.00550",
    "source_code_url": None,
    "reward_location": "N/A",
    "year": 2024,
    "domain": "manipulation",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Agent was supposed to grasp diverse objects. Instead it"
        " optimizes the pre-grasp shaping signal (30x the goal)"
        " instead of reaching the actual grasp."
    ),
    "documented_failure": (
        "Pre-grasp guidance shaping (30.0) is 30x the grasp"
        " success goal (1.0). Dense shaping at this ratio makes"
        " goal discovery nearly impossible — the agent optimizes"
        " the shaping signal instead of reaching the actual goal."
        " Paper confirms needing pre-grasp guidance as a workaround."
    ),
    "failure_mechanism": "shaping_dominance",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Pre-grasp guidance workaround to bootstrap goal"
        " discovery past the shaping dominance."
    ),
    "compute_cost_class": "high",
    "is_negative_example": False,
    "encoding_rationale": {
        "shaping_ratio": (
            "30x shaping-to-goal ratio makes goal discovery"
            " nearly impossible."
        ),
        "can_loop": (
            "Pre-grasp shaping can loop (oscillate near object)"
            " without completing the grasp."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Omnigrasp — held-out evaluation")
    print("=" * 70)
    print()
    print("Source: Luo et al. 2024, NeurIPS 2024")
    print("Sparse goal + dense shaping 30x goal reward.")
    print("Paper uses pre-grasp guidance as a workaround.")
    print()

    model = EnvironmentModel(
        name="Omnigrasp (Held-Out)",
        max_steps=500,
        gamma=0.99,
        n_actions=52,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="grasp_success",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        requires_action=True,
        intentional=True,
        discovery_probability=0.02,
    ))
    model.add_reward_source(RewardSource(
        name="pre_grasp_guidance",
        reward_type=RewardType.SHAPING,
        value=30.0,
        requires_action=True,
        can_loop=True,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="approach_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        requires_action=True,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("The tool should flag that pre_grasp_guidance (30.0) is 30x")
    print("the grasp_success goal (1.0). Dense shaping at this ratio")
    print("makes goal discovery nearly impossible — the agent optimizes")
    print("the shaping signal instead of reaching the actual goal.")
    print("The paper confirms this by needing pre-grasp guidance as a")
    print("workaround for the sparse goal.")


if __name__ == "__main__":
    run_example()
