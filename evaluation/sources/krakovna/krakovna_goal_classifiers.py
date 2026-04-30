"""Krakovna entry: Goal classifiers -- robot arm exploits learned goal.

Agent fools a learned goal classifier by moving in ways that trigger
false positives instead of achieving the true goal.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "krakovna_goal_classifiers",
    "source_paper": (
        'Singh et al 2019, "End-to-End Robotic Reinforcement'
        ' Learning without Reward Engineering"'
    ),
    "paper_url": "https://arxiv.org/abs/1904.07854",
    "year": 2019,
    "domain": "manipulation",
    "encoding_basis": "primary_source",
    "brief_summary": (
        "Learned goal classifier provides the terminal signal."
        " Agent optimizes whatever triggers the classifier"
        " rather than the true goal."
    ),
    "documented_failure": (
        "Robot arm fools the learned goal classifier by"
        " moving in ways that trigger false positives."
    ),
    "failure_mechanism": "learned_reward",
    "detection_type": "specification",
    "is_negative_example": True,
    "compute_cost_class": "low",
    "paper_url": "https://arxiv.org/abs/1904.07854",
    "reward_location": "Singh et al. 2019 Section III: reward = log p_g(y=1|s), binary classifier on state-success pairs.",
    "verification_date": "2026-04-30",
    "discovery_stage": "during_training",
    "source_code_url": None,
    "fix_known": None,
    "encoding_rationale": {
        "basis": "Encoded from Krakovna specification gaming catalog with paper verification where available.",
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Goal classifiers (Singh 2019)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Goal Classifiers (Singh 2019)",
        max_steps=500,
        gamma=0.99,
        n_actions=7,
        action_type="continuous",
    )

    # Distance shaping toward the object target
    model.add_reward_source(RewardSource(
        name="distance_to_target",
        reward_type=RewardType.SHAPING,
        value=0.1,
        scales_with="distance",
        state_dependent=True,
        requires_action=True,
    ))

    # Learned goal classifier provides the terminal signal.
    # Structurally this looks like a normal terminal reward, but
    # the real problem is the classifier can be fooled.
    model.add_reward_source(RewardSource(
        name="learned_goal_signal",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        intentional=True,
        discovery_probability=0.3,
    ))

    engine.print_report(model)

    print()
    print("Advisory: learned_reward — the terminal goal is a learned")
    print("classifier, not ground truth. The agent exploits the classifier")
    print("by moving in ways that trigger false positives. This failure")
    print("mode is outside the scope of structural reward analysis.")


if __name__ == "__main__":
    run_example()
