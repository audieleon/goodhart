"""Kaufmann et al. 2023 — champion-level drone racing (Nature).

Swift drone beats human world champions. Reward combines progress
toward gate center with perception alignment. Well-designed reward
with careful curriculum. Deployed on physical racing drone.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "kaufmann_drone_racing",
    "source_paper": (
        'Kaufmann, Bauersfeld, Loquercio, Mueller, Koltun & '
        'Scaramuzza, "Champion-Level Drone Racing Using Deep '
        'Reinforcement Learning," Nature 2023'
    ),
    "paper_url": "https://www.nature.com/articles/s41586-023-06419-4",
    "source_code_url": None,
    "reward_location": (
        "Methods Section 'Reward Design': r = r_progress + "
        "r_perception. r_progress = progress toward the center of "
        "the next gate (proportional to velocity component along "
        "gate vector). r_perception encourages camera alignment "
        "with upcoming gates for visual detection. PPO, gamma=0.99."
    ),
    "year": 2023,
    "domain": "control",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Autonomous racing drone trained in simulation with reward "
        "combining progress toward gate centers and perception "
        "alignment for camera-based navigation. Beats 3 human world "
        "champions in physical races."
    ),
    "documented_failure": None,
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": None,
    "fix_known": None,
    "compute_cost_class": "high",
    "is_negative_example": True,
    "encoding_rationale": {
        "well_designed": (
            "Progress reward directly measures task objective (gate "
            "traversal speed). Perception reward is additive shaping "
            "for sim-to-real transfer. Successfully deployed."
        ),
        "curriculum": (
            "Training uses curriculum over track difficulty and "
            "opponent speed. The reward structure is fixed; only "
            "the environment complexity increases."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    model = EnvironmentModel(
        name="Swift Drone Racing (Kaufmann et al. 2023, Nature)",
        max_steps=500,
        gamma=0.99,
        n_states=100000,
        n_actions=4,
        action_type="continuous",
        death_probability=0.05,
    )

    # Progress toward next gate center
    model.add_reward_source(RewardSource(
        name="gate_progress",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Perception alignment (camera facing gates)
    model.add_reward_source(RewardSource(
        name="perception_alignment",
        reward_type=RewardType.PER_STEP,
        value=0.1,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Crash penalty (drone hits gate/obstacle)
    model.add_reward_source(RewardSource(
        name="crash_penalty",
        reward_type=RewardType.TERMINAL,
        value=-10.0,
        state_dependent=True,
        requires_action=False,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
