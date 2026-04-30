"""Bimanual lid twisting (Lin et al. 2024, CoRL, UC Berkeley).

Three task rewards: twisting (rotation delta), finger contact
(keypoint proximity), pose (axis alignment) + regularization.
First sim-to-real bimanual dexterous manipulation with RL.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "bimanual_lidtwist",
    "source_paper": (
        'Lin, Yin, Qi, Abbeel & Malik, "Twisting Lids Off with Two '
        'Hands," CoRL 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2403.02338",
    "source_code_url": None,
    "reward_location": (
        "Section 3.3, page 5: (1) r_twisting = delta_theta (lid "
        "rotation per step). (2) r_contact = sum(1/(1+alpha*d(X,F))) "
        "keypoint-fingertip proximity. (3) r_pose = -arccos(axis·v) "
        "bottle alignment. (4) Regularizations: work + action penalty. "
        "PPO, 32 DOF (2 Allegro hands). Fig. 3 right: reward diagram."
    ),
    "year": 2024,
    "domain": "manipulation",
    "encoding_basis": "primary_source",
    "verification_date": "2026-05-01",
    "brief_summary": (
        "Two 16-DOF Allegro hands learn to twist lids off bottles. "
        "Three reward components: rotation progress (twisting), "
        "fingertip-to-keypoint proximity (contact), and bottle "
        "alignment (pose). Keypoint contact reward is the key insight "
        "enabling natural bimanual finger coordination. Sim-to-real."
    ),
    "documented_failure": (
        "Ablation (Fig. 4 left): disabling finger contact reward "
        "reduces performance dramatically — twisting alone is "
        "insufficient, the agent needs guidance on WHERE to contact "
        "the object. Reducing contact reward to 60% also degrades. "
        "Contact reward is necessary for eliciting desired behavior."
    ),
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "high",
    "is_negative_example": True,
    "encoding_rationale": {
        "three_task_rewards": (
            "Twisting (positive, rotation progress), contact "
            "(positive, proximity shaping), pose (negative, alignment "
            "error). All require action, all state-dependent."
        ),
        "contact_keypoint_novel": (
            "The finger contact reward using reference keypoints on "
            "bottle base/lid is the key contribution. Without it, "
            "the agent can't learn bimanual coordination."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # 3 task rewards + regularization
    # 32 DOF (2 × 16-DOF Allegro), PPO

    model = EnvironmentModel(
        name="Bimanual Lid Twisting (Lin et al. 2024)",
        max_steps=500,
        gamma=0.99,
        n_actions=32,  # 2 × 16 DOF Allegro hands
        action_type="continuous",
        death_probability=0.05,  # early termination on drop
    )

    # Twisting reward: delta_theta per step
    model.add_reward_source(RewardSource(
        name="twisting_progress",
        reward_type=RewardType.PER_STEP,
        value=1.0,  # rotation angle change
        requires_action=True,
        intentional=True,
        state_dependent=True,
    ))

    # Finger contact reward: keypoint proximity
    # sum(1/(1+alpha*d)) for each fingertip to reference points
    model.add_reward_source(RewardSource(
        name="finger_contact",
        reward_type=RewardType.PER_STEP,
        value=1.0,  # proximity shaping
        requires_action=True,
        state_dependent=True,
        scales_with="distance",
    ))

    # Pose reward: -arccos(axis alignment)
    model.add_reward_source(RewardSource(
        name="pose_alignment",
        reward_type=RewardType.PER_STEP,
        value=-0.5,  # negative when misaligned
        requires_action=True,
        state_dependent=True,
    ))

    # Regularization: work + action penalties
    model.add_reward_source(RewardSource(
        name="regularization",
        reward_type=RewardType.PER_STEP,
        value=-0.1,  # work + action smoothness
        requires_action=True,
        state_dependent=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
