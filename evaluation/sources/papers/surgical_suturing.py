"""SurgicAI da Vinci suturing benchmark (Wu et al. 2024, NeurIPS D&B).

Dense reward: -(d_trans/100 + d_angle/10). Hierarchical subtasks
for robotic suturing. Safety-critical surgical automation.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "surgical_suturing",
    "source_paper": (
        'Wu, Zhou, Kazanzides, Munawar & Liu, "SurgicAI: A Hierarchical '
        'Platform for Fine-Grained Surgical Policy Learning and '
        'Benchmarking," NeurIPS 2024 Datasets & Benchmarks'
    ),
    "paper_url": "https://arxiv.org/abs/2406.13865",
    "source_code_url": "https://github.com/surgical-robotics-ai/SurgicAI",
    "reward_location": (
        "Section 3.2, page 5: sparse reward (0/-1) and dense reward "
        "-(d_trans/100 + d_angle/10). Section 3.3: 6 subtasks with "
        "specific success thresholds (1-5mm distance, 10-20° angle). "
        "7-DOF continuous actions. da Vinci PSM compatible."
    ),
    "year": 2024,
    "domain": "healthcare",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Hierarchical RL benchmark for da Vinci surgical suturing. "
        "6 subtasks (grasp, place, insert, handoff, pullout). Dense "
        "reward tracks translation and orientation error. Safety-critical "
        "surgical automation on deformable tissue."
    ),
    "documented_failure": (
        "None — benchmark/platform paper. Provides standardized "
        "reward definitions and metrics for surgical subtasks."
    ),
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": None,
    "compute_cost_class": "medium",
    "is_negative_example": True,
    "encoding_rationale": {
        "dense_tracking": (
            "-(d_trans/100 + d_angle/10). All-negative tracking reward. "
            "Same structural pattern as nuclear/warfarin — gradient "
            "is informative (closer = less penalty). goodhart should "
            "fire negative_only_reward as WARNING."
        ),
        "hierarchical_subtasks": (
            "6 subtasks each with own reward/termination. Encoding "
            "the needle-grasping subtask as representative. The "
            "hierarchical structure is outside goodhart's scope."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Dense reward: -(d_trans/100 + d_angle/10)
    # Needle grasping subtask: success if distance<1mm, angle<10°

    model = EnvironmentModel(
        name="SurgicAI Needle Grasping (Wu et al. 2024)",
        max_steps=200,
        gamma=0.99,
        n_actions=7,  # 7-DOF: x,y,z,roll,pitch,yaw,jaw
        action_type="continuous",
        death_probability=0.0,
    )

    # Translation error penalty
    model.add_reward_source(RewardSource(
        name="translation_error",
        reward_type=RewardType.PER_STEP,
        value=-0.01,  # -d_trans/100, normalized
        requires_action=True,
        intentional=True,
        state_dependent=True,
        scales_with="distance",
        value_type="proportional",
    ))

    # Orientation error penalty
    model.add_reward_source(RewardSource(
        name="orientation_error",
        reward_type=RewardType.PER_STEP,
        value=-0.1,  # -d_angle/10, normalized
        requires_action=True,
        intentional=True,
        state_dependent=True,
        value_type="proportional",
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
