"""CurriculumRL Mobile Robot — held-out evaluation.

Freitag et al. 2025. Mobile robot navigation with 5 reward terms.
Sparse goal (100) + dense auxiliary terms (action, velocity,
path progress, cross-track error). Two-stage curriculum.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "heldout_curriculum_rl",
    "source_paper": (
        'Freitag, Ceder, Laezza, Akesson & Haghir Chehreghani, '
        '"Curriculum Reinforcement Learning for Complex Reward '
        'Functions," EWRL 2025'
    ),
    "paper_url": "https://openreview.net/forum?id=DHOxjoy1sP",
    "source_code_url": None,
    "reward_location": (
        "Section 4.2, page 5: r_g = 100 (sparse goal, terminal). "
        "r_a = 1 - sum|a_i|, in [-1,1] (action smoothness). "
        "r_v = 1 - l2k(v-v_ref)*2/max(...), in [-1,1] (velocity "
        "tracking, piecewise quadratic, kappa=0.942). "
        "r_path = path progress shaping (Fig. 2a green). "
        "r_track = cross-track error penalty (Fig. 2a red). "
        "r = r_base + w*r_aux (Eq. 1), w in [0,1]. "
        "State: R^178 (lidar+pos+speed+path+goal). "
        "Action: [-1,1]^2 (trans+angular accel)."
    ),
    "year": 2025,
    "domain": "navigation",
    "encoding_basis": "primary_source",
    "verification_date": "2026-05-01",
    "brief_summary": (
        "Mobile robot navigates randomized maps to reach a goal "
        "while avoiding obstacles, tracking velocity, following "
        "a reference path, and keeping actions smooth. Sparse goal "
        "(100) with 4 dense auxiliary terms in [-1,1]. Two-stage "
        "curriculum: learn base reward first, then add auxiliaries."
    ),
    "documented_failure": (
        "Without curriculum: agent exploits auxiliary terms without "
        "completing the task (reward hacking). Especially at higher "
        "auxiliary weights (w=1.0), baseline gets stuck satisfying "
        "auxiliary objectives (smooth actions, good velocity) without "
        "reaching the goal. Curriculum fixes this by learning goal-"
        "reaching first, then adding behavioral objectives."
    ),
    "failure_mechanism": "compound_trap",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Two-stage curriculum: train on r_base only (goal + path "
        "progress) until converged, then switch to full r = r_base "
        "+ w*r_aux. Automatic switch based on critic convergence."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "sparse_goal_dominant": (
            "r_g = 100 is sparse (terminal) and much larger than "
            "dense terms (each in [-1,1]). But the dense terms "
            "are earned every step, so cumulative dense reward "
            "can dominate the sparse goal over long episodes."
        ),
        "auxiliary_exploitation": (
            "The paper documents that baseline agents 'exploit' "
            "auxiliary terms — satisfying smoothness and velocity "
            "while never reaching the goal. This is the compound "
            "trap: dense shaping dominates sparse goal."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Section 4.2: mobile robot, 5 reward terms
    # State R^178, Action [-1,1]^2, maps with obstacles

    model = EnvironmentModel(
        name="CurriculumRL Mobile Robot (Freitag et al. 2025)",
        max_steps=500,  # episode length
        gamma=0.99,
        n_actions=2,  # translational + angular acceleration
        action_type="continuous",
        death_probability=0.01,  # collision terminates
    )

    # Sparse goal: r_g = 100 if reached goal
    model.add_reward_source(RewardSource(
        name="goal_reward",
        reward_type=RewardType.TERMINAL,
        value=100.0,
        requires_action=True,
        intentional=True,
        discovery_probability=0.1,  # sparse, maps are randomized
    ))

    # Action smoothness: r_a = 1 - sum|a_i|, in [-1,1]
    model.add_reward_source(RewardSource(
        name="action_smoothness",
        reward_type=RewardType.PER_STEP,
        value=0.5,  # average value (range [-1,1])
        requires_action=True,
        state_dependent=True,
        value_range=(-1.0, 1.0),
    ))

    # Velocity tracking: r_v = piecewise quadratic, in [-1,1]
    # kappa=0.942, v_ref=1.2, v_max=5
    model.add_reward_source(RewardSource(
        name="velocity_tracking",
        reward_type=RewardType.PER_STEP,
        value=0.5,  # average (range [-1,1])
        requires_action=True,
        state_dependent=True,
        scales_with="velocity",
        value_range=(-1.0, 1.0),
    ))

    # Path progress: shaping toward goal along reference path
    model.add_reward_source(RewardSource(
        name="path_progress",
        reward_type=RewardType.SHAPING,
        value=0.5,  # from Fig 2a green curve
        requires_action=True,
        can_loop=False,
        intentional=False,
    ))

    # Cross-track error: penalty for deviation from reference path
    model.add_reward_source(RewardSource(
        name="cross_track_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.5,  # from Fig 2a red curve (range [-1,0])
        requires_action=True,
        state_dependent=True,
        value_range=(-1.0, 0.0),
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
