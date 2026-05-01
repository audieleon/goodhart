"""Isele et al. 2018 intersection navigation — correct preference (Knox 2023).

One of only 2 reward functions that pass Knox's preference ordering:
crash < idle < success. Sparse reward: -0.01/step, -10 collision,
+1 goal. This is a WELL-DESIGNED reward for intersection navigation.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "knox_isele_intersection",
    "source_paper": (
        'Knox, Allievi, Banzhaf, Schmitt & Stone, "Reward (Mis)design '
        'for Autonomous Driving," Artificial Intelligence 2023. '
        'Original reward from Isele, Rahimi, Cosgun, Subramanian & '
        'Fujimura, "Navigating Occluded Intersections with '
        'Autonomous Vehicles Using Deep RL," ICRA 2018'
    ),
    "paper_url": "https://arxiv.org/abs/2104.13906",
    "source_code_url": None,
    "reward_location": (
        "Appendix A.5: 3-attribute unweighted sum. (1) -0.01 per "
        "step (time penalty). (2) -10 collision. (3) +1 at goal. "
        "gamma=0.99, T=100-300 (20-60s at 200ms). Unoccluded: 100 "
        "steps, occluded: 300 steps. Section 4.2: one of only 2 "
        "reward functions with correct preference ordering."
    ),
    "year": 2023,
    "domain": "driving",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "AD agent navigates occluded intersections with sparse "
        "reward: small time penalty, large collision penalty, goal "
        "bonus. Knox showed this is one of only 2/9 reward functions "
        "with correct preference ordering (crash < idle < success)."
    ),
    "documented_failure": None,
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": None,
    "fix_known": None,
    "compute_cost_class": "medium",
    "is_negative_example": True,
    "encoding_rationale": {
        "correct_preference_ordering": (
            "G(crash) = -0.01*T/2 - 10 ≈ -11.5. "
            "G(idle) = -0.01*T ≈ -3.0. G(success) = -0.01*T + 1 ≈ -2.0. "
            "crash < idle < success — correct ordering. One of only "
            "2/9 in Knox's survey to pass this check."
        ),
        "sparse_goal": (
            "+1 at goal is sparse but sufficient. The small time "
            "penalty (-0.01) doesn't overwhelm the collision penalty "
            "(-10), unlike most other driving rewards."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Isele et al. 2018: Occluded intersection
    # gamma=0.99, T=300 (occluded), 200ms steps
    model = EnvironmentModel(
        name="Isele 2018 Intersection — Correct Ordering (Knox 2023)",
        max_steps=300,
        gamma=0.99,
        n_states=100000,
        n_actions=5,
        action_type="discrete",
        death_probability=0.0,
    )

    # Time penalty: -0.01 per step
    model.add_reward_source(RewardSource(
        name="time_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.01,
        state_dependent=False,
        requires_action=False,
        intentional=True,
    ))

    # Collision: -10
    model.add_reward_source(RewardSource(
        name="collision_penalty",
        reward_type=RewardType.TERMINAL,
        value=-10.0,
        state_dependent=True,
        requires_action=False,
        intentional=True,
    ))

    # Goal: +1
    model.add_reward_source(RewardSource(
        name="goal_reward",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
