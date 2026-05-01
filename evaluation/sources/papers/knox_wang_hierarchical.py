"""Wang et al. 2020 hierarchical driving — sparse + dense (Knox 2023).

Hierarchical RL for driving: high-level selects goals (terminal
rewards 100/-50/-10/-1), low-level tracks via dense cost-based
shaping. Knox noted only 1/10 focus papers specified full problem
details; this is one of the more complete specifications.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "knox_wang_hierarchical",
    "source_paper": (
        'Knox, Allievi, Banzhaf, Schmitt & Stone, "Reward (Mis)design '
        'for Autonomous Driving," Artificial Intelligence 2023. '
        'Original reward from Wang, Wang, Zhang, Yang, Xiong, '
        '"Learning Hierarchical Behavior and Motion Planning for '
        'AD," IROS 2020'
    ),
    "paper_url": "https://arxiv.org/abs/2104.13906",
    "source_code_url": None,
    "reward_location": (
        "Appendix A.9: Terminal rewards — +100 goal, -50 collision/"
        "timeout, -10 red light, -1 wrong lane. Non-terminal: "
        "negative sum of step costs (speed tracking, obstacle "
        "distance, lateral distance). gamma=0.99, dt~1s (average "
        "from motion planner). Episodic, CARLA."
    ),
    "year": 2023,
    "domain": "driving",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Hierarchical AD with large terminal rewards (+100 goal, "
        "-50 collision) and cost-based per-step shaping. Knox noted "
        "this has correct crash < idle ordering but extreme risk "
        "tolerance via indifference point analysis."
    ),
    "documented_failure": (
        "Knox Section 4.3, Figure 3: despite correct preference "
        "ordering, the indifference point shows the reward function "
        "approves driving with roughly 0.83 km per collision — far "
        "worse than any human driver category."
    ),
    "failure_mechanism": "ontological_proxy",
    "detection_type": "structural",
    "discovery_stage": "post_training",
    "fix_known": (
        "Knox: increase collision penalty relative to goal reward, "
        "or calibrate via indifference points against human risk "
        "tolerance benchmarks."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "correct_ordering_bad_calibration": (
            "G(crash) < G(idle) < G(success), but the margins are "
            "so tight that any small perturbation or discounting "
            "could flip the ordering. The risk tolerance is still "
            "100x worse than drunk teen drivers."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    model = EnvironmentModel(
        name="Wang 2020 Hierarchical AD (Knox 2023)",
        max_steps=500,
        gamma=0.99,
        n_states=100000,
        n_actions=5,
        action_type="discrete",
        death_probability=0.0,
    )

    # Goal reward: +100
    model.add_reward_source(RewardSource(
        name="goal_reward",
        reward_type=RewardType.TERMINAL,
        value=100.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Collision/timeout penalty: -50
    model.add_reward_source(RewardSource(
        name="collision_penalty",
        reward_type=RewardType.TERMINAL,
        value=-50.0,
        state_dependent=True,
        requires_action=False,
        intentional=True,
    ))

    # Red light violation: -10
    model.add_reward_source(RewardSource(
        name="red_light_penalty",
        reward_type=RewardType.PER_STEP,
        value=-10.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Wrong lane: -1
    model.add_reward_source(RewardSource(
        name="wrong_lane_penalty",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Step cost shaping (speed + obstacle + lateral)
    model.add_reward_source(RewardSource(
        name="step_cost_shaping",
        reward_type=RewardType.PER_STEP,
        value=-5.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
