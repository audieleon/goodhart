"""Min et al. 2019 highway driving — overtaking reward (Knox 2023).

Reward includes +0.5 for overtaking other vehicles and (v-40)/40
speed reward. Knox showed the overtaking bonus creates a shaping
incentive to weave between cars rather than drive safely.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "knox_min_highway",
    "source_paper": (
        'Knox, Allievi, Banzhaf, Schmitt & Stone, "Reward (Mis)design '
        'for Autonomous Driving," Artificial Intelligence 2023. '
        'Original reward from Min, Kim, Huh, "Deep Distributional RL '
        'Based High-Level Driving Policy Determination," IEEE T-IV 2019'
    ),
    "paper_url": "https://arxiv.org/abs/2104.13906",
    "source_code_url": None,
    "reward_location": (
        "Appendix A.8: 4-attribute unweighted sum. (1) (v-40)/40, "
        "speed in [40,80] km/h. (2) +0.5 overtaking bonus. (3) -0.25 "
        "lane change. (4) -10 collision. gamma=0.99. Episodic, no "
        "time limit, terminates on collision or track end (2500 units)."
    ),
    "year": 2023,
    "domain": "driving",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Highway driving with +0.5 overtaking bonus incentivizes "
        "weaving between cars. Knox identified this as learnable "
        "loophole reward shaping — the agent is rewarded for passing "
        "vehicles rather than for safe progress."
    ),
    "documented_failure": (
        "Knox Section 4.1: overtaking bonus (+0.5) is reward shaping "
        "that encourages aggressive driving behavior. The agent is "
        "incentivized to weave between vehicles even when unsafe. "
        "Section 4.4: identified as a learnable loophole."
    ),
    "failure_mechanism": "ontological_proxy",
    "detection_type": "structural",
    "discovery_stage": "post_training",
    "fix_known": (
        "Remove overtaking bonus. Use progress-toward-destination "
        "as the true reward instead of relative vehicle passing."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "overtaking_shaping": (
            "+0.5 for overtaking is shaping that rewards risky "
            "behavior. The agent can farm overtaking rewards by "
            "weaving between slower vehicles."
        ),
        "collision_vs_overtaking": (
            "Collision costs -10, overtaking earns +0.5. Agent "
            "needs only 20 successful overtakes to offset one "
            "collision — incentivizes aggressive driving."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    model = EnvironmentModel(
        name="Min 2019 Highway — Overtaking Exploit (Knox 2023)",
        max_steps=2500,
        gamma=0.99,
        n_states=100000,
        n_actions=5,
        action_type="discrete",
        death_probability=0.0,
    )

    # Speed: (v-40)/40, range [0, 1] for v in [40, 80] km/h
    model.add_reward_source(RewardSource(
        name="speed_reward",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Overtaking: +0.5 per overtake
    model.add_reward_source(RewardSource(
        name="overtaking_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.5,
        state_dependent=True,
        requires_action=True,
        intentional=False,
    ))

    # Lane change: -0.25
    model.add_reward_source(RewardSource(
        name="lane_change_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.25,
        state_dependent=False,
        requires_action=True,
        intentional=False,
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

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
