"""Eureka Ball Balance — GPT-4 generated ball balancing reward.

All three components are passive; doing nothing is globally optimal.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "eureka_ball_balance",
    "source_paper": (
        'Ma et al. 2024, "Eureka: Human-Level Reward Design via Coding'
        ' Large Language Models," ICLR 2024'
    ),
    "paper_url": "https://arxiv.org/abs/2310.12931",
    "source_code_url": (
        "https://eureka-research.github.io/assets/reward_functions/"
        "ball_balance.txt"
    ),
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "domain": "control",
    "brief_summary": (
        "GPT-4 generated Ball Balance reward. All components are"
        " passive — doing nothing maximizes every reward term."
    ),
    "documented_failure": (
        "All three reward components (velocity, force, surface)"
        " are maximized by inaction. Ball starts at rest on the"
        " surface, already saturating all terms. Idle EV ~179"
        " vs active EV ~40."
    ),
    "failure_mechanism": "idle_exploit",
    "discovery_stage": "during_training",
    "fix_known": (
        "Replace passive terms with active ones: reward ball"
        " movement toward target position instead of penalizing"
        " velocity/force."
    ),
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "velocity_reward": (
            "Passive (requires_action=False). Zero velocity"
            " gives maximum reward; any movement reduces it."
        ),
        "force_reward": (
            "Passive (requires_action=False). Zero actuator force"
            " gives maximum reward; any action reduces it."
        ),
        "surface_bonus": (
            "Passive (requires_action=False). Ball starts on"
            " surface; any action risks knocking it off."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Eureka Ball Balance — GPT-4 generated ball balancing reward")
    print("=" * 70)
    print()
    print("Source: Ma et al. 2024 (ICLR), Eureka project")
    print("GPT-4 generated reward for Isaac Gym Ball Balance task.")
    print("All three components are passive — doing nothing is optimal.")
    print()

    model = EnvironmentModel(
        name="Eureka Ball Balance",
        max_steps=500,
        gamma=0.99,
        n_actions=6,
        action_type="continuous",
    )
    model.add_reward_source(RewardSource(
        name="velocity_reward",
        reward_type=RewardType.PER_STEP,
        value=0.2,  # active: any movement increases velocity → less reward
        value_range=(0.0, 1.0),  # idle: zero velocity → max reward
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="force_reward",
        reward_type=RewardType.PER_STEP,
        value=0.1,  # active: actuators apply force → less reward
        value_range=(0.0, 0.5),  # idle: zero force → max reward
        requires_action=False,
        intentional=False,
    ))
    model.add_reward_source(RewardSource(
        name="surface_bonus",
        reward_type=RewardType.PER_STEP,
        value=0.1,  # active: risk knocking ball off surface
        value_range=(0.0, 0.3),  # idle: ball stays on surface
        requires_action=False,
        intentional=False,
    ))

    engine.print_report(model)

    print()
    print("Every component rewards passivity. Idle: value_range maxima")
    print("(1.0 + 0.5 + 0.3 = 1.8/step) give EV ~179. Any action")
    print("increases velocity (reducing velocity_reward to ~0.2),")
    print("applies force (reducing force_reward to ~0.1), and risks")
    print("knocking the ball off (reducing surface_bonus to ~0.1).")
    print("Active EV ~40. The ball starts at rest on the surface,")
    print("already saturating all three reward terms.")


if __name__ == "__main__":
    run_example()
