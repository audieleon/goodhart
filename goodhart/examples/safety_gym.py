"""Example: Safety Gym -- reward vs cost tradeoff.

Reward structure looks fine but constrained optimization changes the dynamics.
Source: Ray et al. 2019 (OpenAI Safety Gym)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "safety_gym",
    "source_paper": "Ray et al. 2019 (OpenAI Safety Gym)",
    "paper_url": "https://arxiv.org/abs/1910.01708",
    "source_code_url": "https://github.com/openai/safety-gym",
    "reward_location": "Reward structure from paper description",
    "year": 2019,
    "domain": "safety",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to reach goal while avoiding hazards. Hazard penalty modeled as reward penalty may create idle exploit (standing still avoids both hazards and progress).",
    "documented_failure": "goodhart does not model constraints as a separate channel. True constrained RL (CPO, FOCOPS) enforces hard cost limits, which changes optimal behavior vs soft penalties.",
    "failure_mechanism": "idle_exploit",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Model constrained RL separately with CostSource and constraint budgets",
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "cost_as_penalty": "Honest approximation of safety cost as reward penalty",
        "constraint_limitation": "True constrained RL treats costs differently than reward penalties",
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Safety Gym (PointGoal1) — reward vs constraint cost")
    print("=" * 70)
    print()
    print("Source: Ray et al. 2019 (OpenAI Safety Gym)")
    print("Reward: +1 for reaching goal. Cost: hazard entry = +1 cost.")
    print("We model the cost as a reward penalty (honest approximation).")
    print()
    print("LIMITATION: goodhart doesn't model constraints as a separate")
    print("channel. True constrained RL (CPO, FOCOPS) enforces hard")
    print("cost limits, which changes optimal behavior vs soft penalties.")
    print()

    model = EnvironmentModel(
        name="Safety Gym PointGoal1",
        max_steps=1000,
        gamma=0.99,
        n_states=100000,
        n_actions=2,
        death_probability=0.0,
    )
    model.add_reward_source(
        RewardSource(
            name="goal_reached",
            reward_type=RewardType.ON_EVENT,
            value=1.0,
            respawn=RespawnBehavior.ON_EPISODE,
            requires_action=True,
            requires_exploration=True,
            discovery_probability=0.1,
            intentional=True,
        )
    )
    model.add_reward_source(
        RewardSource(
            name="distance_shaping",
            reward_type=RewardType.SHAPING,
            value=0.01,
            state_dependent=True,
            scales_with="distance",
            requires_action=True,
            can_loop=False,
        )
    )
    # Model cost as penalty (honest approximation)
    model.add_reward_source(
        RewardSource(
            name="hazard_cost_as_penalty",
            reward_type=RewardType.ON_EVENT,
            value=-0.1,
            requires_action=False,
        )
    )

    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.01,
        num_envs=16,
        n_actors=16,
        total_steps=10_000_000,
    )
    engine.print_report(model, config)

    print()
    print("The tool correctly analyzes the reward structure, but misses")
    print("the constraint dimension. In real Safety Gym, the cost budget")
    print("(e.g., 25 hazard entries per episode) fundamentally changes")
    print("the feasible policy set. A future version could model")
    print("constrained RL separately.")


if __name__ == "__main__":
    run_example()
