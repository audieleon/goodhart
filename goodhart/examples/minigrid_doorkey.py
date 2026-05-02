"""Example: MiniGrid DoorKey — staged exploration with time pressure.

Three-stage exploration (find key, open door, reach goal) conflicts with
time pressure from R = 1 - 0.9*(steps/max_steps) penalizing slow discovery.

Source: Chevalier-Boisvert et al. 2023 (Minigrid & Miniworld)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "minigrid_doorkey",
    "source_paper": "Chevalier-Boisvert et al. 2023 (Minigrid & Miniworld)",
    "paper_url": "https://arxiv.org/abs/2306.13831",
    "source_code_url": "https://github.com/Farama-Foundation/Minigrid",
    "reward_location": "Reward structure from paper description",
    "year": 2023,
    "domain": "navigation",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to find key, open door, reach goal. Time pressure from step penalty conflicts with the exploration needed for staged discovery.",
    "documented_failure": "R = 1 - 0.9*(steps/max_steps) creates time pressure that conflicts with 3-stage exploration (find key, open door, reach goal); with 2% discovery probability, agent needs ~50 episodes to see a single success",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Time penalty does not make failure attractive, but slows learning by reducing goal reward at late steps",
    "compute_cost_class": "low",
    "is_negative_example": True,
    "encoding_rationale": {
        "staged_exploration": "Three prerequisite stages each with low discovery probability",
        "time_pressure_conflict": "Step penalty discourages the exploration needed to discover the goal",
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("MiniGrid DoorKey — staged exploration + time pressure")
    print("=" * 70)
    print()
    print("Source: Chevalier-Boisvert et al. 2023")
    print("R = 1 - 0.9*(steps/max_steps). Must find key, open door,")
    print("reach goal. 3-stage exploration problem.")
    print()

    model = EnvironmentModel(
        name="MiniGrid DoorKey-8x8",
        max_steps=300,
        gamma=0.99,
        n_states=5000,
        n_actions=7,
        death_probability=0.0,
        wall_probability=0.3,
    )
    model.add_reward_source(
        RewardSource(
            name="goal_reached",
            reward_type=RewardType.TERMINAL,
            value=1.0,
            requires_action=True,
            requires_exploration=True,
            discovery_probability=0.02,
        )
    )
    # Time pressure modeled as step penalty
    model.add_reward_source(
        RewardSource(
            name="time_penalty",
            reward_type=RewardType.PER_STEP,
            value=-0.003,  # 0.9/300 = -0.003/step
        )
    )

    config = TrainingConfig(
        algorithm="PPO",
        lr=7e-4,
        entropy_coeff=0.01,
        num_epochs=4,
        num_envs=16,
        n_actors=16,
        total_steps=5_000_000,
        rollout_length=128,
    )
    engine.print_report(model, config)

    print()
    print("With 2% discovery probability and 300 steps, the agent")
    print("needs ~50 episodes to see a single success. The time")
    print("penalty (-0.9 total) reduces the goal from 1.0 to 0.1")
    print("at the last step, but doesn't make failure attractive.")


if __name__ == "__main__":
    run_example()
