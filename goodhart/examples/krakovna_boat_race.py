"""Example: Krakovna's Boat Race — checkpoint cycling.

Agent cycles through 3 respawning checkpoints instead of completing
the 12-checkpoint course, earning 33x more than finishing the race.

Source: Krakovna et al. 2020 ("Specification Gaming: The Flip Side
  of AI Ingenuity"), DeepMind Safety Research blog
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "krakovna_boat_race",
    "source_paper": "Krakovna et al. 2020, Specification Gaming: The Flip Side of AI Ingenuity (DeepMind)",
    "paper_url": "https://arxiv.org/abs/2002.03469",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2020,
    "domain": "navigation",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to finish the race. Instead it cycles 3 checkpoints earning 33x more than completing the 12-checkpoint course.",
    "documented_failure": "Respawning checkpoints allow cycling: loop reward ~16.7/episode vs race finish expected value 0.5, a 33x advantage for cycling over racing",
    "failure_mechanism": "respawning_loop",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Make checkpoints one-time or require sequential completion",
    "compute_cost_class": "low",
    "is_negative_example": True,
    "encoding_rationale": {
        "respawning_exploit": "Timed checkpoint respawn creates exploitable loop with 30-step period",
        "specification_gaming": "From Krakovna's canonical specification gaming examples list",
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna Boat Race — checkpoint cycling exploit")
    print("=" * 70)
    print()
    print("Source: Krakovna et al. 2020 (Specification Gaming)")
    print("Agent goes in circles hitting 3 checkpoints repeatedly")
    print("instead of completing the 12-checkpoint course.")
    print()

    model = EnvironmentModel(
        name="Boat Race (checkpoint cycling)",
        max_steps=500,
        gamma=0.99,
        n_states=10000,
        n_actions=5,
        death_probability=0.0,
    )
    model.add_reward_source(
        RewardSource(
            name="checkpoint",
            reward_type=RewardType.ON_EVENT,
            value=1.0,
            respawn=RespawnBehavior.TIMED,
            respawn_time=10,
            can_loop=True,
            loop_period=30,  # 3 checkpoints * ~10 steps each
            requires_action=True,
        )
    )
    model.add_reward_source(
        RewardSource(
            name="finish_race",
            reward_type=RewardType.TERMINAL,
            value=10.0,
            requires_action=True,
            requires_exploration=True,
            discovery_probability=0.05,
        )
    )

    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.01,
        num_envs=16,
        n_actors=16,
        total_steps=1_000_000,
    )
    engine.print_report(model, config)

    print()
    print("The tool catches the respawning checkpoint exploit:")
    print("  Loop reward: 500/30 * 1.0 = ~16.7 per episode")
    print("  Race reward: 10.0 * 0.05 = 0.5 expected")
    print("Cycling dominates racing by 33x in expected value.")


if __name__ == "__main__":
    run_example()
