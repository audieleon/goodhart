"""Example: Atari Road Runner level replay exploit.

Agent dies near end of level 1 to replay it and re-collect rewards instead of advancing.
Source: Saunders et al. 2017, "Trial without Error" (arXiv:1707.05173)
"""

from goodhart.models import *
from goodhart.engine import *
from goodhart.rules.reward import *
from goodhart.rules.training import *

METADATA = {
    "id": "road_runner_replay",
    "source_paper": "Saunders et al. 2017, 'Trial without Error: Towards Safe RL via Human Intervention'",
    "paper_url": "https://arxiv.org/abs/1707.05173",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2017,
    "domain": "game_ai",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to advance through levels. Instead it deliberately died near end of level 1 to replay and re-collect rewards.",
    "documented_failure": "Agent learned to deliberately die near end of level 1 to replay it and re-collect rewards, since die-and-replay EV (80*200=16000) far exceeds level completion EV (100).",
    "failure_mechanism": "respawning_loop",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Don't reset collectible rewards on death, or scale level completion reward to exceed total collectible value",
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "death_reset": "Collectibles reset on death enabling die-and-replay loop",
        "ev_imbalance": "Replaying level 1 is 160x more rewarding than advancing",
    },
}


def run_example():
    model = EnvironmentModel(
        name="Atari Road Runner",
        max_steps=10000,
        n_states=50000,
        n_actions=18,
        death_probability=0.001,
    )

    # Level completion reward (advances to harder level)
    model.add_reward_source(RewardSource(
        name="level completion",
        reward_type=RewardType.TERMINAL,
        value=100.0,
        discovery_probability=0.5,
    ))

    # Level 1 collectibles — THIS IS THE TRAP
    # These reset when the agent dies, allowing re-collection
    model.add_reward_source(RewardSource(
        name="level 1 collectibles",
        reward_type=RewardType.ON_EVENT,
        value=80.0,
        respawn=RespawnBehavior.ON_DEATH,  # resets on death
        discovery_probability=0.9,
    ))

    engine = TrainingAnalysisEngine().add_all_rules()
    engine.print_report(model)

    print("The framework catches this because:")
    print("  - Die-and-replay EV = 80 * 200 lives = +16,000")
    print("  - Level completion EV = +100")
    print("  - Replaying level 1 is 160x more rewarding")
    print()
    print("Fix: Don't reset collectible rewards on death.")
    print("Or: Scale level completion reward to exceed")
    print("     total collectible value (100 → 5000+).")


if __name__ == "__main__":
    run_example()
