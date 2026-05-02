"""Example: CoastRunners reward loop (Clark & Amodei 2016, OpenAI).

Agent goes in circles collecting respawning turbo powerups instead of
finishing the race, scoring 20% higher than humans who completed it.

Source: Clark & Amodei 2016, "Faulty Reward Functions in the Wild" (OpenAI Blog)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "coast_runners",
    "source_paper": "Clark & Amodei 2016, 'Faulty Reward Functions in the Wild' (OpenAI Blog)",
    "paper_url": "https://blog.openai.com/faulty-reward-functions/",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2016,
    "domain": "game_ai",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to finish the boat race. Instead it circles collecting respawning turbo powerups for higher score.",
    "documented_failure": "Agent learned to go in circles collecting respawning turbo powerups instead of finishing the race, scoring ~20% higher than human players who actually completed the course.",
    "failure_mechanism": "respawning_loop",
    "detection_type": "structural",
    "discovery_stage": "post_training",
    "fix_known": "Cap turbo powerups at a few occurrences or weight race completion higher",
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "respawning_turbo": "Turbo powerups respawn infinitely, creating a loopable reward source",
    },
}


def run_example():
    model = EnvironmentModel(
        name="CoastRunners (reward loop)",
        max_steps=2000,
        gamma=0.99,
        n_states=100000,
        n_actions=3,
        action_type="discrete",
        death_probability=0.01,
    )
    model.add_reward_source(
        RewardSource(
            name="finish race",
            reward_type=RewardType.TERMINAL,
            value=100.0,
            requires_action=True,
            intentional=True,
        )
    )
    model.add_reward_source(
        RewardSource(
            name="turbo powerup",
            reward_type=RewardType.ON_EVENT,
            value=20.0,
            respawn=RespawnBehavior.TIMED,
            respawn_time=5,
            can_loop=True,
            loop_period=10,
            requires_action=True,
            intentional=False,
        )
    )
    engine = TrainingAnalysisEngine().add_all_rules()

    print("CoastRunners Reward Analysis")
    print("=" * 50)
    print("Source: Clark & Amodei 2016 (OpenAI blog)")
    print("Reward sources:")
    for s in model.reward_sources:
        print(f"  {s.name}: {s.value:+.1f} ({s.reward_type.value})")
    print()

    engine.print_report(model)

    print("The framework catches this because looping the")
    print("turbo powerup earns far more than finishing the race.")
    print()
    print("Fix: Cap turbo powerups at a few occurrences,")
    print("or restructure scoring to weight race completion higher.")


if __name__ == "__main__":
    run_example()
