"""Example: Autonomous driving reward tradeoffs.

Crash penalties too small relative to speed reward cause agents to
weave aggressively and accept periodic collisions as optimal.

Source: Li et al. 2022 (MetaDrive, NeurIPS); Leurent 2018 (highway-env)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "driving_safety",
    "source_paper": "Li et al. 2022 (MetaDrive, NeurIPS); Leurent 2018 (highway-env)",
    "paper_url": "https://arxiv.org/abs/2109.12674",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2022,
    "domain": "driving",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to drive safely. Instead it weaves aggressively because 2.5 steps of speed reward offsets a crash penalty.",
    "documented_failure": "In highway-env, -1.0 crash penalty vs +0.4/step speed means 2.5 steps of driving offsets a crash. Agents learn aggressive weaving and accept periodic collisions as optimal.",
    "failure_mechanism": "penalty_dominance",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Increase crash penalty relative to accumulated progress reward, not just per-step reward",
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "crash_penalty_ratio": "Crash penalty -1.0 is offset by just 2.5 steps of +0.4 speed reward",
    },
}


def run_example():
    print("Driving Reward Safety Analysis")
    print("=" * 50)

    # --- highway-env ---
    highway_model = EnvironmentModel(
        name="highway-env (driving)", max_steps=40, gamma=0.8,
        n_states=10000, n_actions=5, action_type="discrete", death_probability=0.05,
    )
    highway_model.add_reward_source(RewardSource(name="collision", reward_type=RewardType.ON_EVENT, value=-1.0, state_dependent=True, requires_action=False, intentional=True))
    highway_model.add_reward_source(RewardSource(name="high_speed", reward_type=RewardType.PER_STEP, value=0.4, state_dependent=True, requires_action=True, intentional=True))
    highway_model.add_reward_source(RewardSource(name="right_lane", reward_type=RewardType.PER_STEP, value=0.1, state_dependent=True, requires_action=True, intentional=False))

    highway_config = TrainingConfig(
        algorithm="DQN", lr=5e-4, entropy_coeff=0.01, num_envs=4,
        n_actors=4, total_steps=100_000, replay_buffer_size=15000,
        target_update_freq=50,
    )

    # --- metadrive ---
    metadrive_model = EnvironmentModel(
        name="MetaDrive (multi-agent driving)", max_steps=1000, gamma=0.99,
        n_states=100000, n_actions=2, action_type="continuous", death_probability=0.02,
    )
    metadrive_model.add_reward_source(RewardSource(name="driving_reward", reward_type=RewardType.PER_STEP, value=1.0, state_dependent=True, requires_action=True, intentional=True))
    metadrive_model.add_reward_source(RewardSource(name="speed_reward", reward_type=RewardType.PER_STEP, value=0.1, state_dependent=True, requires_action=True, intentional=False))
    metadrive_model.add_reward_source(RewardSource(name="crash_penalty", reward_type=RewardType.ON_EVENT, value=-5.0, state_dependent=True, requires_action=False, intentional=True))
    metadrive_model.add_reward_source(RewardSource(name="out_of_road_penalty", reward_type=RewardType.ON_EVENT, value=-5.0, state_dependent=True, requires_action=True, intentional=True))
    metadrive_model.add_reward_source(RewardSource(name="arrive_dest", reward_type=RewardType.TERMINAL, value=10.0, requires_action=True, intentional=True))

    for model, config, source_label in [(highway_model, highway_config, "Leurent 2018"), (metadrive_model, None, "Li et al. 2021")]:
        engine = TrainingAnalysisEngine().add_all_rules()

        print(f"\n--- {model.name} ---")
        print(f"Source: {source_label}")
        for s in model.reward_sources:
            extra = ""
            if s.modifies:
                extra = f" [modifies: {s.modifies}]"
            if s.state_dependent:
                extra += " [state-dep]"
            print(f"  {s.name:20s} {s.value:+.1f} ({s.reward_type.value}){extra}")
        print()
        result = engine.analyze(model, config)
        for v in result.verdicts:
            icon = {"critical": "X", "warning": "!", "info": "i"}[v.severity.value]
            print(f"  [{icon}] {v.rule_name}: {v.message[:70]}")

    print()
    print("Key insight: crash penalties must be large relative to")
    print("accumulated progress reward, not just per-step reward.")
    print("A -5.0 crash is large per-step but small per-episode")
    print("when speed reward accumulates +0.4 * 1000 steps = 400.")


if __name__ == "__main__":
    run_example()
