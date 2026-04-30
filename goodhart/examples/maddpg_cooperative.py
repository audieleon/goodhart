"""Example: MADDPG cooperative navigation (MPE).

N agents cover N landmarks with distance-penalty reward. Shared reward
causes ambiguous credit assignment; individual reward causes competition.

Source: Lowe et al. 2017 (NeurIPS, MADDPG), Mordatch & Abbeel 2018
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "maddpg_cooperative",
    "source_paper": "Lowe et al. 2017 (NeurIPS, MADDPG); Mordatch & Abbeel 2018",
    "paper_url": "https://arxiv.org/abs/1706.02275",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2017,
    "domain": "multi_agent",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agents were supposed to cover landmarks cooperatively. Shared reward causes ambiguous credit assignment; individual reward causes competition for same landmark.",
    "documented_failure": "Distance-penalty reward with shared credit creates ambiguous credit assignment; with individual reward, agents may compete for the same landmark instead of covering all landmarks",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "MADDPG uses centralized critic with decentralized actors to address credit assignment",
    "compute_cost_class": "low",
    "is_negative_example": True,
    "encoding_rationale": {
        "credit_assignment": "Shared reward makes individual contribution ambiguous",
        "no_idle_exploit": "No passive alive bonus; standing still gives constant negative reward",
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("MADDPG Cooperative Navigation (MPE)")
    print("=" * 70)
    print()
    print("Source: Lowe et al. 2017 (NeurIPS)")
    print("N agents cover N landmarks. Reward = -sum(min_distance).")
    print("With shared reward, credit assignment is ambiguous.")
    print()

    # Model from the perspective of a single agent
    model = EnvironmentModel(
        name="MPE Cooperative Navigation",
        max_steps=25,
        gamma=0.95,
        n_states=1000,
        n_actions=5,
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="distance_penalty",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
        value_range=(-3.0, 0.0),
        state_dependent=True,
        scales_with="distance",
        requires_action=True,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="collision_penalty",
        reward_type=RewardType.ON_EVENT,
        value=-1.0,
        requires_action=False,
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=7e-4,
        entropy_coeff=0.01,
        num_envs=128,
        n_actors=128,
        total_steps=2_000_000,
    )
    engine.print_report(model, config)

    print()
    print("The cooperative navigation reward is purely distance-based.")
    print("Unlike locomotion tasks, there's no passive alive bonus —")
    print("standing still gives constant negative reward unless you")
    print("happen to start on a landmark. The tool correctly sees the")
    print("intentional distance penalty as the primary signal.")


if __name__ == "__main__":
    run_example()
