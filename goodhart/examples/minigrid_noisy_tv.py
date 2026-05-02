"""Example: MiniGrid 6-room with noisy TV — 20x exploration collapse.

MSE curiosity agent collapses from 100+ novel states to 5 because the
noisy TV's irreducible prediction error dominates the exploration signal.

Source: Mavor-Parker et al. 2022, "How to Stay Curious while avoiding
  Noisy TVs using Aleatoric Uncertainty Estimation" (ICML, arXiv:2102.04399)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "minigrid_noisy_tv",
    "source_paper": "Mavor-Parker et al. 2022, How to Stay Curious while avoiding Noisy TVs using Aleatoric Uncertainty Estimation (ICML)",
    "paper_url": "https://arxiv.org/abs/2102.04399",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2022,
    "domain": "navigation",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Agent was supposed to explore 6 rooms. Instead it watches a noisy TV whose irreducible prediction error traps the curiosity signal.",
    "documented_failure": "MSE curiosity agent collapses from 100+ novel states to 5 because the noisy TV produces irreducible prediction error that dominates the exploration signal; agent stays near TV instead of entering new rooms",
    "failure_mechanism": "curiosity_trap",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Aleatoric uncertainty estimation to distinguish reducible from irreducible prediction error",
    "compute_cost_class": "low",
    "is_negative_example": True,
    "encoding_rationale": {
        "intrinsic_dominance": "Curiosity bonus (0.05/step infinite) dominates sparse goal (1.0 at 2% discovery)",
        "noisy_tv_problem": "Stochastic observations create irreducible prediction error that never decays",
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("MiniGrid 6-room + noisy TV — 20x exploration collapse")
    print("=" * 70)
    print()
    print("Source: Mavor-Parker et al. 2022 (ICML)")
    print("100+ novel states without noise, 5 with noisy TV.")
    print()

    model = EnvironmentModel(
        name="MiniGrid 6-Room + Noisy TV",
        max_steps=500,
        gamma=0.99,
        n_actions=7,
        action_type="discrete",
        death_probability=0.0,
    )
    model.add_reward_source(
        RewardSource(
            name="room_goal",
            reward_type=RewardType.TERMINAL,
            value=1.0,
            requires_action=True,
            intentional=True,
            discovery_probability=0.02,
        )
    )
    model.add_reward_source(
        RewardSource(
            name="mse_curiosity",
            reward_type=RewardType.PER_STEP,
            value=0.05,
            respawn=RespawnBehavior.INFINITE,
            requires_action=True,
            intentional=False,
        )
    )

    engine.print_report(model)

    print()
    print("The noisy TV produces irreducible prediction error.")
    print("MSE curiosity never decays for stochastic observations.")
    print("The agent earns more from watching TV than from exploring")
    print("new rooms, collapsing from 100+ to 5 novel states visited.")


if __name__ == "__main__":
    run_example()
