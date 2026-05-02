"""Example: PettingZoo adversarial — non-stationarity advisory.

Symmetric predator-prey rewards (+10 catch, -10 timeout) create
non-stationary dynamics as the opponent improves during self-play.

Source: Terry et al. 2021 (PettingZoo), inspired by Bansal et al. 2018
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "pettingzoo_adversarial",
    "source_paper": "Terry et al. 2021 (PettingZoo); Bansal et al. 2018",
    "paper_url": "https://arxiv.org/abs/2009.14471",
    "source_code_url": "https://github.com/Farama-Foundation/PettingZoo",
    "reward_location": "Reward structure from paper description",
    "year": 2021,
    "domain": "multi_agent",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "Predator-prey with symmetric rewards. Self-play creates non-stationary dynamics as the opponent improves during training.",
    "documented_failure": "Symmetric terminal rewards (+10 catch, -10 timeout) create competitive dynamics; in self-play the effective MDP shifts continuously as the prey improves, causing forgetting cycles and potential strategy collapse",
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": "Population-based training or league training to avoid co-adaptation collapse",
    "compute_cost_class": "low",
    "is_negative_example": True,
    "encoding_rationale": {
        "nonstationarity": "Symmetric rewards suggest competitive dynamics with shifting opponent",
        "advisory_fires": "advisory_nonstationarity fires on symmetric win/lose structure",
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("PettingZoo Simple Tag — adversarial non-stationarity")
    print("=" * 70)
    print()
    print("Source: Terry et al. 2021, Bansal et al. 2018")
    print("Predator +1 for catch, prey -1 for caught.")
    print("Symmetric rewards hint at competitive dynamics.")
    print()

    # Predator's perspective
    model = EnvironmentModel(
        name="Simple Tag (predator)",
        max_steps=100,
        gamma=0.99,
        n_states=10000,
        n_actions=5,
        death_probability=0.0,
    )
    model.add_reward_source(RewardSource(
        name="catch",
        reward_type=RewardType.TERMINAL,
        value=10.0,
        requires_action=True,
        discovery_probability=0.3,
        intentional=True,
    ))
    model.add_reward_source(RewardSource(
        name="timeout",
        reward_type=RewardType.TERMINAL,
        value=-10.0,
        requires_action=False,
    ))
    model.add_reward_source(RewardSource(
        name="distance_shaping",
        reward_type=RewardType.SHAPING,
        value=0.1,
        state_dependent=True,
        scales_with="distance",
        requires_action=True,
        can_loop=False,
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=3e-4,
        entropy_coeff=0.01,
        num_envs=8,
        n_actors=8,
        total_steps=5_000_000,
    )
    engine.print_report(model, config)

    print()
    print("The advisory_nonstationarity fires because +10 catch")
    print("and -10 timeout are roughly symmetric, suggesting")
    print("competitive dynamics. In self-play, the prey improves")
    print("while the predator trains — the effective MDP shifts")
    print("continuously. goodhart can't model this, but the hint")
    print("tells you to watch for forgetting cycles and collapse.")


if __name__ == "__main__":
    run_example()
