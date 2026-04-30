"""Krakovna entry: Hide-and-seek physics exploits.

Reward structure is clean; agents discover physics engine bugs
(box surfing, ramp clipping) to win.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "krakovna_hide_and_seek",
    "source_paper": (
        'Baker et al 2020, "Emergent Tool Use from Multi-Agent'
        ' Autocurricula," ICLR 2020'
    ),
    "paper_url": "https://arxiv.org/abs/1909.07528",
    "year": 2020,
    "domain": "multi_agent",
    "encoding_basis": "catalog_derived",
    "brief_summary": (
        "Binary win/lose reward plus per-step survival. Reward"
        " structure is sound; exploits come from MuJoCo physics"
        " engine bugs, not reward misspecification."
    ),
    "documented_failure": (
        "Agents discover box surfing and ramp exploitation"
        " via physics engine bugs."
    ),
    "failure_mechanism": "physics_exploit",
    "detection_type": "dynamic",
    "is_negative_example": True,
    "compute_cost_class": "low",
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Krakovna: Hide-and-seek (Baker et al 2019)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Hide-and-Seek (Baker et al 2019)",
        max_steps=240,
        gamma=0.998,
        n_actions=6,
        action_type="continuous",
        death_probability=0.0,
    )

    # Binary win/lose at episode end.
    # Hiders win if not found; seekers win if they find hiders.
    model.add_reward_source(RewardSource(
        name="win_reward",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        intentional=True,
        discovery_probability=0.5,
    ))

    model.add_reward_source(RewardSource(
        name="lose_penalty",
        reward_type=RewardType.TERMINAL,
        value=-1.0,
        intentional=True,
        discovery_probability=0.5,
    ))

    # Per-step survival signal for hiders (rewarded each step hidden)
    model.add_reward_source(RewardSource(
        name="survival_per_step",
        reward_type=RewardType.PER_STEP,
        value=0.1,
        requires_action=True,
        intentional=True,
        max_occurrences=0,
    ))

    engine.print_report(model)

    print()
    print("Advisory: physics_exploit — the reward structure is sound for")
    print("the intended game. Box surfing and ramp exploitation are MuJoCo")
    print("dynamics bugs, not reward specification problems. The structural")
    print("encoding shows a clean competitive reward; what broke was the")
    print("physics engine, not the incentive design.")


if __name__ == "__main__":
    run_example()
