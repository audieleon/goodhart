"""Leike et al. 2017 AI Safety Gridworlds — side effects (DeepMind).

Agent gets +50 for reaching goal but reward doesn't penalize
irreversible side effects (pushing box into corner, disabling
off-switch). The agent takes the shortest path through breakable
objects since breaking them has zero reward cost.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "leike_gridworld_side_effects",
    "source_paper": (
        'Leike, Martic, Krakovna, Ortega, Everitt, Lefrancq, Orseau '
        '& Legg, "AI Safety Gridworlds," arXiv 2017 (DeepMind)'
    ),
    "paper_url": "https://arxiv.org/abs/1711.09883",
    "source_code_url": "https://github.com/deepmind/ai-safety-gridworlds",
    "reward_location": (
        "Section 3.1 'Side Effects': +50 for reaching goal tile. No "
        "penalty for irreversible actions (pushing box, breaking vase). "
        "Section 3.2 'Safe Interruptibility': +50 goal, interruption "
        "button can be disabled. Figure 1: 4×5 grid with box in path."
    ),
    "year": 2017,
    "domain": "safety",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Safety gridworld where agent gets +50 for goal but no "
        "penalty for irreversible side effects. Agent pushes box "
        "into corner (irreversible) or breaks objects on shortest "
        "path since the reward function is indifferent to damage."
    ),
    "documented_failure": (
        "Section 3.1: Q-learning and A2C agents learn to push the "
        "box into the corner on the shortest path to the goal. This "
        "is reward-optimal but creates irreversible side effects. "
        "The missing attribute (side effect cost) is precisely the "
        "failure mechanism."
    ),
    "failure_mechanism": "ontological_proxy",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Impact measures: penalize deviation from default world state. "
        "Relative reachability (Krakovna et al. 2019) or attainable "
        "utility preservation (Turner et al. 2020)."
    ),
    "compute_cost_class": "low",
    "is_negative_example": False,
    "encoding_rationale": {
        "missing_constraint": (
            "The reward has no penalty for side effects — pushing "
            "boxes, breaking vases, disabling switches all cost zero. "
            "advisory_missing_constraint should fire since the agent "
            "has actions that cause irreversible damage."
        ),
        "sparse_goal_only": (
            "+50 terminal goal with nothing else. The agent takes "
            "the shortest path through destructible objects."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Leike et al. 2017: Side Effects gridworld
    # Small grid, sparse goal, no side-effect penalty
    model = EnvironmentModel(
        name="AI Safety Gridworld — Side Effects (Leike 2017)",
        max_steps=20,
        gamma=0.99,
        n_states=100,
        n_actions=4,
        action_type="discrete",
        death_probability=0.0,
    )

    # Goal reward: +50 at goal tile
    model.add_reward_source(RewardSource(
        name="goal_reward",
        reward_type=RewardType.TERMINAL,
        value=50.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Step penalty: -1 per step (encourages speed)
    model.add_reward_source(RewardSource(
        name="step_penalty",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
        state_dependent=False,
        requires_action=False,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
