"""Kim et al. 2023 — constraints replace reward penalties for locomotion.

Demonstrates that converting 11 reward penalty terms to hard/soft
constraints reduces reward engineering from 10+ coefficients to 1.
Constraints use intuitive physical thresholds (e.g. 0.1m foot height)
instead of arbitrary penalty weights.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "kim_constraints_locomotion",
    "source_paper": (
        'Kim, Oh, Lee & Park, "Not Only Rewards But Also Constraints: '
        'Applications on Legged Robot Locomotion," IEEE T-RO 2024 '
        '(arXiv 2023)'
    ),
    "paper_url": "https://arxiv.org/abs/2308.12517",
    "source_code_url": None,
    "reward_location": (
        "Table I: r = r_c + r_tau + r_s. r_c = -k_c(||cmd_vxy - "
        "V_xy||^2 + (cmd_wz - w_z)^2) command tracking. r_tau = "
        "-k_tau||tau||^2 joint torque. r_s = -k_s(||q_t - q_{t-1}||^2 "
        "+ ||q_t - 2q_{t-1} + q_{t-2}||^2) smoothness. 11 constraints "
        "(6 probabilistic, 5 average) replace reward penalties: joint "
        "position/velocity/torque limits, body contact, COM frame, "
        "gait pattern, orthogonal velocity, contact velocity, foot "
        "clearance, foot height, symmetry."
    ),
    "year": 2023,
    "domain": "locomotion",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Legged robot reward engineering reduced from 10+ penalty terms "
        "to 3 rewards + 11 constraints. The key insight: physical "
        "limits (joint ranges, contact forces, foot clearance) should "
        "be constraints with intuitive thresholds, not penalties with "
        "arbitrary weights. advisory_constrained_rl should fire."
    ),
    "documented_failure": (
        "The paper documents that traditional reward-only approaches "
        "require extensive trial-and-error tuning of 10+ penalty "
        "coefficients, with small weight changes causing catastrophic "
        "policy changes. Converting to constraints eliminates this."
    ),
    "failure_mechanism": "ontological_proxy",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Convert safety/limit penalties to hard constraints using "
        "constrained policy optimization. Only 1 reward coefficient "
        "(torque scaling) needs tuning."
    ),
    "compute_cost_class": "high",
    "is_negative_example": True,
    "encoding_rationale": {
        "advisory_constrained_rl": (
            "The paper explicitly argues for constrained RL over "
            "reward penalties. The encoding shows the reward-only "
            "version (without constraints) which has many penalty "
            "terms that should be constraints. advisory_constrained_rl "
            "should fire since continuous control has few safety "
            "penalties relative to degrees of freedom."
        ),
        "reward_only_baseline": (
            "We encode the 3-term reward function (command tracking, "
            "torque, smoothness) without the 11 constraints, showing "
            "what goodhart would flag as missing constraints."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Kim et al. 2023: Legged robot with constraints
    # Multiple robots (Raibo, ANYmal, A1), PPO-Lagrangian
    model = EnvironmentModel(
        name="Constrained Locomotion (Kim et al. 2023)",
        max_steps=1000,
        gamma=0.99,
        n_states=100000,
        n_actions=12,
        action_type="continuous",
        death_probability=0.01,
    )

    # Command tracking: -k_c * (velocity error)
    model.add_reward_source(RewardSource(
        name="command_tracking",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # Joint torque penalty: -k_tau * ||tau||^2
    model.add_reward_source(RewardSource(
        name="torque_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.01,
        state_dependent=False,
        requires_action=True,
        intentional=True,
    ))

    # Action smoothness: -k_s * (action diff)
    model.add_reward_source(RewardSource(
        name="smoothness_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.1,
        state_dependent=False,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
