"""LeTS-Drive (Cai et al. 2019, RSS) — undiscounted negative-only driving.

All rewards are negative: -0.1/step efficiency, -0.1 acceleration
smoothness, -1000*(v^2+0.5) collision. With gamma=1 (undiscounted),
the agent minimizes total cost. Knox et al. (2023) identified this
as having extreme risk tolerance via indifference point analysis.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "knox_cai_letsdrive",
    "source_paper": (
        'Knox, Allievi, Banzhaf, Schmitt & Stone, "Reward (Mis)design '
        'for Autonomous Driving," Artificial Intelligence 2023. '
        'Original reward from Cai, Luo, Saxena, Hsu, Lee, "LeTS-Drive: '
        'Driving in a Crowd by Learning from Tree Search," RSS 2019'
    ),
    "paper_url": "https://arxiv.org/abs/2104.13906",
    "source_code_url": None,
    "reward_location": (
        "Appendix A.1: 3-attribute unweighted sum. (1) [efficiency] "
        "-0.1 per non-terminal step (0 at terminal). (2) [smoothness] "
        "-0.1 if action includes non-zero acceleration. (3) [safety] "
        "-1000*(v^2+0.5) on collision, v in m/s. gamma=1, T=1200 "
        "(120s at 100ms steps). Figure 3: extreme risk tolerance — "
        "0.1 km per collision indifference point."
    ),
    "year": 2023,
    "domain": "driving",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "AD agent with all-negative reward: time penalty, acceleration "
        "penalty, and velocity-dependent collision penalty. Knox et al. "
        "showed this has extreme risk tolerance — the most risk-averse "
        "function they analyzed would still approve 2000x more crashes "
        "than drunk 16-17 year old US drivers."
    ),
    "documented_failure": (
        "Knox et al. Section 4.3, Figure 3: indifference point analysis "
        "shows this reward function has a collision rate of ~0.1 km per "
        "collision. For comparison, drunk US 16-17 year olds have ~10 km "
        "per collision. The reward is 100x less safe than drunk teens."
    ),
    "failure_mechanism": "ontological_proxy",
    "detection_type": "structural",
    "discovery_stage": "post_training",
    "fix_known": (
        "Knox et al. sanity check 3: compute indifference points and "
        "compare to human-derived risk tolerance. Increase collision "
        "penalty weight substantially."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "negative_only": (
            "All three reward terms are negative: -0.1/step, -0.1 "
            "acceleration, -1000*(v^2+0.5) collision. With gamma=1, "
            "the agent minimizes total cost. negative_only_reward "
            "should fire."
        ),
        "undiscounted": (
            "gamma=1 means no temporal discounting. Combined with "
            "the time penalty, the agent wants to terminate episodes "
            "as quickly as possible — which can mean crashing."
        ),
        "collision_velocity_dependence": (
            "Collision penalty -1000*(v^2+0.5) scales with velocity "
            "squared. At low speeds (v=1 m/s), penalty is -1500. "
            "At high speeds (v=10 m/s), penalty is -100500. This "
            "creates an incentive to drive slowly, not to avoid "
            "collisions entirely."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Cai et al. 2019: LeTS-Drive
    # gamma=1.0 (undiscounted), T=1200 (120s at 100ms), crowd navigation
    model = EnvironmentModel(
        name="LeTS-Drive Crowd Navigation (Cai 2019, via Knox 2023)",
        max_steps=1200,
        gamma=1.0,
        n_states=100000,
        n_actions=2,
        action_type="continuous",
        death_probability=0.0,
    )

    # Efficiency: -0.1 per non-terminal step
    model.add_reward_source(RewardSource(
        name="efficiency_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.1,
        state_dependent=False,
        requires_action=False,
        intentional=True,
    ))

    # Smoothness: -0.1 if non-zero acceleration
    model.add_reward_source(RewardSource(
        name="smoothness_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.1,
        state_dependent=False,
        requires_action=True,
        intentional=True,
    ))

    # Safety: -1000*(v^2+0.5) on collision
    # At typical speed v=5 m/s: -1000*(25+0.5) = -25500
    model.add_reward_source(RewardSource(
        name="collision_penalty",
        reward_type=RewardType.TERMINAL,
        value=-25500.0,
        state_dependent=True,
        requires_action=False,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
