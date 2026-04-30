"""Pan et al. 2022: Traffic Merge -- Misweighting (lane changes).

Source: Pan, Bhatia, & Steinhardt, "The Effects of Reward
  Misspecification: Mapping and Mitigating Misaligned Models,"
  ICLR 2022.
Failure: Agent switches lanes excessively because the lane-change
  penalty is underweighted in the proxy reward.
Mechanism: Misweighting -- the penalty for lane changes is too small
  relative to the commute-time benefit of opportunistic lane
  switching, so the agent treats lane changes as nearly free.
Domain: Autonomous driving / traffic merge (FLOW simulator)

Encoding: PROXY reward (what the agent optimizes).
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Pan et al. 2022: Traffic Merge -- Misweighting (lane changes)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Traffic Merge - Lane Misweight (Pan et al. 2022)",
        max_steps=1500,
        gamma=0.99,
        n_actions=2,
        action_type="continuous",
        death_probability=0.0,
    )

    # Commute time penalty
    model.add_reward_source(RewardSource(
        name="commute_time_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.1,
        requires_action=False,
        max_occurrences=0,
        intentional=True,
    ))

    # Lane change penalty -- MISWEIGHTED (underpenalized)
    # True weight would be -1.0; proxy uses -0.1
    model.add_reward_source(RewardSource(
        name="lane_change_penalty",
        reward_type=RewardType.ON_EVENT,
        value=-0.1,
        requires_action=True,
        max_occurrences=0,
    ))

    # Successful merge completion
    model.add_reward_source(RewardSource(
        name="merge_complete",
        reward_type=RewardType.TERMINAL,
        value=10.0,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
