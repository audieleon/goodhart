"""Example: YouTube Watch Time -- respawning exploit at scale.

Infinite loopable engagement reward with no goal for user satisfaction or content quality.
Source: Covington et al. 2016 (RecSys), Ribeiro et al. 2023 (PNAS)
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine

METADATA = {
    "id": "youtube_watchtime",
    "source_paper": "Covington et al. 2016 (RecSys), 'Deep Neural Networks for YouTube Recommendations'; Ribeiro et al. 2023 (PNAS)",
    "paper_url": "https://dl.acm.org/doi/10.1145/2959100.2959190",
    "source_code_url": None,
    "reward_location": "Reward structure from paper description",
    "year": 2016,
    "domain": "industrial",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": "System was supposed to recommend satisfying content. Instead it maximized watch time by recommending progressively more extreme and outrage-inducing content.",
    "documented_failure": "Watch time is an infinite, loopable, non-intentional reward with no goal to compete against. Leads to recommending progressively more extreme content because outrage and controversy maximize engagement.",
    "failure_mechanism": "respawning_exploit",
    "detection_type": "structural",
    "discovery_stage": "post_training",
    "fix_known": "Add goal rewards for content quality/diversity/satisfaction, or use constrained optimization",
    "compute_cost_class": "high",
    "is_negative_example": False,
    "encoding_rationale": {
        "no_goal_reward": "No structural incentive for user satisfaction or content quality",
        "infinite_loopable": "Watch time is PER_STEP with INFINITE respawn and can_loop=True",
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("YouTube Watch Time — respawning exploit at billion-user scale")
    print("=" * 70)
    print()
    print("Source: Covington et al. 2016, Ribeiro et al. 2023")
    print("Reward: expected watch time per impression")
    print("Missing: any term for content quality or user wellbeing")
    print()

    model = EnvironmentModel(
        name="YouTube Watch Time Optimization",
        max_steps=10000,  # user session length
        gamma=0.99,
        n_states=10000000,  # content catalog
        n_actions=1000000,  # videos to recommend
        death_probability=0.0,  # session can continue indefinitely
    )
    model.add_reward_source(
        RewardSource(
            name="watch_time",
            reward_type=RewardType.PER_STEP,
            value=1.0,
            respawn=RespawnBehavior.INFINITE,
            can_loop=True,
            loop_period=1,
            requires_action=True,
            intentional=False,  # Watch time is a PROXY, not the real goal
        )
    )
    # No goal reward for user satisfaction, learning, or wellbeing.
    # That's the structural problem.

    config = TrainingConfig(
        lr=1e-4,
        entropy_coeff=0.001,
        num_envs=1024,
        n_actors=1024,
        total_steps=10_000_000_000,
    )
    engine.print_report(model, config)

    print()
    print("The tool correctly identifies the structural problem:")
    print("watch_time is an infinite, loopable, non-intentional reward")
    print("with no goal to compete against.")
    print()
    print("In practice, this meant:")
    print("  - Recommending progressively more extreme content")
    print("  - Outrage and controversy maximize engagement")
    print("  - No structural incentive to stop or diversify")
    print()
    print("The fix: add goal rewards for content quality, diversity,")
    print("user satisfaction surveys, or cap watch_time per session.")
    print("Or use constrained optimization (watch time subject to")
    print("quality constraints) instead of pure maximization.")
    print()
    print("This is the most impactful real-world Goodhart's Law")
    print("violation in existence — and the tool catches it from")
    print("the reward structure alone.")


if __name__ == "__main__":
    run_example()
