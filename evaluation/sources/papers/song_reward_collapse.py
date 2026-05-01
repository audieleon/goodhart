"""Song et al. 2023 — reward collapse in LLM alignment (Princeton/Meta).

Ranking-based reward model training converges to degenerate
distributions where all outputs receive identical reward scores,
regardless of quality. This is an aggregation trap: the ranking
loss (Bradley-Terry) optimizes pairwise ordering but collapses
the absolute reward distribution.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "song_reward_collapse",
    "source_paper": (
        'Song, Namboodiri, Hutter & Lukasiewicz, "Reward Collapse '
        'in Aligning Large Language Models," arXiv 2023 (Princeton/'
        'Meta)'
    ),
    "paper_url": "https://arxiv.org/abs/2305.17608",
    "source_code_url": None,
    "reward_location": (
        "Section 3: ranking loss L = -E[log(sigma(r(y_w) - r(y_l)))] "
        "(Bradley-Terry). Theorem 1: closed-form expression for the "
        "collapsed reward distribution. Reward model r: Y → R maps "
        "outputs to scalar rewards. Training on ranking data causes "
        "r to converge to constant function."
    ),
    "year": 2023,
    "domain": "rlhf",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "Reward model for LLM alignment collapses during training: "
        "all outputs receive identical reward scores regardless of "
        "quality. The ranking-based loss preserves ordering but "
        "destroys reward magnitude information."
    ),
    "documented_failure": (
        "Theorem 1: reward model trained with ranking loss converges "
        "to a degenerate distribution where r(y) → constant for all "
        "y. The reward signal becomes uninformative for policy "
        "optimization, causing RLHF to fail silently."
    ),
    "failure_mechanism": "aggregation_trap",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Add reward regularization to prevent magnitude collapse. "
        "Use direct preference optimization (DPO) to avoid learning "
        "an explicit reward model."
    ),
    "compute_cost_class": "high",
    "is_negative_example": False,
    "encoding_rationale": {
        "advisory_learned_reward": (
            "The reward model is learned from human preferences. "
            "advisory_learned_reward should fire since the reward "
            "signal degrades during training."
        ),
        "aggregation_trap": (
            "The ranking loss aggregates pairwise comparisons into "
            "a scalar reward. This aggregation destroys magnitude "
            "information — advisory_aggregation_trap should fire."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Song et al. 2023: RLHF reward model collapse
    # LLM generates tokens, reward model scores output
    model = EnvironmentModel(
        name="RLHF Reward Collapse (Song et al. 2023)",
        max_steps=512,
        gamma=1.0,
        n_states=100000,
        n_actions=50000,
        action_type="discrete",
        death_probability=0.0,
    )

    # Learned reward model score (per-episode)
    # Small per-step value since total reward is per-output
    model.add_reward_source(RewardSource(
        name="reward_model_score",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    # KL penalty: -beta * KL(pi || pi_ref) per token
    model.add_reward_source(RewardSource(
        name="kl_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.01,
        state_dependent=True,
        requires_action=True,
        intentional=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
