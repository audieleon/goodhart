"""Pan et al. 2022: COVID policy misweighting (health cost).

Proxy replaces INFECTION_SUMMARY_ABSOLUTE with threshold-based health
terms and reduces their weight, so the agent delays lockdowns and
allows excess infections.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "pan_covid_health",
    "source_paper": (
        'Pan, Bhatia & Steinhardt, "The Effects of Reward'
        ' Misspecification: Mapping and Mitigating Misaligned'
        ' Models," ICLR 2022'
    ),
    "paper_url": "https://arxiv.org/abs/2201.03544",
    "source_code_url": "https://github.com/aypan17/reward-misspecification",
    "reward_location": "Table 1 row 5; code: covid/rewards.py",
    "year": 2022,
    "domain": "policy",
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "COVID proxy replaces 10*INFECTION_SUMMARY_ABSOLUTE with"
        " weaker threshold terms (0.4*ABOVE_CAP + 1.0*ABOVE_3CAP)"
        " and keeps LOWER_STAGE + SMOOTH. Agent underpenalizes"
        " infections relative to economic disruption."
    ),
    "documented_failure": (
        "Agent delays lockdown measures because health cost is"
        " underweighted. Infections rise above safe thresholds"
        " while the agent optimizes for lower regulation stages."
    ),
    "failure_mechanism": "misweighting",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Use INFECTION_SUMMARY_ABSOLUTE with weight 10.0 and"
        " include POLITICAL term."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "requires_action": (
            "Policy actions (increase/decrease/maintain stage)"
            " affect infection dynamics indirectly."
        ),
        "intentional": (
            "Health and political costs are the true objectives;"
            " LOWER_STAGE and SMOOTH are regularizers."
        ),
        "can_loop": "No looping; sequential daily decisions.",
        "value": (
            "True: 10*INFECTION_ABSOLUTE + 10*POLITICAL"
            " + 0.1*LOWER_STAGE + 0.02*SMOOTH."
            " Proxy: 0.4*ABOVE_CAP + 1.0*ABOVE_3CAP"
            " + 0.1*LOWER_STAGE + 0.02*SMOOTH."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Pan et al. 2022: COVID Policy -- Misweighting (health cost)")
    print("=" * 70)

    # True: 10*INFECTION_ABSOLUTE + 10*POLITICAL + 0.1*LOWER_STAGE
    #       + 0.02*SMOOTH_STAGE_CHANGES
    # Proxy: 0.4*ABOVE_THRESHOLD(cap) + 1.0*ABOVE_THRESHOLD(3*cap)
    #        + 0.1*LOWER_STAGE + 0.02*SMOOTH
    # (omits INFECTION_ABSOLUTE and POLITICAL, replaces with weaker
    #  threshold-based health terms)
    # gamma=0.99, max_steps=192, n_actions=3, discrete

    model = EnvironmentModel(
        name="COVID Health Misweight (Pan et al. 2022)",
        max_steps=192,
        gamma=0.99,
        n_actions=3,
        action_type="discrete",
        death_probability=0.0,
    )

    # ABOVE_THRESHOLD(cap) -- weak health signal
    model.add_reward_source(RewardSource(
        name="above_capacity_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.4,
        requires_action=False,
        max_occurrences=0,
        state_dependent=True,
        scales_with="infections_above_cap",
        value_type="proportional",
        value_params={"scale": 0.4},
    ))

    # ABOVE_THRESHOLD(3*cap) -- severe overflow signal
    model.add_reward_source(RewardSource(
        name="above_3x_capacity_penalty",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
        requires_action=False,
        max_occurrences=0,
        state_dependent=True,
        scales_with="infections_above_3cap",
        value_type="proportional",
        value_params={"scale": 1.0},
    ))

    # LOWER_STAGE: prefer lower regulation stages (economic cost)
    model.add_reward_source(RewardSource(
        name="lower_stage",
        reward_type=RewardType.PER_STEP,
        value=0.1,
        requires_action=True,
        max_occurrences=0,
        state_dependent=True,
        scales_with="regulation_stage",
        value_type="proportional",
        value_params={"scale": 0.1},
    ))

    # SMOOTH_STAGE_CHANGES: penalize rapid policy swings
    model.add_reward_source(RewardSource(
        name="smooth_stage_changes",
        reward_type=RewardType.PER_STEP,
        value=0.02,
        requires_action=True,
        max_occurrences=0,
        state_dependent=True,
        scales_with="stage_stability",
        value_type="proportional",
        value_params={"scale": 0.02},
    ))

    # NOTE: True reward includes 10*INFECTION_SUMMARY_ABSOLUTE
    # and 10*POLITICAL, both omitted from this proxy.

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
