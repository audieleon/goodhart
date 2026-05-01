"""Pan et al. 2022: COVID policy ontological (missing political cost).

Proxy drops the POLITICAL term entirely. Agent imposes early strict
lockdowns that are epidemiologically optimal but politically infeasible.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "pan_covid_political",
    "source_paper": (
        'Pan, Bhatia & Steinhardt, "The Effects of Reward'
        ' Misspecification: Mapping and Mitigating Misaligned'
        ' Models," ICLR 2022'
    ),
    "paper_url": "https://arxiv.org/abs/2201.03544",
    "source_code_url": "https://github.com/aypan17/reward-misspecification",
    "reward_location": "Table 1 row 6; code: covid/rewards.py",
    "year": 2022,
    "domain": "policy",
    "encoding_basis": "code_derived",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "COVID proxy drops the POLITICAL term from the true reward."
        " Agent imposes early aggressive lockdowns that minimize"
        " infections but are politically unacceptable."
        " Exhibits a phase transition with model capacity."
    ),
    "documented_failure": (
        "Larger models enforce strict regulations preemptively."
        " ICU usage and infection stay low, but the policy is"
        " politically infeasible -- restrictions are imposed"
        " before public evidence of infection justifies them."
    ),
    "failure_mechanism": "ontological",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Include POLITICAL cost term with weight 10.0 in the"
        " reward function."
    ),
    "compute_cost_class": "medium",
    "is_negative_example": False,
    "encoding_rationale": {
        "requires_action": (
            "Policy actions (increase/decrease/maintain stage)"
            " affect infection dynamics indirectly."
        ),
        "intentional": (
            "INFECTION_ABSOLUTE is the health objective;"
            " POLITICAL is missing from proxy (the ontological gap)."
        ),
        "can_loop": "No looping; sequential daily decisions.",
        "value": (
            "True: 10*INFECTION_ABSOLUTE + 10*POLITICAL"
            " + 0.1*LOWER_STAGE + 0.02*SMOOTH."
            " Proxy: 10*INFECTION_ABSOLUTE + 0.1*LOWER_STAGE"
            " + 0.02*SMOOTH. (POLITICAL omitted.)"
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Pan et al. 2022: COVID Policy -- Ontological (no political)")
    print("=" * 70)

    # True: 10*INFECTION_ABSOLUTE + 10*POLITICAL + 0.1*LOWER_STAGE
    #       + 0.02*SMOOTH
    # Proxy: same but POLITICAL dropped entirely
    # gamma=0.99, max_steps=192, n_actions=3, discrete

    model = EnvironmentModel(
        name="COVID Missing Political (Pan et al. 2022)",
        max_steps=192,
        gamma=0.99,
        n_actions=3,
        action_type="discrete",
        death_probability=0.0,
    )

    # INFECTION_SUMMARY_ABSOLUTE (same as true reward)
    model.add_reward_source(RewardSource(
        name="infection_summary_absolute",
        reward_type=RewardType.PER_STEP,
        value=-10.0,
        requires_action=False,
        max_occurrences=0,
        state_dependent=True,
        scales_with="infections",
        value_type="proportional",
        value_params={"scale": 10.0},
        intentional=True,
    ))

    # NOTE: POLITICAL term is MISSING from this proxy.
    # True reward includes 10*POLITICAL, penalizing politically
    # unpopular restrictions. Without it, the agent freely
    # imposes aggressive early lockdowns.

    # LOWER_STAGE: prefer lower regulation stages
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

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
