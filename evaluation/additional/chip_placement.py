"""Google chip placement (Mirhoseini et al. 2021, Nature).

Terminal-only reward: -Wirelength - lambda*Congestion at final
placement step. Density enforced via hard action mask. Deployed
for Google TPU tapeout. Well-designed negative example.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "chip_placement",
    "source_paper": (
        'Mirhoseini, Goldie et al., "A Graph Placement Methodology '
        'for Fast Chip Design," Nature 594, 207-212, 2021'
    ),
    "paper_url": "https://arxiv.org/abs/2004.10746",
    "source_code_url": None,
    "reward_location": (
        "Section 3.3: r_t=0 for t<T, r_T = -Wirelength - lambda*Congestion. "
        "Eq. 2: HPWL wirelength. Section 3.3.5: top-10% avg congestion. "
        "Section 3.3.6: density as hard constraint (action mask). "
        "Fig. 1: sequential macro placement, reward only at final step."
    ),
    "year": 2021,
    "domain": "chip_design",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "RL agent places chip macros on a canvas to minimize wirelength "
        "and congestion. Terminal-only reward at final placement step. "
        "Density enforced as hard constraint via action masking. "
        "Deployed for Google TPU-v5 tapeout — superhuman in <6 hours."
    ),
    "documented_failure": (
        "None — well-designed reward. Deployed in production for Google "
        "TPU chip design. The 2-component reward with hard density "
        "constraint is elegant: wirelength correlates with power and "
        "timing (Section 3.3.1), congestion as soft cost, density as "
        "hard mask. No documented reward exploits."
    ),
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "post_deployment",
    "fix_known": None,
    "compute_cost_class": "high",
    "is_negative_example": True,
    "encoding_rationale": {
        "terminal_only": (
            "Reward is 0 for all steps except the last. This is "
            "intentional — intermediate placements don't have a "
            "meaningful cost until the full placement is evaluated. "
            "Modeled as TERMINAL reward."
        ),
        "negative_well_designed": (
            "Both components are negative (minimize wirelength and "
            "congestion). goodhart will fire negative_only_reward "
            "as WARNING (tracking signal, state-dependent). This is "
            "correct behavior — the reward IS all-negative by design."
        ),
        "density_hard_constraint": (
            "Density is NOT a reward component — it's enforced by "
            "masking infeasible grid cells. This is the cleanest "
            "constraint handling: impossible to violate. Not modeled "
            "as a RewardSource."
        ),
        "lambda": (
            "lambda=0.01 in the Nature paper (Section 3.3). Balances "
            "wirelength (primary) vs congestion (secondary)."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # Terminal-only reward: r_T = -W(p,g) - lambda*C(p,g)
    # T = number of macros (~20-80 per chip block)
    # Density enforced via action mask (not in reward)

    model = EnvironmentModel(
        name="Chip Placement (Mirhoseini et al. 2021)",
        max_steps=80,  # T = number of macros to place
        gamma=1.0,  # undiscounted (terminal only)
        n_actions=900,  # 30x30 grid cells
        action_type="discrete",
        n_states=1000000,  # netlist graph embeddings
        death_probability=0.0,
    )

    # Wirelength cost (HPWL, Eq. 2)
    # Negative: lower wirelength = better placement
    model.add_reward_source(RewardSource(
        name="wirelength_cost",
        reward_type=RewardType.TERMINAL,
        value=-1.0,  # normalized HPWL
        requires_action=True,
        intentional=True,
        state_dependent=True,
    ))

    # Congestion cost (top-10% average, Section 3.3.5)
    # Weighted by lambda=0.01
    model.add_reward_source(RewardSource(
        name="congestion_cost",
        reward_type=RewardType.TERMINAL,
        value=-0.01,  # lambda * normalized congestion
        requires_action=True,
        state_dependent=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
