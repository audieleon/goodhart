"""CaRL: simple reward scales with mini-batch size (Jaeger et al. 2025).

Multiplicative route-completion reward avoids local minima that
trap complex additive rewards at large batch sizes.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "carl_simple_reward",
    "source_paper": (
        'Jaeger, Dauner, Beisswenger, Gerstenecker, Chitta & Geiger, '
        '"CaRL: Learning Scalable Planning Policies with Simple Rewards," '
        'CoRL 2025'
    ),
    "paper_url": "https://arxiv.org/abs/2504.17838",
    "source_code_url": "https://github.com/autonomousvision/CaRL",
    "reward_location": (
        "Section 3, Equation 1: r_t = RC_t * prod(p_t) - T. "
        "Appendix F: 6 terminal penalties (Table F.1), 5 soft penalties "
        "(Section F.2), comfort thresholds (Table 16). "
        "nuPlan survival bonus Equation 8: r* = (1-s)*r_t + s*100/N, s=0.6."
    ),
    "year": 2025,
    "domain": "driving",
    "encoding_basis": "primary_source",
    "verification_date": "2026-04-30",
    "brief_summary": (
        "CaRL uses route completion as the sole reward, with multiplicative "
        "soft penalties and terminal episode endings for hard infractions. "
        "This simple design scales with mini-batch size where complex "
        "additive rewards (Roach) get stuck in local minima."
    ),
    "documented_failure": (
        "Documented failure of the PRIOR complex reward (Roach): agent "
        "waits at green traffic lights because speed reward is maximized "
        "by not moving during red-to-green transitions. At mini-batch 1024, "
        "complex reward drops from 34 to 2 DS while CaRL improves from "
        "21 to 38 DS. CaRL's multiplicative design prevents this because "
        "route completion is zero when stationary."
    ),
    "failure_mechanism": "compound_trap",
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Replace additive multi-component reward with multiplicative "
        "route-completion reward. CaRL achieves 64 DS at batch 16384 "
        "vs Roach's 0 DS."
    ),
    "compute_cost_class": "high",
    "is_negative_example": True,
    "source_code_url": "https://github.com/autonomousvision/CaRL",
    "encoding_rationale": {
        "route_completion": (
            "PER_STEP, value=1.0 (normalized, max 100 total RC). "
            "requires_action=True (must drive to complete route). "
            "intentional=True (this IS the objective). "
            "can_loop=False (finite RC, max 100)."
        ),
        "soft_penalties": (
            "Multiplicative factors p_t in [0,1]. Modeled as modifiers "
            "on route_completion. Outside lanes p=0, lane center linear "
            "1.0-0.0, speeding linear over 0-8 km/h, TTC p=0.5, "
            "comfort p=1.0-0.5*#/6."
        ),
        "terminal_penalties": (
            "T=1 for collision and red light (episode ends). "
            "Other terminals (stop sign, route deviation, blocked) "
            "end episode with T=0."
        ),
        "well_designed": (
            "This is a NEGATIVE example — CaRL's reward is well-designed. "
            "The paper documents the failure of the PRIOR reward (Roach). "
            "CaRL fixes it with: (1) finite total reward prevents loops, "
            "(2) multiplicative penalties prevent earning reward while "
            "violating, (3) hard penalties = termination."
        ),
        "goodhart_note": (
            "goodhart fires idle_exploit because it models the "
            "multiplicative penalty factors as additive subtractions. "
            "In reality, CaRL's penalties are multiplicative (p_t in "
            "[0,1]) — the agent cannot earn negative reward, and the "
            "minimum is 0 (RC=0 when stationary). This is a known "
            "limitation of the EnvironmentModel: it does not fully "
            "capture multiplicative reward structures. The false "
            "positive is documented as a goodhart limitation."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # CaRL reward: r_t = RC_t * prod(p_t) - T
    # From Appendix F, Equation 7
    # RC_t = route completion percentage this timestep (max 100 total)
    # p_t = soft penalty factors in [0,1]
    # T = terminal penalty (1 for collision/red light, 0 otherwise)

    model = EnvironmentModel(
        name="CaRL Autonomous Driving (Jaeger et al. 2025)",
        max_steps=3000,  # 36 routes, 1-2km each, 10 Hz
        gamma=0.99,
        n_actions=2,  # acceleration + steering
        action_type="continuous",
        death_probability=0.01,  # collisions end episode
    )

    # Route completion — the primary (and only real) reward
    # Max 100 RC total per route, finite by design
    model.add_reward_source(RewardSource(
        name="route_completion",
        reward_type=RewardType.PER_STEP,
        value=1.0,  # normalized; actual is RC percentage per step
        requires_action=True,
        intentional=True,
        can_loop=False,  # finite RC, max 100 per route
        max_occurrences=0,
        state_dependent=True,
        scales_with="distance",
    ))

    # Soft penalties — multiplicative on route completion
    # Outside lanes: p=0 (zeroes out reward)
    model.add_reward_source(RewardSource(
        name="outside_lanes_penalty",
        reward_type=RewardType.PER_STEP,
        value=-1.0,  # p=0 when violated → zeroes reward
        requires_action=True,
        modifies="route_completion",
        modifier_type="multiplicative",
    ))

    # Speeding penalty: p linearly 1.0→0.0 over 0-8 km/h excess
    model.add_reward_source(RewardSource(
        name="speeding_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.5,  # average penalty factor
        requires_action=True,
        modifies="route_completion",
        modifier_type="multiplicative",
        state_dependent=True,
        scales_with="velocity",
    ))

    # TTC (time to collision): p=0.5 when violated
    model.add_reward_source(RewardSource(
        name="ttc_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.5,
        requires_action=True,
        modifies="route_completion",
        modifier_type="multiplicative",
    ))

    # Terminal penalty: collision or red light → episode ends, T=1
    model.add_reward_source(RewardSource(
        name="collision_terminal",
        reward_type=RewardType.TERMINAL,
        value=-1.0,
        requires_action=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
