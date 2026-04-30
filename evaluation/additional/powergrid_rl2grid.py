"""RL2Grid power grid operations benchmark (Marchesini et al. 2025).

R = alpha*survive + beta*overload + eta*cost. Three-component
reward with CMDP safety constraints. Built with RTE France.
7 grids from 14 to 118 buses, month-long episodes.
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


METADATA = {
    "id": "powergrid_rl2grid",
    "source_paper": (
        'Marchesini, Donnot, Crozier, Dytham, Merz, Schewe, '
        'Westerbeck, Wu, Marot & Donti, "RL2Grid: Benchmarking '
        'Reinforcement Learning in Power Grid Operations," '
        'arXiv:2503.23101, 2025'
    ),
    "paper_url": "https://arxiv.org/abs/2503.23101",
    "source_code_url": "https://github.com/emarche/RL2Grid",
    "reward_location": (
        "Section 3, page 5: R_t = alpha*R_survive + beta*R_overload "
        "+ eta*R_cost. R_overload: line flow vs capacity margin, "
        "disconnection penalty. R_cost: -(losses + |redisp| + |storage|) "
        "* c_marginal. Normalized [-1,1] and [-1,0]. Weights in "
        "Appendix F. Section 3.2: CMDP constraints LSI (hard) and "
        "TLO (soft). 7 grids, 5-min steps, month-long episodes."
    ),
    "year": 2025,
    "domain": "energy",
    "encoding_basis": "primary_source",
    "verification_date": "2026-05-01",
    "brief_summary": (
        "Power grid operations benchmark with 3-component reward: "
        "survival bonus, line overload penalty, and economic cost. "
        "CMDP constraints for load shedding and thermal overloads. "
        "Designed with transmission system operators (RTE France, "
        "National Grid ESO, 50Hertz). >180K CPU hours of experiments."
    ),
    "documented_failure": (
        "RL baselines struggle with real-world grid complexity. "
        "Vanilla DQN/PPO/SAC achieve low survival rates on larger "
        "grids. Heuristic-guided transitions (idle + recovery) "
        "significantly improve performance, highlighting that "
        "the reward alone is insufficient without operational priors."
    ),
    "failure_mechanism": None,
    "detection_type": "structural",
    "discovery_stage": "during_training",
    "fix_known": (
        "Expert heuristics (idle at 95% capacity threshold, recovery "
        "to restore topology) improve sample efficiency and survival."
    ),
    "compute_cost_class": "extreme",
    "is_negative_example": True,
    "encoding_rationale": {
        "three_component_reward": (
            "R_survive (positive, passive — grid stays alive), "
            "R_overload (mixed — capacity margin vs disconnection), "
            "R_cost (negative — economic penalties). Well-balanced "
            "multi-objective design from TSO expertise."
        ),
        "cmdp_constraints": (
            "LSI (load shedding): hard constraint, zero tolerance. "
            "TLO (thermal overload): soft constraint with budget. "
            "Modeled as ON_EVENT penalties."
        ),
        "benchmark_not_failure": (
            "Well-designed reward — the difficulty is in the "
            "environment complexity (exponential action space, "
            "long horizons, hard physics constraints), not the "
            "reward specification."
        ),
    },
}


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    # R = alpha*survive + beta*overload + eta*cost
    # bus14 grid: 14 substations, 20 lines, 6 generators, 11 loads
    # Month-long episodes at 5-min steps (~8640 steps)

    model = EnvironmentModel(
        name="RL2Grid Power Grid (Marchesini et al. 2025)",
        max_steps=8640,  # 1 month at 5-min intervals
        gamma=0.99,
        n_actions=209,  # bus14 topology actions
        action_type="discrete",
        n_states=100000,  # grid state (flows, topology, forecasts)
        death_probability=0.01,  # grid collapse terminates
    )

    # R_survive: positive constant for each step grid is alive
    # Normalized by episode length, range [0, 1]
    model.add_reward_source(RewardSource(
        name="survival_bonus",
        reward_type=RewardType.PER_STEP,
        value=1.0,  # alpha * R_survive, normalized
        requires_action=False,  # passive — grid stays alive
        intentional=True,
    ))

    # R_overload: line capacity margin - disconnection penalty
    # Range [-1, 1]
    model.add_reward_source(RewardSource(
        name="overload_penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.5,  # beta * R_overload, avg case
        requires_action=True,
        state_dependent=True,
        scales_with="power_flow",
    ))

    # R_cost: economic cost of redispatch/curtailment
    # Range [-1, 0]
    model.add_reward_source(RewardSource(
        name="economic_cost",
        reward_type=RewardType.PER_STEP,
        value=-0.3,  # eta * R_cost, avg case
        requires_action=True,
        state_dependent=True,
    ))

    # LSI constraint: load shedding / islanding (hard)
    model.add_reward_source(RewardSource(
        name="lsi_constraint",
        reward_type=RewardType.ON_EVENT,
        value=-10.0,  # hard constraint — zero tolerance
        requires_action=True,
    ))

    # TLO constraint: transmission line overload (soft, budget)
    model.add_reward_source(RewardSource(
        name="tlo_constraint",
        reward_type=RewardType.ON_EVENT,
        value=-1.0,  # soft constraint with cumulative budget
        requires_action=True,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
