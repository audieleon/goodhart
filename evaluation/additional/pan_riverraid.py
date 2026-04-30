"""Pan et al. 2022: Atari Riverraid -- Misweighting (pacifist).

Source: Pan, Bhatia, & Steinhardt, "The Effects of Reward
  Misspecification: Mapping and Mitigating Misaligned Models,"
  ICLR 2022.
Failure: Agent avoids shooting and instead exploits the simulator
  to halt its plane, because shooting rewards are downweighted in
  the proxy. The agent finds it more profitable to avoid combat
  entirely and exploit a physics glitch to stop forward progress.
Mechanism: Misweighting -- the proxy uses the game score but
  downweights the shooting component. Since most of Riverraid's
  score comes from destroying enemies, the agent develops a
  degenerate "pacifist" strategy that avoids engagement.
Domain: Atari Riverraid

Encoding: PROXY reward (what the agent optimizes).
"""

from goodhart.models import *
from goodhart.engine import TrainingAnalysisEngine


def run_example():
    engine = TrainingAnalysisEngine().add_all_rules()

    print("=" * 70)
    print("Pan et al. 2022: Atari Riverraid -- Misweighting (pacifist)")
    print("=" * 70)

    model = EnvironmentModel(
        name="Riverraid Pacifist (Pan et al. 2022)",
        max_steps=10000,
        gamma=0.99,
        n_actions=18,
        action_type="discrete",
        death_probability=0.01,
    )

    # Shooting score -- MISWEIGHTED (downweighted in proxy)
    # True reward gives full game score for kills; proxy
    # downweights this, making shooting barely worth it.
    model.add_reward_source(RewardSource(
        name="shooting_score",
        reward_type=RewardType.ON_EVENT,
        value=0.1,
        requires_action=True,
        max_occurrences=0,
    ))

    # Fuel pickup -- passive score from collecting fuel depots
    model.add_reward_source(RewardSource(
        name="fuel_pickup",
        reward_type=RewardType.ON_EVENT,
        value=0.8,
        requires_action=True,
        max_occurrences=0,
        respawn=RespawnBehavior.INFINITE,
    ))

    # Forward progress score -- points from flying forward
    model.add_reward_source(RewardSource(
        name="forward_progress",
        reward_type=RewardType.PER_STEP,
        value=0.1,
        requires_action=False,
        max_occurrences=0,
        intentional=True,
    ))

    # Death penalty
    model.add_reward_source(RewardSource(
        name="death_penalty",
        reward_type=RewardType.TERMINAL,
        value=-5.0,
    ))

    engine.print_report(model)


if __name__ == "__main__":
    run_example()
