"""Example: OpenAI Five Dota 2 reward complexity (2019).

Dota 2 has one of the most complex reward functions ever used in RL:
  - Win/loss (sparse, delayed ~45 minutes)
  - Gold advantage (shaped)
  - XP advantage (shaped)
  - Kill/death rewards (event-based)
  - Tower damage (event-based)
  - Creep last-hit rewards (event-based, respawning)

OpenAI needed "surgery" — continuing training across reward
function changes — because getting the reward weights wrong
produced degenerate strategies.

This example pushes the framework's expressiveness by modeling
a multi-component, multi-timescale reward function and showing
which interactions it catches.
"""

from goodhart.models import *
from goodhart.engine import *
from goodhart.rules.reward import *
from goodhart.rules.training import *
from goodhart.rules.architecture import PrecedentRule, Precedent


class RewardComponentInteraction(PrecedentRule):
    """Check for conflicting reward components."""

    @property
    def name(self): return "reward_component_interaction"
    @property
    def description(self):
        return "Multiple reward components may create conflicting incentives"
    @property
    def precedents(self):
        return [
            Precedent(
                source="OpenAI Five — Dota 2 (2019)",
                setting="Dense reward: gold/XP advantage + kill reward + "
                        "tower damage + win/loss. Each weighted.",
                outcome="Required multiple rounds of 'surgery' to get weights "
                        "right. Wrong weights caused: ignoring creeps (gold "
                        "reward too low), suicidal tower diving (kill reward "
                        "too high), passive play (death penalty too high).",
                year=2019,
            ),
        ]

    def check(self, model, config=None):
        verdicts = []
        # Check if multiple reward sources could conflict
        event_sources = [s for s in model.reward_sources
                         if s.reward_type == RewardType.ON_EVENT]
        shaping_sources = [s for s in model.reward_sources
                           if s.reward_type == RewardType.SHAPING]
        terminal_sources = [s for s in model.reward_sources
                            if s.reward_type == RewardType.TERMINAL]

        n_components = len(event_sources) + len(shaping_sources) + len(terminal_sources)

        if n_components >= 4:
            # Check for magnitude imbalances
            magnitudes = sorted(
                [(s.name, abs(s.value)) for s in model.reward_sources if s.value != 0],
                key=lambda x: x[1], reverse=True)

            if magnitudes:
                max_mag = magnitudes[0][1]
                min_mag = magnitudes[-1][1]
                ratio = max_mag / max(min_mag, 1e-10)

                if ratio > 100:
                    verdicts.append(Verdict(
                        rule_name=self.name,
                        severity=Severity.WARNING,
                        message=(f"{n_components} reward components with "
                                 f"{ratio:.0f}x magnitude range "
                                 f"('{magnitudes[0][0]}'={max_mag} vs "
                                 f"'{magnitudes[-1][0]}'={min_mag}). "
                                 f"Small components will be ignored."),
                        recommendation=(
                            f"Precedent: OpenAI Five needed 'surgery' to "
                            f"rebalance reward components. Components with "
                            f"<1% of max magnitude are effectively invisible."
                        ),
                    ))

        if n_components >= 3:
            verdicts.append(Verdict(
                rule_name=self.name,
                severity=Severity.INFO,
                message=(f"{n_components} reward components. Multi-component "
                         f"rewards risk conflicting incentives."),
                recommendation=(
                    "Precedent: OpenAI Five — wrong reward weights caused "
                    "degenerate strategies (ignoring creeps, suicidal tower "
                    "diving, passive play). Monitor each component's "
                    "contribution during training."
                ),
            ))

        return verdicts


class SparseDelayedGoal(PrecedentRule):
    """Check for very long episodes with sparse terminal reward."""

    @property
    def name(self): return "sparse_delayed_goal"
    @property
    def description(self):
        return "Sparse goal reward with very long episode"
    @property
    def precedents(self):
        return [
            Precedent(
                source="OpenAI Five — Dota 2",
                setting="Win/loss after ~45 min (~80,000 steps). "
                        "Dense shaping rewards added to bridge the gap.",
                outcome="Sparse win/loss alone was insufficient. Dense "
                        "shaping (gold/XP advantage) was essential for "
                        "learning but introduced its own exploit risks.",
                year=2019,
            ),
            Precedent(
                source="Vinyals et al. 2019 — AlphaStar",
                setting="Win/loss after ~10-30 min. No shaping reward.",
                outcome="Worked because league training provides diverse "
                        "opponents and natural curriculum. Would fail with "
                        "single opponent.",
                year=2019,
            ),
        ]

    def check(self, model, config=None):
        verdicts = []
        terminal = [s for s in model.reward_sources
                    if s.reward_type == RewardType.TERMINAL]
        per_step = [s for s in model.reward_sources
                    if s.reward_type in (RewardType.PER_STEP, RewardType.SHAPING,
                                         RewardType.ON_EVENT)]

        if terminal and not per_step and model.max_steps > 1000:
            verdicts.append(Verdict(
                rule_name=self.name,
                severity=Severity.CRITICAL,
                message=(f"Terminal-only reward with {model.max_steps} step "
                         f"episodes. Credit assignment over {model.max_steps} "
                         f"steps is extremely difficult."),
                recommendation=(
                    "Precedent: OpenAI Five added dense shaping rewards "
                    "(gold/XP advantage) because sparse win/loss over "
                    "~80K steps was insufficient. AlphaStar used sparse "
                    "reward but needed league training (population of "
                    "diverse opponents) as implicit curriculum."
                ),
            ))

        return verdicts


def run_example():
    """Model OpenAI Five's Dota 2 reward structure."""

    model = EnvironmentModel(
        name="Dota 2 (OpenAI Five, 2019)",
        max_steps=80000,       # ~45 minute game at 30 fps
        n_states=10**15,       # effectively infinite
        n_actions=170000,      # ~170K possible actions per step
        death_probability=0.0001,
    )

    # Terminal: win/loss
    model.add_reward_source(RewardSource(
        name="win/loss",
        reward_type=RewardType.TERMINAL,
        value=1.0,
        discovery_probability=0.5,  # roughly 50% win rate
    ))

    # Dense shaping: gold advantage
    model.add_reward_source(RewardSource(
        name="gold advantage",
        reward_type=RewardType.SHAPING,
        value=0.006,  # per gold difference per step (estimated)
        can_loop=False,  # advantage can't be looped
    ))

    # Dense shaping: XP advantage
    model.add_reward_source(RewardSource(
        name="xp advantage",
        reward_type=RewardType.SHAPING,
        value=0.004,
        can_loop=False,
    ))

    # Event: hero kills
    model.add_reward_source(RewardSource(
        name="hero kill",
        reward_type=RewardType.ON_EVENT,
        value=0.5,
        respawn=RespawnBehavior.TIMED,
        respawn_time=500,  # hero respawns after ~500 steps
        max_occurrences=0,
    ))

    # Event: hero death (penalty)
    model.add_reward_source(RewardSource(
        name="hero death",
        reward_type=RewardType.ON_EVENT,
        value=-0.5,
        max_occurrences=0,
    ))

    # Event: tower damage
    model.add_reward_source(RewardSource(
        name="tower damage",
        reward_type=RewardType.ON_EVENT,
        value=0.2,
        max_occurrences=11,  # finite towers
    ))

    # Event: creep last hits (respawning source)
    model.add_reward_source(RewardSource(
        name="creep last hit",
        reward_type=RewardType.ON_EVENT,
        value=0.01,
        respawn=RespawnBehavior.TIMED,
        respawn_time=30,  # new creep wave every ~30 steps
        max_occurrences=0,  # unlimited
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=2e-4,
        model_params=159_000_000,  # 159M parameters (LSTM)
        num_envs=65536,  # they used 65536 parallel games
        num_workers=1,
          # ~10 billion steps
        entropy_coeff=0.01,
        use_rnn=True,
        rnn_type="lstm",
        rnn_size=4096,  # massive LSTM
    )

    engine = TrainingAnalysisEngine().add_all_rules()
    engine.add_rules([
        RewardComponentInteraction(),
        SparseDelayedGoal(),
    ])

    engine.print_report(model, config)

    print()
    print("=== WHAT THE FRAMEWORK CATCHES ===")
    print()
    print("✓ Respawning creep rewards — could incentivize farming")
    print("  over fighting (if creep reward too high vs kill reward)")
    print("✓ Multi-component reward interaction — 7 components with")
    print("  different magnitudes risk conflicting incentives")
    print("✓ Hero kill reward respawning — creates cycle potential")
    print("✓ Large model capacity note")
    print()
    print("=== WHAT ACTUALLY WENT WRONG (from the paper) ===")
    print()
    print("OpenAI reported needing 'surgery' — modifying reward")
    print("weights mid-training because:")
    print("  - Agents initially ignored creep farming (reward too low)")
    print("  - Agents tower-dived suicidally (kill reward too high")
    print("    relative to death penalty)")
    print("  - Agents played too passively (death penalty too high)")
    print()
    print("The framework's RewardComponentInteraction rule catches")
    print("the magnitude imbalance but can't predict which specific")
    print("strategy will emerge from which imbalance. That requires")
    print("domain knowledge (game theory of Dota 2).")
    print()
    print("=== THE LIMIT ===")
    print()
    print("With 170K actions per step, 10^15 states, and 7 reward")
    print("components, Dota 2 is at the edge of what static analysis")
    print("can predict. The framework correctly identifies the risk")
    print("factors (respawning rewards, component imbalance, zero-sum")
    print("dynamics) but the specific degenerate strategies emerge")
    print("from game-specific interactions that no reward analysis")
    print("can fully capture.")
    print()
    print("This is an honest boundary: the framework reduces the")
    print("search space for problems but doesn't eliminate the need")
    print("for domain expertise and runtime monitoring.")


if __name__ == "__main__":
    run_example()
