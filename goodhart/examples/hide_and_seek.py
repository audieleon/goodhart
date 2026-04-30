"""Example: OpenAI Hide-and-Seek emergent exploits (ICLR 2020).

Multi-agent competition produced 6 emergent strategies including
box surfing — seekers riding unlocked boxes through walls to
reach hidden agents. This exploited a physics engine bug where
agents could move with boxes regardless of ground contact.

This is the HARDEST test for our framework because the exploit
arises from:
  1. Multi-agent interaction (not single-agent)
  2. Physics engine bugs (not reward structure)
  3. Emergent strategy (not predictable from rewards alone)

What the framework CAN catch vs what it CAN'T:
  - CAN: reward structure incentivizes the arms race
  - CAN: identify that the reward is zero-sum (one side's exploit
    is the other's failure, driving escalation)
  - CAN'T: predict specific physics exploits
  - CAN'T: predict which strategies will emerge

This example shows the LIMITS of pre-training analysis and
where runtime monitoring is needed instead.

Source: Baker et al. 2020, "Emergent Tool Use from Multi-Agent
  Autocurricula" (ICLR)
"""

from goodhart.models import *
from goodhart.engine import *
from goodhart.rules.reward import *
from goodhart.rules.training import *
from goodhart.rules.architecture import PrecedentRule, Precedent


# ---- Multi-agent specific rules ----

class ZeroSumArmsRace(PrecedentRule):
    """Zero-sum multi-agent reward creates escalating arms race."""

    @property
    def name(self): return "zero_sum_arms_race"
    @property
    def description(self):
        return ("Zero-sum reward in multi-agent setting drives "
                "escalating counterstrategies that may exploit "
                "environment physics")
    @property
    def precedents(self):
        return [
            Precedent(
                source="Baker et al. 2019 — Emergent Tool Use (ICLR 2020)",
                setting="Hiders: +1 if hidden, -1 if seen. "
                        "Seekers: opposite. Zero-sum.",
                outcome="6 emergent strategies including box surfing "
                        "(riding boxes through walls via physics bug). "
                        "Each strategy forced a counter-strategy. Arms "
                        "race escalated beyond intended complexity.",
                year=2020,
            ),
        ]

    def check(self, model, config=None):
        verdicts = []
        # Detect zero-sum: positive and negative rewards of equal magnitude
        pos = [s for s in model.reward_sources if s.value > 0]
        neg = [s for s in model.reward_sources if s.value < 0]
        if pos and neg:
            pos_total = sum(s.value for s in pos)
            neg_total = sum(abs(s.value) for s in neg)
            if 0.8 < pos_total / max(neg_total, 0.001) < 1.2:
                verdicts.append(Verdict(
                    rule_name=self.name,
                    severity=Severity.WARNING,
                    message=(f"Near zero-sum reward structure "
                             f"(+{pos_total:.1f} / -{neg_total:.1f}). "
                             f"Multi-agent competition may produce "
                             f"escalating exploits of environment physics."),
                    recommendation=(
                        "Precedent: OpenAI Hide-and-Seek — zero-sum reward "
                        "drove agents to discover box surfing (physics bug). "
                        "Consider: (1) physics engine robustness testing, "
                        "(2) action space constraints, (3) runtime exploit "
                        "monitoring."
                    ),
                ))
        return verdicts


class MultiAgentRewardAsymmetry(PrecedentRule):
    """Check for reward asymmetries in competitive settings."""

    @property
    def name(self): return "multi_agent_asymmetry"
    @property
    def description(self):
        return "Reward asymmetry between agents in competitive settings"
    @property
    def precedents(self):
        return [
            Precedent(
                source="OpenAI Five — Dota 2 (2019)",
                setting="Reward includes gold advantage, XP advantage, "
                        "kills, tower damage. Each component weighted.",
                outcome="Reward shaping required 'surgery' — continuing "
                        "training across reward function changes. Getting "
                        "the weights wrong caused degenerate strategies "
                        "(e.g., not last-hitting creeps, ignoring courier).",
                year=2019,
            ),
            Precedent(
                source="Vinyals et al. 2019 — AlphaStar",
                setting="Win/loss reward only (sparse). But with league "
                        "training: each agent trains against a population.",
                outcome="Sparse reward worked BECAUSE league training "
                        "provides diverse opponents. Single opponent "
                        "causes co-adaptation and strategy collapse.",
                year=2019,
            ),
        ]

    def check(self, model, config=None):
        return []  # Needs multi-agent config to fire


class PhysicsExploitRisk(PrecedentRule):
    """Flag environments with complex physics for exploit risk."""

    @property
    def name(self): return "physics_exploit_risk"
    @property
    def description(self):
        return "Complex physics simulation creates exploit surface"
    @property
    def precedents(self):
        return [
            Precedent(
                source="Baker et al. 2019 — Hide-and-Seek box surfing",
                setting="MuJoCo physics with moveable objects, ramps, walls",
                outcome="Agents discovered that applying force while on top "
                        "of a box moves the box — not physically realistic "
                        "but allowed by the simulator. Led to 'surfing' "
                        "through walls.",
                year=2020,
            ),
            Precedent(
                source="Lilian Weng 2024 — Reward Hacking survey",
                setting="Robot jumping task in physics simulator",
                outcome="Agent exploited physics simulator bug to achieve "
                        "unrealistic jump height for maximum reward.",
                year=2024,
            ),
            Precedent(
                source="DeepMind 2017 — Robot grasping",
                setting="Grasping reward with wrong reference point on brick",
                outcome="Agent flipped the brick because reward was "
                        "calculated from wrong reference point. Technically "
                        "maximized reward without grasping.",
                year=2017,
            ),
        ]

    def check(self, model, config=None):
        # Can't automatically detect physics complexity, but flag it
        return []


# ---- The example ----

def run_example():
    """Model OpenAI's Hide-and-Seek environment."""

    # Hider perspective
    model = EnvironmentModel(
        name="Hide-and-Seek (Hider, Baker et al. 2019)",
        max_steps=240,  # 240 steps per round
        n_states=100000,  # continuous state space (approx)
        n_actions=10,  # move, grab, lock
    )

    # Hider reward: +1 per step hidden, -1 per step seen
    model.add_reward_source(RewardSource(
        name="hidden bonus",
        reward_type=RewardType.PER_STEP,
        value=1.0,
        respawn=RespawnBehavior.INFINITE,
        requires_action=True,
    ))
    model.add_reward_source(RewardSource(
        name="seen penalty",
        reward_type=RewardType.PER_STEP,
        value=-1.0,
        respawn=RespawnBehavior.INFINITE,
        requires_action=False,  # happens when seeker finds you
    ))
    # Boundary penalty
    model.add_reward_source(RewardSource(
        name="boundary penalty",
        reward_type=RewardType.PER_STEP,
        value=-0.1,
    ))

    config = TrainingConfig(
        algorithm="PPO",
        lr=1.5e-4,
        num_envs=512,  # they used 512 parallel games
        num_workers=1,
          # ~500M steps per agent
        model_params=5_000_000,
        entropy_coeff=0.01,
    )

    # Build engine with standard + multi-agent rules
    engine = TrainingAnalysisEngine().add_all_rules()
    engine.add_rules([
        ZeroSumArmsRace(),
        MultiAgentRewardAsymmetry(),
        PhysicsExploitRisk(),
    ])

    engine.print_report(model, config)

    print()
    print("=== WHAT THE FRAMEWORK CATCHES ===")
    print()
    print("✓ Zero-sum structure → arms race warning")
    print("✓ Respawning reward (per-step +1/-1) → flagged")
    print("✓ High compute budget context")
    print()
    print("=== WHAT THE FRAMEWORK CANNOT CATCH ===")
    print()
    print("✗ Box surfing — a physics engine bug. No amount of")
    print("  reward analysis predicts that agents will discover")
    print("  they can ride boxes through walls.")
    print()
    print("✗ The specific 6-stage arms race. The ORDER of")
    print("  strategy emergence depends on learning dynamics,")
    print("  not reward structure.")
    print()
    print("✗ Whether the emergent behaviors are 'bugs' or")
    print("  'features'. Box surfing is creative problem-solving")
    print("  from the agent's perspective.")
    print()
    print("=== THE LESSON ===")
    print()
    print("Pre-training analysis catches REWARD-LEVEL exploits")
    print("(wrong incentives, degenerate equilibria). It cannot")
    print("catch ENVIRONMENT-LEVEL exploits (physics bugs,")
    print("unintended interactions). Those need:")
    print("  1. Physics engine testing (adversarial state search)")
    print("  2. Runtime behavior monitoring")
    print("  3. Human review of emergent strategies")
    print()
    print("This is the honest boundary of the framework.")


if __name__ == "__main__":
    run_example()
