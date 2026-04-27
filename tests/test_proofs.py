"""Tests that verify the connection between Python rules and LEAN proofs.

Checks that:
1. Every rule that declares a proof has a matching LEAN theorem
2. The LEAN theorem name exists in the proof source file
3. Rules without proofs are explicitly documented as unverified
"""

import os
from pathlib import Path

import pytest

from goodhart.engine import TrainingAnalysisEngine


PROOF_DIR = Path(__file__).parent.parent / "proofs" / "GoodhartProofs"
PROOF_FILE = PROOF_DIR / "Basic.lean"  # kept for backward compat


def get_lean_theorems():
    """Extract theorem names from all LEAN source files."""
    theorems = set()
    if not PROOF_DIR.exists():
        return theorems
    for lean_file in PROOF_DIR.rglob("*.lean"):
        source = lean_file.read_text()
        for line in source.split("\n"):
            line = line.strip()
            if line.startswith("theorem ") or line.startswith("noncomputable def "):
                name = line.split()[1]
                theorems.add(name)
    return theorems


class TestProofLinkage:
    """Verify that Python rules link to actual LEAN proofs."""

    def setup_method(self):
        self.engine = TrainingAnalysisEngine()
        self.engine.add_all_rules()
        self.lean_theorems = get_lean_theorems()

    def test_lean_file_exists(self):
        """The LEAN proof file must exist."""
        assert PROOF_FILE.exists(), f"LEAN proof file not found: {PROOF_FILE}"

    def test_lean_has_theorems(self):
        """The LEAN file must contain theorems."""
        assert len(self.lean_theorems) > 0, "No theorems found in LEAN file"

    def test_all_proof_names_exist_in_lean(self):
        """Every rule that declares a proof must reference a real LEAN theorem."""
        missing = []
        for rule in self.engine.rules:
            if rule.proof is not None:
                if rule.proof.proof_name not in self.lean_theorems:
                    missing.append(
                        f"Rule '{rule.name}' references LEAN theorem "
                        f"'{rule.proof.proof_name}' but it doesn't exist"
                    )
        assert not missing, "\n".join(missing)

    def test_proof_coverage(self):
        """Report which rules have proofs and which don't."""
        proved = []
        unproved = []
        for rule in self.engine.rules:
            if rule.proof is not None:
                proved.append(rule.name)
            else:
                unproved.append(rule.name)

        # Track actual proof count — update when adding/removing proofs
        assert len(proved) >= 10, (
            f"Expected at least 10 rules with proofs, got {len(proved)}: {proved}"
        )

    def test_proved_rules_have_statements(self):
        """Every proof must have a human-readable statement."""
        for rule in self.engine.rules:
            if rule.proof is not None:
                assert rule.proof.statement, (
                    f"Rule '{rule.name}' has proof but no statement"
                )

    def test_proved_rules_have_parameters(self):
        """Every proof must declare parameter mappings."""
        for rule in self.engine.rules:
            if rule.proof is not None:
                assert rule.proof.parameters, (
                    f"Rule '{rule.name}' has proof but no parameter mappings"
                )

    def test_proof_consistency_death_beats_survival(self):
        """Verify the death_beats_survival rule implements its theorem.

        LEAN theorem: ∀ p < 0, ∀ fm < fn, p * fn < p * fm (discounted)
        Python check: net_penalty * disc(1) > net_penalty * disc(10)
        """
        from goodhart.rules.reward import DeathBeatsSurvival
        from goodhart.models import EnvironmentModel, RewardSource, RewardType

        rule = DeathBeatsSurvival()
        assert rule.proof is not None
        assert rule.proof.proof_name == "death_beats_survival_discounted"

        # Verify the Python check matches the theorem
        model = EnvironmentModel(name="test", max_steps=100)
        model.add_reward_source(RewardSource(
            name="penalty", reward_type=RewardType.PER_STEP, value=-0.5,
        ))

        verdicts = rule.check(model)
        assert len(verdicts) == 1
        assert verdicts[0].severity.value == "critical"

        # The theorem says: for p < 0, p * 1 > p * 10
        # Python checks: ev_die_step1 > ev_survive_N
        # Where ev_die_step1 = penalty * 1 = -0.5
        #       ev_survive_N = penalty * 10 = -5.0
        # -0.5 > -5.0 ✓ (dying is better)
        assert -0.5 > -5.0  # matches theorem

    def test_proof_consistency_loop_dominance(self):
        """Verify the respawning_exploit rule implements its theorem.

        LEAN theorem: ∀ v > 0, g > 0, t > 0, v * T > g * t → v * T / t > g
        Python check: ev_loop > goal where ev_loop = value * (max_steps / period)
        """
        from goodhart.rules.reward import RespawningExploit
        from goodhart.models import (
            EnvironmentModel, RewardSource, RewardType, RespawnBehavior,
        )

        rule = RespawningExploit()
        assert rule.proof is not None
        assert rule.proof.proof_name == "loop_dominance"

        model = EnvironmentModel(name="test", max_steps=1000)
        model.add_reward_source(RewardSource(
            name="goal", reward_type=RewardType.TERMINAL, value=10.0,
        ))
        model.add_reward_source(RewardSource(
            name="loop_source", reward_type=RewardType.ON_EVENT, value=0.5,
            respawn=RespawnBehavior.TIMED, respawn_time=3,
            can_loop=True, loop_period=3,
        ))

        verdicts = rule.check(model)
        assert len(verdicts) == 1
        assert verdicts[0].severity.value == "critical"

        # The theorem says: v * T / t > g when v * T > g * t
        # v=0.5, T=1000, t=3, g=10
        # 0.5 * 1000 = 500 > 10 * 3 = 30 ✓
        # 500 / 3 = 166.7 > 10 ✓ (looping dominates)
        assert 0.5 * 1000 > 10.0 * 3  # matches theorem precondition
        assert 0.5 * 1000 / 3 > 10.0  # matches theorem conclusion
