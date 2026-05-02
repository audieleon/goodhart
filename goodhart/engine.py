"""Analysis engines that compose and run rules.

AnalysisEngine runs reward rules only. TrainingAnalysisEngine extends
it with training and architecture rules for full pre-flight checks.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from goodhart.models import EnvironmentModel, FormalBasis, Result, Severity, Verdict


# =====================================================================
# Rule base class
# =====================================================================


class Rule(ABC):
    """Base class for all analysis rules.

    Each rule has:
      - name: human-readable identifier
      - description: what it checks
      - applies_to(model): precondition -- does this rule apply?
      - check(model): the actual analysis
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    def proof(self) -> Optional[FormalBasis]:
        """The LEAN theorem that verifies this rule's math.

        Override to declare the formal basis. Returns None if
        the rule has no machine-verified proof (yet).
        """
        return None

    def applies_to(self, model: EnvironmentModel) -> bool:
        """Override to restrict when this rule runs."""
        return True

    @abstractmethod
    def check(self, model: EnvironmentModel, config=None) -> List[Verdict]:
        """Run the analysis. Return list of verdicts.

        CONTRACT: check() MUST NOT mutate model or config.
        The engine does not make defensive copies for performance.
        Rules are pure analysis functions: read model, return verdicts.
        """
        ...


# =====================================================================
# Analysis engine -- composes rules
# =====================================================================


class AnalysisEngine:
    """Composes and runs analysis rules over an environment model."""

    def __init__(self):
        self.rules: List[Rule] = []

    def add_rule(self, rule: Rule):
        if rule.name not in {r.name for r in self.rules}:
            self.rules.append(rule)
        return self  # chainable

    def add_all_rules(self):
        """Set the standard reward rule library (replaces any existing rules)."""
        from goodhart.rules.reward import REWARD_RULES

        existing_names = {r.name for r in self.rules}
        for rule in REWARD_RULES:
            if rule.name not in existing_names:
                self.rules.append(rule)
        return self

    def analyze(self, model: EnvironmentModel, config=None) -> Result:
        """Run all applicable rules and return a Result."""
        verdicts = []
        for rule in self.rules:
            if rule.applies_to(model):
                try:
                    verdicts.extend(rule.check(model, config))
                except Exception as e:
                    verdicts.append(
                        Verdict(
                            rule_name=rule.name,
                            severity=Severity.WARNING,
                            message=f"Rule crashed: {e}",
                        )
                    )
        verdicts.extend(self._check_contradictions(verdicts))
        # Enrich verdicts with educational content from explanations DB
        self._enrich_verdicts(verdicts)
        passed = not any(v.severity == Severity.CRITICAL for v in verdicts)
        return Result(verdicts=verdicts, passed=passed)

    @staticmethod
    def _enrich_verdicts(verdicts: List[Verdict]):
        """Inject learn_more from explanations DB into verdicts that lack it."""
        from goodhart.rules.explanations import get_learn_more

        for v in verdicts:
            if v.learn_more is None:
                learn_more = get_learn_more(v.rule_name)
                if learn_more:
                    v.learn_more = learn_more

    @staticmethod
    def _check_contradictions(verdicts: List[Verdict]) -> List[Verdict]:
        """Detect contradictions between verdict recommendations."""
        contradiction_verdicts = []

        # Collect all recommendation text (lowercased) with their rule names
        recs = []
        for v in verdicts:
            if v.recommendation:
                recs.append((v.rule_name, v.recommendation.lower()))

        # Define contradiction pairs: (pattern_a, pattern_b, description)
        contradiction_patterns = [
            # Penalty contradictions
            (
                lambda r: "add" in r and "penalty" in r,
                lambda r: ("remove" in r and "penalty" in r) or (r.startswith("set") and "penalty" in r and "0" in r),
                "One rule says add step penalty, another says remove it",
            ),
            # Entropy contradictions
            (
                lambda r: "increase" in r and "entropy" in r,
                lambda r: ("decrease" in r and "entropy" in r) or ("entropy" in r and "too high" in r),
                "One rule says increase entropy, another says decrease it",
            ),
            # Intrinsic motivation contradictions
            (
                lambda r: "add" in r and "intrinsic" in r,
                lambda r: "intrinsic" in r and ("too" in r or "exceed" in r or "reduce" in r),
                "One rule says add intrinsic motivation, another says it is already too strong",
            ),
        ]

        for pattern_a, pattern_b, description in contradiction_patterns:
            rules_a = [name for name, rec in recs if pattern_a(rec)]
            rules_b = [name for name, rec in recs if pattern_b(rec)]
            if rules_a and rules_b:
                # Avoid self-contradictions from the same rule
                if set(rules_a) != set(rules_b) or len(rules_a) > 1:
                    contradiction_verdicts.append(
                        Verdict(
                            rule_name="contradiction",
                            severity=Severity.WARNING,
                            message=(
                                f"Contradictory recommendations: {description}. "
                                f"Rules involved: {', '.join(sorted(set(rules_a + rules_b)))}"
                            ),
                            details={"rules_a": rules_a, "rules_b": rules_b},
                            recommendation="Review these rules manually; the fix for one may worsen the other",
                        )
                    )

        return contradiction_verdicts

    def print_report(self, model: EnvironmentModel, config=None, verbose=False):
        """Run analysis and print formatted report."""
        from goodhart.fmt import (
            header,
            section,
            verdict as fmt_verdict,
            summary,
            passed_banner,
        )

        result = self.analyze(model, config)

        header(f"REWARD ANALYSIS: {model.name}")

        if result.criticals:
            section("CRITICAL", len(result.criticals))
            for v in result.criticals:
                fmt_verdict(v, verbose=verbose)

        if result.warnings:
            section("WARNINGS", len(result.warnings))
            for v in result.warnings:
                fmt_verdict(v, verbose=verbose)

        if result.infos:
            section("INFO", len(result.infos))
            for v in result.infos:
                fmt_verdict(v, verbose=verbose)

        if not result.verdicts:
            passed_banner()

        summary(len(result.criticals), len(result.warnings), len(result.infos))

        return result


class TrainingAnalysisEngine(AnalysisEngine):
    """Composes reward rules and training rules.

    Extends AnalysisEngine with training-specific rule libraries
    and a richer print_report that shows algorithm info.
    """

    def add_all_rules(self):
        """Add all available rules from all libraries (appends with dedup)."""
        from goodhart.rules.reward import REWARD_RULES
        from goodhart.rules.training import TRAINING_RULES
        from goodhart.rules.architecture import ARCHITECTURE_RULES
        from goodhart.rules.advisories import ADVISORY_RULES

        existing_names = {r.name for r in self.rules}
        for rule in list(REWARD_RULES) + list(TRAINING_RULES) + list(ARCHITECTURE_RULES) + list(ADVISORY_RULES):
            if rule.name not in existing_names:
                self.rules.append(rule)
                existing_names.add(rule.name)
        return self

    def add_rules(self, rules: list):
        """Add a list of rules (project-specific or custom, with dedup)."""
        existing_names = {r.name for r in self.rules}
        for rule in rules:
            if rule.name not in existing_names:
                self.rules.append(rule)
                existing_names.add(rule.name)
        return self

    def print_report(self, model: EnvironmentModel, config=None, verbose=False):
        from goodhart.fmt import (
            header,
            section,
            verdict as fmt_verdict,
            summary,
            passed_banner,
            DIM_COLOR,
            RULE_COLOR,
            RESET,
            HEADER_COLOR,
            REC_COLOR,
        )

        result = self.analyze(model, config)

        subtitle = None
        if config:
            subtitle = f"Algorithm: {config.algorithm}, lr={config.lr:.0e}"
        header(f"FULL ANALYSIS: {model.name}", subtitle)

        # Show what we're analyzing
        if model.reward_sources:
            print(f"  {HEADER_COLOR}Reward sources:{RESET}")
            for s in model.reward_sources:
                intent = f" {REC_COLOR}(intentional){RESET}" if s.intentional else ""
                passive = f" {DIM_COLOR}(passive){RESET}" if not s.requires_action else ""
                print(
                    f"    {RULE_COLOR}{s.name:28s}{RESET} "
                    f"{s.value:+8.4f}  {DIM_COLOR}{s.reward_type.value}{RESET}"
                    f"{intent}{passive}"
                )
            print(
                f"  {DIM_COLOR}max_steps={model.max_steps}, "
                f"gamma={model.gamma}, "
                f"death_prob={model.death_probability}{RESET}"
            )
            print()

        if result.criticals:
            section("CRITICAL", len(result.criticals))
            for v in result.criticals:
                fmt_verdict(v, verbose=verbose)

        if result.warnings:
            section("WARNINGS", len(result.warnings))
            for v in result.warnings:
                fmt_verdict(v, verbose=verbose)

        if result.infos:
            section("INFO", len(result.infos))
            for v in result.infos:
                fmt_verdict(v, verbose=verbose)

        if not result.verdicts:
            passed_banner()

        summary(len(result.criticals), len(result.warnings), len(result.infos))

        return result
