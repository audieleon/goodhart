"""Architecture and capacity analysis rules with canonical examples.

These rules check architecture parameters against documented failure
cases from published papers and experiments. Unlike reward and
training rules (which are formal predicates), these are PRECEDENT-BASED:
they fire when your configuration matches a range where a specific,
documented failure occurred.
"""

from dataclasses import dataclass
from typing import List

from goodhart.models import Severity, Verdict, TrainingConfig
from goodhart.engine import Rule


@dataclass
class Precedent:
    """A documented case where specific parameters caused failure."""
    source: str          # paper/entry reference
    setting: str         # what they configured
    outcome: str         # what happened
    year: int = 2024     # when


class PrecedentRule(Rule):
    """Rule backed by documented precedents.

    Subclass this to create project-specific rules. Provide:
    - name, description (standard Rule interface)
    - precedents: list of documented cases
    - check(): the parameter check + precedent citation
    """

    @property
    def precedents(self) -> List[Precedent]:
        return []


class EmbedDimCapacity(PrecedentRule):
    """Check if embedding dimension provides sufficient capacity."""

    @property
    def name(self): return "embed_dim_capacity"

    @property
    def description(self):
        return "Embedding dimension vs task complexity"

    @property
    def precedents(self):
        return [
            Precedent(
                source="Empirical -- MiniHack small-scale multi-specialist",
                setting="embed_dim=64, 3 specialists at ~10K params each",
                outcome="No advantage over monolithic baseline. Specialists "
                        "too small to develop different representations.",
                year=2026,
            ),
            Precedent(
                source="Samvelyan et al. 2021 (MiniHack paper, Appendix D.3)",
                setting="Small model: embed=16, hidden=64",
                outcome="'Poor performance on Room-Ultimate-15 and "
                        "CorridorBattle'. Failed on tasks medium model solved.",
                year=2021,
            ),
            Precedent(
                source="Empirical -- model scaling experiments",
                setting="5x transformer: 21.5M params, 16 layers, embed=256",
                outcome="WORSE than 1x baseline (4.6M). +37.1 vs +40.4. "
                        "Overfitting with excessive capacity.",
                year=2026,
            ),
        ]

    def check(self, model, config: TrainingConfig = None):
        if config is None:
            return []
        verdicts = []

        if config.num_specialists > 1:
            params_per_specialist = config.model_params / config.num_specialists
            if params_per_specialist < 20_000:
                verdicts.append(Verdict(
                    rule_name=self.name,
                    severity=Severity.WARNING,
                    message=(f"~{params_per_specialist/1000:.0f}K params per specialist "
                             f"({config.num_specialists} specialists, "
                             f"{config.model_params/1000:.0f}K total). "
                             f"May be too small for meaningful specialization."),
                    details={"params_per_specialist": params_per_specialist},
                    recommendation=(
                        f"Precedent: Empirical -- 10K params/specialist showed "
                        f"no advantage. MiniHack paper -- embed=16 failed on "
                        f"hard tasks. Consider >=50K params per specialist."
                    ),
                ))

        if config.embed_dim < 32 and model.n_states > 500:
            verdicts.append(Verdict(
                rule_name=self.name,
                severity=Severity.WARNING,
                message=(f"embed_dim={config.embed_dim} with {model.n_states} "
                         f"states. Limited representational capacity."),
                recommendation=(
                    f"Precedent: MiniHack small (embed=16) failed on "
                    f"Room-Ultimate. Medium (embed=64) succeeded."
                ),
            ))

        if config.model_params > 10_000_000:
            verdicts.append(Verdict(
                rule_name=self.name,
                severity=Severity.INFO,
                message=(f"Large model ({config.model_params/1e6:.1f}M params). "
                         f"Risk of overfitting on simple tasks."),
                recommendation=(
                    f"Precedent: larger models do not always outperform "
                    f"smaller ones on simple tasks. More params != better."
                ),
            ))

        return verdicts


class RoutingFloorNecessity(PrecedentRule):
    """Check if routing floor is needed and sufficient."""

    @property
    def name(self): return "routing_floor_necessity"

    @property
    def description(self):
        return "Routing floor constraint for multi-specialist models"

    @property
    def precedents(self):
        return [
            Precedent(
                source="Empirical -- load balancing loss",
                setting="3 specialists, no floor, softmax routing",
                outcome="Third specialist consistently marginalized to "
                        "<5% weight. Expert collapse.",
                year=2026,
            ),
            Precedent(
                source="Empirical -- MiniHack expert collapse",
                setting="3 specialists, gate MLP, no floor",
                outcome="95% MLP, 5% CNN, 0% Transformer. Complete "
                        "collapse detected by inspecting trained weights.",
                year=2026,
            ),
            Precedent(
                source="Shazeer et al. 2017 -- MoE",
                setting="Sparse gating without load balancing",
                outcome="Most experts receive zero traffic. Load balancing "
                        "loss introduced as standard fix.",
                year=2017,
            ),
        ]

    def check(self, model, config: TrainingConfig = None):
        if config is None:
            return []
        verdicts = []

        if config.num_specialists > 1 and config.routing_floor == 0:
            precedent_text = "; ".join(
                f"{p.source}: {p.outcome[:60]}..."
                for p in self.precedents[:2]
            )
            verdicts.append(Verdict(
                rule_name=self.name,
                severity=Severity.CRITICAL,
                message=(f"{config.num_specialists} specialists with "
                         f"routing_floor=0. Expert collapse is near-certain."),
                recommendation=(
                    f"Set routing_floor >= {1.0/config.num_specialists/3:.2f}. "
                    f"Precedents: {precedent_text}"
                ),
            ))

        return verdicts


class RecurrenceType(PrecedentRule):
    """Check RNN type selection."""

    @property
    def name(self): return "recurrence_type"

    @property
    def description(self):
        return "LSTM vs GRU selection and implications"

    @property
    def precedents(self):
        return [
            Precedent(
                source="Empirical -- MiniHack GRU vs LSTM",
                setting="Used GRU (SF default) instead of paper's LSTM",
                outcome="Different training dynamics. Results not comparable "
                        "to published baselines using LSTM.",
                year=2026,
            ),
            Precedent(
                source="Samvelyan et al. 2021 (MiniHack, Appendix D.2)",
                setting="LSTM with hidden=256 for all experiments",
                outcome="Standard choice for NetHack/MiniHack. GRU may "
                        "work but isn't validated in this domain.",
                year=2021,
            ),
        ]

    def check(self, model, config: TrainingConfig = None):
        if config is None or not config.use_rnn:
            return []
        verdicts = []

        if config.rnn_type == "gru":
            verdicts.append(Verdict(
                rule_name=self.name,
                severity=Severity.INFO,
                message="Using GRU instead of LSTM.",
                recommendation=(
                    f"Precedent: Empirical -- GRU vs LSTM produced different "
                    f"results on MiniHack. If comparing to published baselines "
                    f"that use LSTM, results may not be directly comparable."
                ),
            ))

        return verdicts


class ActorCountEffect(PrecedentRule):
    """Check actor count vs published baselines."""

    @property
    def name(self): return "actor_count_effect"

    @property
    def description(self):
        return "Actor count affects exploration diversity"

    @property
    def precedents(self):
        return [
            Precedent(
                source="Samvelyan et al. 2021 -- actor count comparison",
                setting="64 actors (M3 Mac) vs 256 (MiniHack paper)",
                outcome="Less diverse batches, fewer goal discoveries per "
                        "second. Both models get same count so relative "
                        "comparison is controlled.",
                year=2026,
            ),
            Precedent(
                source="Samvelyan et al. 2021 (MiniHack, Appendix D.2)",
                setting="256 actors, 7667 steps/sec on 2 GPUs + 20 CPUs",
                outcome="High throughput enables sparse reward tasks. "
                        "Fewer actors may fail on same tasks.",
                year=2021,
            ),
        ]

    def check(self, model, config: TrainingConfig = None):
        if config is None:
            return []
        verdicts = []

        total_actors = config.num_envs * max(config.num_workers, 1)
        if total_actors < 64 and model.n_states > 1000:
            verdicts.append(Verdict(
                rule_name=self.name,
                severity=Severity.WARNING,
                message=(f"Only {total_actors} actors for {model.n_states} "
                         f"states. May be insufficient for exploration."),
                recommendation=(
                    f"Precedent: MiniHack paper used 256 actors. 64 "
                    f"actors produced similar results on easy tasks but "
                    f"may struggle on sparse-reward tasks."
                ),
            ))

        return verdicts


# Standard architecture rule collection
ARCHITECTURE_RULES = [
    EmbedDimCapacity(),
    RoutingFloorNecessity(),
    RecurrenceType(),
    ActorCountEffect(),
]
