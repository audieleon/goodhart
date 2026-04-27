# Goodhart Examples

Each example shows how to express a specific RL failure case
using the goodhart analysis framework. These are teaching examples --
they demonstrate the framework's expressiveness, not hardcoded rules.

Use these as templates for your own projects. Copy the pattern,
change the parameters, add your own precedent rules.

## Examples

| File | Paper/Source | Failure type |
|------|-------------|-------------|
| coast_runners.py | OpenAI 2016 | Respawning reward loops |
| cartpole_suicide.py | Classic RL | Suicidal step penalty |
| road_runner_replay.py | Atari 2017 | Death reset exploit |
| bicycle_circles.py | Weng 2024 | Shaping loop exploit |
| multiroom_traps.py | Entries 041-047 | Sparse reward degeneracy |
| expert_collapse.py | Shazeer 2017 + ours | Expert collapse |
| ppo_37_details.py | ICLR Blog 2022 | PPO pitfalls |
| atari_exploits.py | 2013-2018 | Respawning + limits |
| robotics_exploits.py | 2017-2022 | Idle + shaping |
| evolution_exploits.py | 1983-2019 | Historical + limits |
| mujoco_locomotion.py | Standard envs | Alive bonus tradeoff |
| sparse_reward_traps.py | 2016-2022 | Budget + threshold |
| hide_and_seek.py | ICLR 2020 | Limits: physics |
| dota2_openai_five.py | OpenAI 2019 | Limits: emergence |
| tic_tac_toe_crash.py | 2019 | Limits: adversarial |

## Adding your own

```python
from goodhart.models import *
from goodhart.engine import *
from goodhart.rules.reward import *
from goodhart.rules.training import *
from goodhart.rules.architecture import PrecedentRule, Precedent

# 1. Define your environment
model = EnvironmentModel(name="My Task", max_steps=1000)
model.add_reward_source(RewardSource(
    name="goal", reward_type=RewardType.TERMINAL, value=10.0,
    discovery_probability=0.05,
))

# 2. Define your training config
config = TrainingConfig(
    lr=3e-4,
    entropy_coeff=0.01,
    num_specialists=3,
    routing_floor=0.10,
)

# 3. Add project-specific rules
class MyFailureRule(PrecedentRule):
    @property
    def name(self): return "my_failure"
    @property
    def description(self): return "Something I learned"
    @property
    def precedents(self):
        return [Precedent(
            source="My lab notebook, experiment 42",
            setting="lr=1e-3 with 3 specialists",
            outcome="Expert collapse within 100 updates",
        )]
    def check(self, model, config=None):
        if config and config.lr > 5e-4 and config.num_specialists > 2:
            return [Verdict(
                rule_name=self.name,
                severity=Severity.WARNING,
                message="High lr with multiple specialists -- risk of collapse",
                recommendation=f"Precedent: {self.precedents[0].outcome}",
            )]
        return []

# 4. Run analysis
engine = TrainingAnalysisEngine().add_all_rules()
engine.add_rules([MyFailureRule()])
engine.print_report(model, config)
```
