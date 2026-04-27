# Contributing to goodhart

Thanks for your interest in contributing.

## Development setup

```bash
git clone https://github.com/audieleon/goodhart.git
cd goodhart
pip install -e ".[dev]"
pytest
```

## Adding a rule

1. Create a class inheriting from `Rule` in the appropriate module:
   - `goodhart/rules/reward.py` for reward structure rules (algorithm-agnostic)
   - `goodhart/rules/training.py` for training hyperparameter rules (algorithm-specific: PPO, DQN, SAC, etc.)
   - `goodhart/rules/architecture.py` for precedent-based architecture rules
   - `goodhart/rules/advisories.py` for blind-spot advisories (pattern hints for things static analysis can't detect)

2. Implement `name`, `description`, `check(model, config)`, and optionally `applies_to(model)` and `proof` (for formal verification linkage).

3. Add your rule to the module's `*_RULES` list.

4. Add an explanation entry in `goodhart/rules/explanations.py` with `learn_more`, `examples`, and `papers`.

5. Add the rule name to the appropriate category in the `--rules` listing in `goodhart/cli.py`.

6. Add tests — at minimum: fires when it should, silent when it shouldn't.

7. If your rule has a formal basis, add the LEAN theorem name to the `proof` property and verify it exists via `pytest tests/test_proofs.py`.

## Adding an example

1. Create a file in `goodhart/examples/` with a module docstring and `run_example()` function.
2. Examples are auto-discovered — no need to edit a list.
3. Run `pytest tests/test_examples.py` to verify it executes cleanly.
4. If the example demonstrates a specific rule, add it to that rule's `examples` list in `goodhart/rules/explanations.py`.

## Adding a preset

1. Add a function in `goodhart/presets.py` returning `(EnvironmentModel, TrainingConfig)`.
2. Add it to the `PRESETS` dict at the bottom of the file.
3. Include the source paper in the function's docstring.
4. Set `algorithm` to match what the paper actually used (PPO, DQN, SAC, etc.).

## LEAN proofs

Proofs are in `proofs/GoodhartProofs/`. To verify:

```bash
cd proofs
lake build
```

Requires LEAN 4 and Mathlib. Zero sorry is required for all proofs.

## Code style

- No hardcoded counts — use `RULE_COUNT` from `goodhart.rules`
- All rule addition methods deduplicate by name
- Use `yaml.safe_load` (never `yaml.load`)
- Public functions need type annotations and docstrings
- Training rules should gate on `config.algorithm` with appropriate thresholds per algorithm
