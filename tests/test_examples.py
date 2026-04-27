"""Test all examples run without error."""

import importlib
import pkgutil
import pytest

import goodhart.examples


def _discover_examples():
    """Auto-discover all example modules (no hardcoded list)."""
    examples = []
    for importer, modname, ispkg in pkgutil.iter_modules(goodhart.examples.__path__):
        if not ispkg and modname != "__init__":
            examples.append(modname)
    return sorted(examples)


EXAMPLES = _discover_examples()


@pytest.mark.parametrize("example_name", EXAMPLES)
def test_example_runs(example_name):
    """Each example should import and run without error."""
    mod = importlib.import_module(f"goodhart.examples.{example_name}")
    assert hasattr(mod, "run_example"), f"{example_name} missing run_example()"
    mod.run_example()
