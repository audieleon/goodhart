#!/usr/bin/env python3
"""Analyze all evaluation encodings and generate statistics.

Runs goodhart on every evaluation file, collects results,
and outputs summary statistics for the paper.
"""

import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from io import StringIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from goodhart.engine import TrainingAnalysisEngine
from goodhart.models import Severity


def load_module(path):
    """Load a Python module from path."""
    name = os.path.basename(path)[:-3]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def extract_metadata(mod, path):
    """Extract metadata from module docstring."""
    doc = mod.__doc__ or ""
    meta = {
        "file": path,
        "source": os.path.basename(os.path.dirname(path)),
        "name": os.path.basename(path)[:-3],
    }

    for line in doc.split("\n"):
        line = line.strip()
        if line.startswith("Source:"):
            meta["source_paper"] = line[7:].strip()
        elif line.startswith("Failure:"):
            meta["failure"] = line[8:].strip()
        elif line.startswith("Mechanism:"):
            meta["mechanism"] = line[10:].strip()
        elif line.startswith("Domain:"):
            meta["domain"] = line[7:].strip()
        elif line.startswith("Detection:"):
            meta["detection"] = line[10:].strip()
        elif line.startswith("Tool should catch:"):
            meta["expected"] = line[18:].strip()
        elif line.startswith("Tool result:"):
            meta["expected"] = line[12:].strip()

    return meta


def run_analysis(mod):
    """Run goodhart analysis on a module and capture results."""
    engine = TrainingAnalysisEngine()
    engine.add_all_rules()

    # Find the model in run_example by inspecting the function
    # We need to actually call it and capture output
    import goodhart.engine as eng

    results = {"criticals": 0, "warnings": 0, "infos": 0, "rules_fired": []}

    # Try to extract model from the module
    # Most modules create model inside run_example, so we need to
    # intercept the engine calls
    original_print_report = eng.TrainingAnalysisEngine.print_report
    original_analyze = eng.TrainingAnalysisEngine.analyze

    captured = {}

    def mock_print_report(self, model, config=None):
        result = self.analyze(model, config)
        captured["result"] = result
        captured["model"] = model

    eng.TrainingAnalysisEngine.print_report = mock_print_report

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        mod.run_example()
    except Exception:
        pass
    finally:
        sys.stdout = old_stdout
        eng.TrainingAnalysisEngine.print_report = original_print_report

    if "result" in captured:
        result = captured["result"]
        for v in result.verdicts:
            if v.severity == Severity.CRITICAL:
                results["criticals"] += 1
            elif v.severity == Severity.WARNING:
                results["warnings"] += 1
            else:
                results["infos"] += 1
            results["rules_fired"].append(v.rule_name)

    return results


def main():
    eval_dir = os.path.dirname(__file__)
    all_results = []

    for dirpath, _, filenames in sorted(os.walk(eval_dir)):
        for f in sorted(filenames):
            if not f.endswith(".py") or f in ("__init__.py", "analyze.py"):
                continue
            path = os.path.join(dirpath, f)
            try:
                mod = load_module(path)
                meta = extract_metadata(mod, path)
                analysis = run_analysis(mod)
                meta.update(analysis)

                # Determine overall result
                if meta["criticals"] > 0:
                    meta["result"] = "FAIL"
                elif meta["warnings"] > 0:
                    meta["result"] = "WARN"
                elif meta["infos"] > 0:
                    meta["result"] = "INFO"
                else:
                    meta["result"] = "PASS"

                all_results.append(meta)
            except Exception as e:
                print(f"ERROR: {path}: {e}", file=sys.stderr)

    # --- Summary Statistics ---
    print("=" * 70)
    print("REWARD FAILURE DATASET — Analysis Summary")
    print("=" * 70)

    print(f"\nTotal evaluation encodings: {len(all_results)}")

    # By source
    by_source = Counter(r["source"] for r in all_results)
    print("\nBy source:")
    for src, count in sorted(by_source.items()):
        print(f"  {src}: {count}")

    # By result
    by_result = Counter(r["result"] for r in all_results)
    print("\nBy result:")
    for res in ["FAIL", "WARN", "INFO", "PASS"]:
        print(f"  {res}: {by_result.get(res, 0)}")

    # Rules fired frequency
    all_rules = []
    for r in all_results:
        all_rules.extend(r.get("rules_fired", []))
    rule_freq = Counter(all_rules)
    print("\nMost common rules fired:")
    for rule, count in rule_freq.most_common(15):
        print(f"  {rule}: {count}")

    # By source + result breakdown
    print("\nDetailed breakdown:")
    for src in sorted(by_source.keys()):
        subset = [r for r in all_results if r["source"] == src]
        results = Counter(r["result"] for r in subset)
        print(f"  {src} ({len(subset)}): "
              f"FAIL={results.get('FAIL', 0)} "
              f"WARN={results.get('WARN', 0)} "
              f"INFO={results.get('INFO', 0)} "
              f"PASS={results.get('PASS', 0)}")

    # Krakovna-specific stats
    krakovna = [r for r in all_results if r["source"] == "krakovna"]
    if krakovna:
        structural = [r for r in krakovna if r["result"] in ("FAIL", "WARN")]
        print(f"\nKrakovna catalog coverage:")
        print(f"  Encoded: {len(krakovna)} of 82 entries")
        print(f"  Structural issues detected: {len(structural)}/{len(krakovna)}")

    # Eureka stats
    eureka = [r for r in all_results if r["source"] == "eureka"]
    if eureka:
        rejected = [r for r in eureka if r["result"] == "FAIL"]
        print(f"\nEureka coverage:")
        print(f"  Encoded: {len(eureka)} of 29 tasks")
        print(f"  Rejected (CRITICAL): {len(rejected)}/{len(eureka)}")

    # Write taxonomy.json
    taxonomy_path = os.path.join(eval_dir, "taxonomy.json")
    taxonomy = {
        "generated": "2026-04-30",
        "total_entries": len(all_results),
        "entries": all_results,
    }
    with open(taxonomy_path, "w") as f:
        json.dump(taxonomy, f, indent=2, default=str)
    print(f"\nTaxonomy written to {taxonomy_path}")

    # Also include the 66 built-in examples count
    examples_dir = os.path.join(eval_dir, "..", "goodhart", "examples")
    n_examples = len([f for f in os.listdir(examples_dir)
                      if f.endswith(".py") and f != "__init__.py"])
    total = len(all_results) + n_examples
    print(f"\nTotal dataset size: {total} "
          f"({n_examples} built-in + {len(all_results)} evaluation)")


if __name__ == "__main__":
    main()
