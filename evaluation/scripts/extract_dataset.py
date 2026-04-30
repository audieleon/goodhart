#!/usr/bin/env python3
"""Extract the Reward Failure Dataset to structured formats.

Reads all encoding files (goodhart/examples/ + evaluation/sources/),
captures METADATA + EnvironmentModel + analysis results, and writes
JSONL, CSV, and summary statistics.

Usage:
    python evaluation/scripts/extract_dataset.py
    python evaluation/scripts/extract_dataset.py --validate
"""

import argparse
import csv
import dataclasses
import importlib.util
import json
import os
import sys
from io import StringIO

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)

from goodhart.engine import TrainingAnalysisEngine
from goodhart.models import Severity


def load_module(path):
    name = os.path.basename(path)[:-3]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def serialize_model(model):
    """Convert EnvironmentModel to a JSON-safe dict."""
    d = dataclasses.asdict(model)
    for src in d["reward_sources"]:
        src["reward_type"] = (
            src["reward_type"].value
            if hasattr(src["reward_type"], "value")
            else str(src["reward_type"])
        )
        src["respawn"] = (
            src["respawn"].value
            if hasattr(src["respawn"], "value")
            else str(src["respawn"])
        )
        if src["value_range"] is not None:
            src["value_range"] = list(src["value_range"])
    return d


def capture_run(mod):
    """Run mod.run_example() and capture model + analysis result."""
    captured = {}
    original = TrainingAnalysisEngine.print_report

    def mock(self, model, config=None, verbose=False):
        result = self.analyze(model, config)
        captured["model"] = model
        captured["config"] = config
        captured["result"] = result

    TrainingAnalysisEngine.print_report = mock
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        mod.run_example()
    except Exception:
        pass
    finally:
        sys.stdout = old_stdout
        TrainingAnalysisEngine.print_report = original

    return captured


def classify_source(path):
    """Determine the source category from file path."""
    if "goodhart/examples" in path:
        return "builtin"
    elif "sources/eureka" in path:
        return "eureka"
    elif "sources/heldout" in path:
        return "heldout"
    elif "sources/krakovna" in path:
        return "krakovna"
    elif "sources/published_papers" in path:
        return "published_papers"
    return "unknown"


def build_record(path, mod, captured):
    """Build a single dataset record from module + captured data."""
    meta = getattr(mod, "METADATA", {})
    record = {
        "id": meta.get("id", os.path.basename(path)[:-3]),
        "source_category": classify_source(path),
        "source_file": os.path.relpath(path, ROOT),
    }

    # Provenance
    for key in [
        "source_paper", "paper_url", "source_code_url",
        "reward_location", "year", "domain", "encoding_basis",
        "verification_date",
    ]:
        record[key] = meta.get(key)

    # Ground truth
    for key in [
        "brief_summary", "documented_failure", "failure_mechanism",
        "detection_type", "discovery_stage", "fix_known",
        "compute_cost_class", "is_negative_example",
    ]:
        record[key] = meta.get(key)

    # Encoding rationale
    record["encoding_rationale"] = meta.get("encoding_rationale", {})

    # Environment + reward sources
    if "model" in captured:
        model_dict = serialize_model(captured["model"])
        record["environment"] = {
            k: v for k, v in model_dict.items() if k != "reward_sources"
        }
        record["reward_sources"] = model_dict["reward_sources"]
        record["num_reward_sources"] = len(model_dict["reward_sources"])
    else:
        record["environment"] = None
        record["reward_sources"] = []
        record["num_reward_sources"] = 0

    # Analysis results
    if "result" in captured:
        result = captured["result"]
        record["analysis"] = {
            "passed": result.passed,
            "num_criticals": len(
                [v for v in result.verdicts if v.severity == Severity.CRITICAL]
            ),
            "num_warnings": len(
                [v for v in result.verdicts if v.severity == Severity.WARNING]
            ),
            "num_infos": len(
                [v for v in result.verdicts if v.severity == Severity.INFO]
            ),
            "rules_fired": [v.rule_name for v in result.verdicts],
            "goodhart_result": (
                "FAIL" if any(v.severity == Severity.CRITICAL for v in result.verdicts)
                else "WARN" if any(v.severity == Severity.WARNING for v in result.verdicts)
                else "INFO" if result.verdicts
                else "PASS"
            ),
        }
    else:
        record["analysis"] = None

    return record


def discover_files():
    """Find all encoding files."""
    dirs = [
        os.path.join(ROOT, "goodhart", "examples"),
        os.path.join(ROOT, "evaluation", "sources", "eureka"),
        os.path.join(ROOT, "evaluation", "sources", "heldout"),
        os.path.join(ROOT, "evaluation", "sources", "krakovna"),
        os.path.join(ROOT, "evaluation", "sources", "published_papers"),
    ]
    files = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".py") and f not in ("__init__.py", "analyze.py"):
                files.append(os.path.join(d, f))
    return files


def main():
    parser = argparse.ArgumentParser(description="Extract Reward Failure Dataset")
    parser.add_argument("--validate", action="store_true", help="Validate round-trips")
    args = parser.parse_args()

    files = discover_files()
    print(f"Discovered {len(files)} encoding files")

    records = []
    errors = []
    for path in files:
        try:
            mod = load_module(path)
            captured = capture_run(mod)
            record = build_record(path, mod, captured)
            records.append(record)
        except Exception as e:
            errors.append((path, str(e)))

    print(f"Extracted {len(records)} records, {len(errors)} errors")
    if errors:
        for path, err in errors[:5]:
            print(f"  ERROR: {os.path.basename(path)}: {err}")

    # Write JSONL
    output_dir = os.path.join(ROOT, "evaluation", "dataset")
    os.makedirs(output_dir, exist_ok=True)

    jsonl_path = os.path.join(output_dir, "reward_failure_dataset.jsonl")
    with open(jsonl_path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    print(f"Wrote {jsonl_path} ({len(records)} records)")

    # Write summary CSV
    csv_path = os.path.join(output_dir, "reward_failure_dataset.csv")
    csv_fields = [
        "id", "source_category", "domain", "year", "encoding_basis",
        "is_negative_example", "failure_mechanism", "detection_type",
        "num_reward_sources", "source_paper",
    ]
    # Add analysis fields
    csv_fields_with_analysis = csv_fields + ["goodhart_result", "rules_fired"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields_with_analysis)
        writer.writeheader()
        for record in records:
            row = {k: record.get(k) for k in csv_fields}
            if record.get("analysis"):
                row["goodhart_result"] = record["analysis"]["goodhart_result"]
                row["rules_fired"] = "; ".join(record["analysis"].get("rules_fired", []))
            else:
                row["goodhart_result"] = "N/A"
                row["rules_fired"] = ""
            writer.writerow(row)
    print(f"Wrote {csv_path}")

    # Summary statistics
    print()
    print("=" * 60)
    print("REWARD FAILURE DATASET — Summary")
    print("=" * 60)
    print(f"Total records: {len(records)}")

    from collections import Counter
    print(f"\nBy source:")
    for src, count in Counter(r["source_category"] for r in records).most_common():
        print(f"  {src}: {count}")

    print(f"\nBy domain:")
    for dom, count in Counter(r.get("domain", "?") for r in records).most_common():
        print(f"  {dom}: {count}")

    print(f"\nBy encoding basis:")
    for basis, count in Counter(r.get("encoding_basis", "?") for r in records).most_common():
        print(f"  {basis}: {count}")

    print(f"\nBy goodhart result:")
    for result, count in Counter(
        r["analysis"]["goodhart_result"] if r.get("analysis") else "N/A"
        for r in records
    ).most_common():
        print(f"  {result}: {count}")

    neg = sum(1 for r in records if r.get("is_negative_example"))
    print(f"\nNegative examples (well-designed): {neg}")
    print(f"Positive examples (failures): {len(records) - neg}")

    with_url = sum(1 for r in records if r.get("paper_url"))
    print(f"With paper URL: {with_url} ({100*with_url/len(records):.0f}%)")


if __name__ == "__main__":
    main()
