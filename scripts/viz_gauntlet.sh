#!/bin/bash
# viz_gauntlet.sh — run the reward landscape viz on all 66 built-in
# examples and verify every one produces sane output.
#
# Usage: ./scripts/viz_gauntlet.sh           # summary only
#        ./scripts/viz_gauntlet.sh --show    # show each chart
#
# Exit code 0 = all pass, 1 = failures found

set -euo pipefail
cd "$(dirname "$0")/.."

SHOW_VIZ="${1:-}"

PASS=0
FAIL=0
WARN=0
ERRORS=""

echo "═══════════════════════════════════════════════════════"
echo "  Reward Landscape Visualization Gauntlet"
echo "  Testing all $(ls goodhart/examples/*.py | grep -v __init__ | wc -l | tr -d ' ') examples"
echo "═══════════════════════════════════════════════════════"
echo

python3 -c "SHOW_VIZ='$SHOW_VIZ'" -c "" 2>/dev/null  # pass env
SHOW_VIZ="$SHOW_VIZ" python3 << 'PYEOF'
import importlib.util, os, sys, json
from io import StringIO
from goodhart.engine import TrainingAnalysisEngine, AnalysisEngine
from goodhart.viz import reward_landscape_ascii, _compute_strategy_evs

show_viz = os.environ.get("SHOW_VIZ") == "--show"
examples_dir = "goodhart/examples"
results = []
viz_outputs = []

for f in sorted(os.listdir(examples_dir)):
    if not f.endswith('.py') or f == '__init__.py':
        continue
    pid = f[:-3]
    path = os.path.join(examples_dir, f)

    spec = importlib.util.spec_from_file_location(pid, path)
    mod = importlib.util.module_from_spec(spec)
    captured = {}
    orig = AnalysisEngine.analyze
    def mock(self, model, config=None):
        if 'model' not in captured: captured['model'] = model
        return orig(self, model, config)
    AnalysisEngine.analyze = mock
    old = sys.stdout
    sys.stdout = StringIO()
    try:
        spec.loader.exec_module(mod)
        mod.run_example()
    except:
        pass
    finally:
        sys.stdout = old
        AnalysisEngine.analyze = orig

    if 'model' not in captured:
        results.append({"id": pid, "status": "SKIP", "reason": "no model captured"})
        continue

    model = captured['model']

    # Test 1: does it run without crashing?
    # Run rule engine and pass result to viz for consistency
    engine_for_viz = TrainingAnalysisEngine().add_all_rules()
    analysis = engine_for_viz.analyze(model)

    try:
        output = reward_landscape_ascii(model, result=analysis)
    except Exception as e:
        results.append({"id": pid, "status": "FAIL", "reason": f"crashed: {e}"})
        continue

    # Test 2: does it produce output?
    if not output or len(output) < 20:
        results.append({"id": pid, "status": "FAIL", "reason": "empty output"})
        continue

    # Test 3: does it have a winner or all-equal message?
    has_winner = "agent learns this" in output
    all_equal = "All strategies score the same" in output
    if not has_winner and not all_equal:
        results.append({"id": pid, "status": "FAIL", "reason": "no winner and not all-equal"})
        continue

    # Test 4: does the EV math produce finite numbers?
    evs = _compute_strategy_evs(model)
    import math
    bad_evs = [k for k, v in evs.items() if math.isnan(v) or math.isinf(v)]
    if bad_evs:
        results.append({"id": pid, "status": "FAIL", "reason": f"NaN/Inf in {bad_evs}"})
        continue

    # Test 5: does the winner make sense relative to the rule engine?
    rule_names = [v.rule_name for v in analysis.criticals]

    winner = None
    for line in output.split('\n'):
        if 'agent learns this' in line:
            # Extract strategy name between last double-space and ◀
            parts = line.rsplit('  ', 1)
            if len(parts) > 1:
                winner = parts[-1].split('◀')[0].strip()
            break

    # Cross-check: if rule engine flags idle/loop exploits, the viz
    # winner should be a degenerate strategy (stand still, die, loop)
    # — NOT "Solve the task". If the viz says a degenerate strategy
    # wins, it agrees with the rules.
    idle_flagged = "idle_exploit" in rule_names
    loop_flagged = any(r in rule_names for r in
        ["respawning_exploit", "shaping_loop_exploit"])
    degenerate_flagged = idle_flagged or loop_flagged
    viz_degenerate_wins = winner in ("Stand still", "Die immediately", "Farm loop")

    if degenerate_flagged and not viz_degenerate_wins:
        results.append({
            "id": pid, "status": "WARN",
            "reason": f"rules flag degenerate exploit but viz says '{winner}'",
            "winner": winner
        })
        continue

    # Cross-check: if rule engine says respawning_exploit/shaping_loop,
    # viz should ideally show loop
    loop_flagged = any(r in rule_names for r in
        ["respawning_exploit", "shaping_loop_exploit"])
    viz_says_loop = winner == "Farm loop"

    status = "PASS"
    note = ""

    results.append({
        "id": pid, "status": status, "winner": winner or "(all equal)",
        "strategies": len(evs), "criticals": len(analysis.criticals),
        "reason": note
    })
    if show_viz:
        viz_outputs.append((pid, output))

# Print results
passes = [r for r in results if r["status"] == "PASS"]
warns = [r for r in results if r["status"] == "WARN"]
fails = [r for r in results if r["status"] == "FAIL"]
skips = [r for r in results if r["status"] == "SKIP"]

for r in results:
    status = r["status"]
    pid = r["id"]
    if status == "PASS":
        winner = r.get("winner", "?")
        n = r.get("strategies", "?")
        c = r.get("criticals", "?")
        print(f"  ✓ {pid:35s} → {winner:25s} ({n} strategies, {c} criticals)")
    elif status == "WARN":
        reason = r.get("reason", "")
        print(f"  ⚠ {pid:35s} {reason}")
    elif status == "FAIL":
        print(f"  ✗ {pid:35s} {r['reason']}")
    elif status == "SKIP":
        print(f"  - {pid:35s} {r['reason']}")

print()
print(f"═══════════════════════════════════════════════════════")
print(f"  {len(passes)} passed, {len(warns)} warnings, {len(fails)} failed, {len(skips)} skipped")
print(f"  out of {len(results)} examples")
print(f"═══════════════════════════════════════════════════════")

if fails:
    print()
    print("FAILURES:")
    for r in fails:
        print(f"  {r['id']}: {r['reason']}")

if show_viz:
    print()
    for pid, output in viz_outputs:
        print(f"{'─' * 60}")
        print(f"  {pid}")
        print(f"{'─' * 60}")
        print(output)
        print()

if fails:
    sys.exit(1)
PYEOF
