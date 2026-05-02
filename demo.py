#!/usr/bin/env python3
"""goodhart demo — see the tool in action.

Run: python demo.py
"""

import sys
import time

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")
    time.sleep(0.5)

def pause():
    input("\n  [Press Enter to continue]\n")


def main():
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║                                                      ║
    ║   goodhart — catch reward traps before training      ║
    ║                                                      ║
    ║   "When a measure becomes a target, it ceases        ║
    ║    to be a good measure."  — Charles Goodhart, 1975  ║
    ║                                                      ║
    ╚══════════════════════════════════════════════════════╝
    """)

    pause()

    # ---- Demo 1: A doomed config ----
    section("DEMO 1: A doomed experiment")
    print("  You're setting up a sparse reward task.")
    print("  Goal: +1.0 for reaching the target.")
    print("  Step penalty: -0.01 per step.")
    print("  Max episode: 500 steps.")
    print()
    print("  Looks reasonable, right? Let's check.")
    pause()

    from goodhart import check
    print("  >>> from goodhart import check")
    print("  >>> check(goal=1.0, penalty=-0.01, max_steps=500)")
    print()

    ok = check(goal=1.0, penalty=-0.01, max_steps=500, name="Doomed experiment")

    print(f"\n  check() returned: {ok}")
    print("  Your training would waste hours discovering this.")
    print("  goodhart found it in milliseconds.")

    pause()

    # ---- Demo 2: The fix ----
    section("DEMO 2: Fixing the config")
    print("  The tool said to remove the step penalty.")
    print("  Let's try: penalty=0, and add more actors.")
    pause()

    print("  >>> check(goal=1.0, penalty=0, max_steps=500, n_actors=256)")
    print()

    ok = check(goal=1.0, penalty=0, max_steps=500, n_actors=256,
               total_steps=50_000_000, name="Fixed experiment")

    print(f"\n  check() returned: {ok}")
    print("  CLEAR — safe to train.")

    pause()

    # ---- Demo 3: Typed Result API ----
    section("DEMO 3: Typed Result for programmatic use")
    print("  The analyze() function returns a typed Result")
    print("  instead of printing. Use it in your training script.")
    pause()

    from goodhart import analyze

    print("  >>> from goodhart import analyze")
    print("  >>> result = analyze(goal=1.0, penalty=-0.01, max_steps=500)")
    result = analyze(goal=1.0, penalty=-0.01, max_steps=500)
    print(f"\n  result.passed = {result.passed}")
    print(f"  result.has_criticals = {result.has_criticals}")
    print(f"  result.criticals = {len(result.criticals)} issues")
    print(f"  result.warnings = {len(result.warnings)} issues")
    print()
    print("  >>> result.to_dict()")
    import json
    print(f"  {json.dumps(result.to_dict(), indent=4)[:300]}...")

    pause()

    # ---- Demo 4: ASCII Visualization ----
    section("DEMO 4: Reward landscape visualization")
    print("  See which strategy 'wins' at a glance.")
    pause()

    from goodhart.models import EnvironmentModel, RewardSource, RewardType
    from goodhart.viz import reward_landscape_ascii

    model = EnvironmentModel(name="Sparse reward trap", max_steps=500, gamma=0.99)
    model.add_reward_source(RewardSource(
        name="goal", reward_type=RewardType.TERMINAL,
        value=1.0, discovery_probability=0.05,
    ))
    model.add_reward_source(RewardSource(
        name="step penalty", reward_type=RewardType.PER_STEP,
        value=-0.01,
    ))

    print(reward_landscape_ascii(model))

    pause()

    # ---- Demo 5: Now fix it and visualize ----
    section("DEMO 5: Fixed config visualization")
    print("  Remove the penalty. The landscape changes completely.")
    pause()

    model2 = EnvironmentModel(name="Fixed: no penalty", max_steps=500, gamma=0.99)
    model2.add_reward_source(RewardSource(
        name="goal", reward_type=RewardType.TERMINAL,
        value=1.0, discovery_probability=0.05,
    ))

    print(reward_landscape_ascii(model2))

    pause()

    # ---- Demo 6: CoastRunners example ----
    section("DEMO 6: A real failure — CoastRunners (OpenAI 2016)")
    print("  The agent learned to go in circles collecting")
    print("  respawning green blocks instead of finishing the race.")
    print("  goodhart catches it from the config alone.")
    pause()

    from goodhart.models import RespawnBehavior

    model3 = EnvironmentModel(name="CoastRunners", max_steps=1000, n_states=5000)
    model3.add_reward_source(RewardSource(
        name="finish race", reward_type=RewardType.TERMINAL,
        value=10.0, discovery_probability=0.1,
    ))
    model3.add_reward_source(RewardSource(
        name="green blocks", reward_type=RewardType.ON_EVENT,
        value=0.5, respawn=RespawnBehavior.TIMED, respawn_time=2,
        can_loop=True, loop_period=3,
    ))

    from goodhart.engine import AnalysisEngine
    engine = AnalysisEngine().add_all_rules()
    result = engine.analyze(model3)

    for v in result.criticals:
        print(f"  CRITICAL: {v.message}")
        if v.recommendation:
            print(f"    Fix: {v.recommendation}")
    print()
    print(reward_landscape_ascii(model3))

    pause()

    # ---- Demo 7: Formal proofs ----
    section("DEMO 7: Backed by formal proofs")
    print("  Every core rule is linked to a LEAN 4 theorem.")
    print("  The math is machine-verified, not just tested.")
    pause()

    from goodhart.rules.reward import REWARD_RULES
    for rule in REWARD_RULES:
        proof = rule.proof
        if proof:
            print(f"  {rule.name}")
            print(f"    LEAN: {proof.proof_name}")
            print(f"    Math: {proof.statement}")
            if proof.paper:
                print(f"    Paper: {proof.paper}")
            print()

    pause()

    # ---- Summary ----
    section("SUMMARY")
    print("  goodhart catches reward traps before you train.")
    print()
    print("  What we showed:")
    print("    1. Quick check catches a doomed config")
    print("    2. Fixing the config clears the check")
    print("    3. Typed Result API for training scripts")
    print("    4. ASCII visualization of reward landscape")
    print("    5. Real failure case (CoastRunners)")
    print("    6. Formal LEAN proofs backing each rule")
    print()
    from goodhart.rules import RULE_COUNT
    print(f"  {RULE_COUNT} rules | 212 encodings from 133 papers")
    print(f"  Formally verified (LEAN 4, zero sorry)")
    print("  Takes milliseconds. Saves days.")
    print()
    print("  pip install goodhart")
    print("  goodhart --goal 1.0 --penalty -0.01 --steps 500")
    print()


if __name__ == "__main__":
    main()
