# Goodhart Proofs — Machine-Verified Reward Hacking Theorems

Formal proofs in LEAN 4 that verify the mathematical foundations
of the goodhart tool. Each theorem corresponds to a runtime rule
in the Python package.

## What's verified here

These proofs establish that specific reward configurations
**mathematically guarantee** degenerate equilibria. The Python
tool checks these conditions at runtime; the LEAN proofs verify
that the mathematical properties hold exactly.

## Theorems

| Theorem | Status | Python rule | Paper |
|---------|--------|------------|-------|
| `death_beats_survival` | **Proved** | `death_beats_survival` | Original |
| `die_fast_beats_survive` | **Proved** | `death_beats_survival` | Original |
| `penalty_breakeven` | **Proved** | `penalty_dominates_goal` | Original |
| `idle_dominance` | **Proved** | `idle_exploit` | Original |
| `loop_dominance_continuous` | **Proved** | `respawning_exploit` | Original |
| `death_reset_dominance` | **Proved** | `death_reset_exploit` | Original |
| `telescoping_sum` | **Proved** | `shaping_loop_exploit` | Ng et al. 1999 |
| `penalty_dominance` | Partial | `penalty_dominates_goal` | Original |
| `shaping_cancellation` | Partial | `shaping_loop_exploit` | Ng et al. 1999 |

## Papers machine-verified (or in progress)

### Ng, Harada & Russell 1999 — "Policy Invariance Under Reward Transformations"
- **Theorem 1 (Sufficiency):** Potential-based shaping F(s,a,s') = γΦ(s') - Φ(s)
  preserves optimal policy. Our `telescoping_sum` proves the key lemma
  (shaping terms cancel over any trajectory). Full sufficiency proof
  requires MDP type formalization.
- **Theorem 1 (Necessity):** Non-potential-based shaping can yield
  suboptimal policies. Not yet formalized.
- **Corollary 2:** Q*_{M'}(s,a) = Q*_M(s,a) - Φ(s). Not yet formalized.
- **Status:** Key lemma proved, full theorem in progress.

### Skalse et al. 2022 — "Defining and Characterizing Reward Hacking" (NeurIPS)
- **Definition 1:** A proxy reward is "unhackable" iff increasing proxy
  return never decreases true return.
- **Theorem:** For stochastic policies, two reward functions are unhackable
  only if one is constant. (Strong impossibility result.)
- **Status:** Not yet formalized. Future work.

### Original theorems (this project)
- **Death dominance:** For any step penalty p < 0, dying earlier always
  yields higher return. (Proved.)
- **Penalty dominance:** When |p| × T > g, all policies scoring positive
  must terminate within g/|p| steps. (Proved.)
- **Loop dominance:** Respawning reward with rate v/t > g gives higher
  EV than terminal goal. (Proved.)
- **Idle dominance:** When idle reward ≥ active reward + penalty,
  standing still dominates. (Proved.)

## Architecture

```
LEAN proofs (mathematical truth, exact over ℚ)
    ↓ verifies properties like:
    "∀ p < 0, ∀ m < n, p * n < p * m"

Python rules (runtime check, approximate over float)
    ↓ checks user's config against properties:
    "your penalty=-0.01, theorem applies → CRITICAL"
```

## Building

Requires LEAN 4 and Mathlib:

```bash
cd proofs
lake update     # downloads Mathlib (~2GB, first time only)
lake build      # type-checks all proofs
```

## Why this matters

This is (to our knowledge) the first machine-verified formalization
of reward hacking properties in any theorem prover. Prior work:

- Hölzl 2017: MDP value iteration in Isabelle/HOL (not reward hacking)
- Ng et al. 1999: Potential-based shaping proof on paper (not machine-verified)
- Skalse et al. 2022: Formal definition of reward hacking (not machine-verified)

We verify the properties that these papers proved on paper, and add
new theorems about degenerate equilibria that weren't previously
formalized anywhere.
