/-
  Finite Markov Decision Process — Core Definitions

  First LEAN 4 formalization of finite MDPs with the structure
  needed for reward shaping (Ng 1999) and reward hacking (Skalse 2022).

  No MDP definitions exist in Mathlib. This module provides:
  - FiniteMDP structure with finite state/action spaces
  - Deterministic and stochastic policies
  - Bellman operators (in Bellman.lean)
  - Shaped MDP for potential-based reward shaping
-/

import Mathlib.Tactic
import Mathlib.Probability.ProbabilityMassFunction.Basic
import Mathlib.Probability.ProbabilityMassFunction.Monad

open Finset ENNReal

noncomputable section

namespace MDP

/-!
## Finite MDP Structure
-/

/-- A finite Markov Decision Process.
    S and A are finite types representing states and actions.
    T is a transition kernel mapping (state, action) pairs to
    probability distributions over next states.
    R is a 3-argument reward function R(s, a, s') — needed for
    Ng 1999 shaping where F(s,a,s') = γΦ(s') - Φ(s).
    γ is the discount factor, strictly between 0 and 1. -/
structure FiniteMDP where
  S : Type
  A : Type
  [instFintypeS : Fintype S]
  [instFintypeA : Fintype A]
  [instDecEqS : DecidableEq S]
  [instDecEqA : DecidableEq A]
  [instInhabitedA : Inhabited A]
  T : S → A → PMF S
  γ : ℝ
  R : S → A → S → ℝ
  hγ_pos : 0 < γ
  hγ_lt : γ < 1

attribute [instance] FiniteMDP.instFintypeS FiniteMDP.instFintypeA
attribute [instance] FiniteMDP.instDecEqS FiniteMDP.instDecEqA
attribute [instance] FiniteMDP.instInhabitedA

variable (M : FiniteMDP)

/-- Discount factor is non-negative.
    Follows immediately from 0 < γ (every positive number is non-negative).
    Used wherever we need γ ≥ 0 as a hypothesis (e.g., multiplying inequalities). -/
theorem FiniteMDP.hγ_nonneg : 0 ≤ M.γ := le_of_lt M.hγ_pos

/-- Discount factor is at most 1.
    Follows immediately from γ < 1 (every number less than 1 is at most 1).
    Used in contraction proofs where we need ‖γ‖ ≤ 1. -/
theorem FiniteMDP.hγ_le_one : M.γ ≤ 1 := le_of_lt M.hγ_lt

/-!
## Policies
-/

/-- A stochastic policy maps states to distributions over actions.
    Needed for Skalse (topology of policy space). -/
def StochPolicy (M : FiniteMDP) := M.S → PMF M.A

/-- A deterministic policy maps states to actions.
    Sufficient for Ng 1999 (optimal policies are deterministic). -/
def DetPolicy (M : FiniteMDP) := M.S → M.A

/-- Embed a deterministic policy into a stochastic one
    using PMF.pure (point mass). -/
def DetPolicy.toStoch (π : DetPolicy M) : StochPolicy M :=
  fun s => PMF.pure (π s)

/-!
## Expected Reward
-/

/-- Expected immediate reward: E_{s'~T(s,a)}[R(s,a,s')].
    Finite sum since S is Fintype. -/
def expectedReward (s : M.S) (a : M.A) : ℝ :=
  ∑ s' : M.S, (M.T s a s').toReal * M.R s a s'

/-!
## Shaped MDP (Ng et al. 1999)

Given a potential function Φ : S → ℝ, the shaped MDP replaces
R(s,a,s') with R(s,a,s') + γΦ(s') - Φ(s).

Theorem 1 of Ng 1999 proves this preserves optimal policies.
-/

/-- The shaped MDP under potential-based shaping.
    R'(s,a,s') = R(s,a,s') + γΦ(s') - Φ(s).
    All other MDP components (states, actions, transitions, discount)
    remain unchanged. -/
def shapedMDP (Φ : M.S → ℝ) : FiniteMDP where
  S := M.S
  A := M.A
  T := M.T
  γ := M.γ
  R := fun s a s' => M.R s a s' + M.γ * Φ s' - Φ s
  hγ_pos := M.hγ_pos
  hγ_lt := M.hγ_lt

/-- The shaped MDP has the same state type. -/
theorem shapedMDP_S (Φ : M.S → ℝ) : (shapedMDP M Φ).S = M.S := rfl

/-- The shaped MDP has the same action type. -/
theorem shapedMDP_A (Φ : M.S → ℝ) : (shapedMDP M Φ).A = M.A := rfl

/-- The shaped MDP has the same transitions. -/
theorem shapedMDP_T (Φ : M.S → ℝ) : (shapedMDP M Φ).T = M.T := rfl

/-- The shaped MDP has the same discount. -/
theorem shapedMDP_γ (Φ : M.S → ℝ) : (shapedMDP M Φ).γ = M.γ := rfl

end MDP

end
