/-
  Undiscounted MDP (γ = 1) — Ng 1999 Shaping for Episodic Tasks

  Our main MDP formalization requires 0 < γ < 1 for the Banach fixed
  point theorem (contraction needs γ < 1). But many real environments
  use γ = 1 (episodic tasks with absorbing terminal state).

  Ng 1999 handles γ = 1 separately: the shaped reward is Φ(s') - Φ(s)
  (no γ multiplier on Φ(s')). This telescopes over any trajectory
  ending in the absorbing state s₀ with Φ(s₀) = 0.

  This module proves the γ = 1 shaping result WITHOUT the Banach
  contraction argument. Instead, we axiomatize the fixed-point
  property (V* = T*V*) as a hypothesis. This is mathematically
  weaker (we don't prove V* exists) but sufficient for the shaping
  result, and it avoids the machinery needed for proper MDPs.

  References: Ng et al. 1999, Theorem 1 (γ = 1 case)
-/

import GoodhartProofs.MDP.Bellman

open Finset ENNReal

noncomputable section

namespace MDP

/-!
## Helpers
-/

private theorem pmf_sum_one {α : Type*} [Fintype α] (p : PMF α) :
    ∑ s : α, (p s).toReal = 1 := by
  rw [← ENNReal.toReal_sum (fun s _ =>
    ne_top_of_le_ne_top one_ne_top (PMF.tsum_coe p ▸ ENNReal.le_tsum s))]
  have : ∑ s ∈ Finset.univ, p s = 1 := by
    rw [← PMF.tsum_coe p]
    exact (tsum_eq_sum (fun s hs => absurd (Finset.mem_univ s) hs)).symm
  rw [this]; rfl

private theorem sup'_sub_const {ι : Type*}
    (s : Finset ι) (hne : s.Nonempty) (f : ι → ℝ) (c : ℝ) :
    s.sup' hne (fun i => f i - c) = s.sup' hne f - c := by
  apply le_antisymm
  · exact Finset.sup'_le hne _ (fun i hi => sub_le_sub_right (Finset.le_sup' f hi) c)
  · suffices h : s.sup' hne f ≤ s.sup' hne (fun i => f i - c) + c by linarith
    apply Finset.sup'_le; intro i hi
    linarith [Finset.le_sup' (fun i => f i - c) hi]

/-!
## Raw Bellman operators (γ = 1)
-/

def qValueU {S A : Type} [Fintype S]
    (T : S → A → PMF S) (R : S → A → S → ℝ) (V : S → ℝ)
    (s : S) (a : A) : ℝ :=
  ∑ s' : S, (T s a s').toReal * (R s a s' + V s')

def bellmanOptOpU {S A : Type} [Fintype S] [Fintype A]
    [DecidableEq A] [Inhabited A]
    (T : S → A → PMF S) (R : S → A → S → ℝ) (V : S → ℝ) (s : S) : ℝ :=
  Finset.sup' Finset.univ ⟨default, Finset.mem_univ _⟩ (qValueU T R V s)

/-!
## Proper Undiscounted MDP
-/

structure ProperUndiscountedMDP where
  S : Type
  A : Type
  [instFintypeS : Fintype S]
  [instFintypeA : Fintype A]
  [instDecEqS : DecidableEq S]
  [instDecEqA : DecidableEq A]
  [instInhabitedA : Inhabited A]
  T : S → A → PMF S
  R : S → A → S → ℝ
  s₀ : S
  h_absorbing : ∀ a, T s₀ a = PMF.pure s₀
  h_reward_s0 : ∀ a, R s₀ a s₀ = 0
  vStar : S → ℝ
  h_vstar_fixed : ∀ s, bellmanOptOpU T R vStar s = vStar s
  h_bellman_unique : ∀ V : S → ℝ,
    (∀ s, bellmanOptOpU T R V s = V s) → V = vStar

attribute [instance] ProperUndiscountedMDP.instFintypeS
attribute [instance] ProperUndiscountedMDP.instFintypeA
attribute [instance] ProperUndiscountedMDP.instDecEqS
attribute [instance] ProperUndiscountedMDP.instDecEqA
attribute [instance] ProperUndiscountedMDP.instInhabitedA

variable (M : ProperUndiscountedMDP) (Φ : M.S → ℝ)

/-!
## Shaped Reward (γ = 1)
-/

def shapedRU (s : M.S) (a : M.A) (s' : M.S) : ℝ :=
  M.R s a s' + Φ s' - Φ s

theorem shapedRU_s0 (hΦ : Φ M.s₀ = 0) (a : M.A) :
    shapedRU M Φ M.s₀ a M.s₀ = 0 := by
  simp [shapedRU, M.h_reward_s0 a, hΦ]

/-!
## Algebraic Core (γ = 1)
-/

theorem shaped_qValueU_eq (V : M.S → ℝ) (s : M.S) (a : M.A) :
    qValueU M.T (shapedRU M Φ) V s a =
    qValueU M.T M.R (fun s' => V s' + Φ s') s a - Φ s := by
  simp only [qValueU, shapedRU]
  have step : ∀ x, (M.T s a x).toReal * (M.R s a x + Φ x - Φ s + V x) =
      (M.T s a x).toReal * (M.R s a x + (V x + Φ x)) -
      (M.T s a x).toReal * Φ s := by intro x; ring
  simp_rw [step, Finset.sum_sub_distrib]
  rw [show ∑ x ∈ univ, (M.T s a x).toReal * Φ s =
      Φ s * ∑ x ∈ univ, (M.T s a x).toReal from by
    rw [Finset.mul_sum]; congr 1; ext x; ring]
  rw [pmf_sum_one (M.T s a)]; ring

theorem shaped_bellmanU_eq (V : M.S → ℝ) (s : M.S) :
    bellmanOptOpU M.T (shapedRU M Φ) V s =
    bellmanOptOpU M.T M.R (fun s' => V s' + Φ s') s - Φ s := by
  simp only [bellmanOptOpU]
  conv_lhs => arg 3; ext a; rw [shaped_qValueU_eq M Φ V s a]
  exact sup'_sub_const _ _ _ _

/-!
## Ng 1999, Theorem 1 (γ = 1)
-/

theorem shaped_bellman_fixed_pointU :
    ∀ s, bellmanOptOpU M.T (shapedRU M Φ) (fun s => M.vStar s - Φ s) s =
    M.vStar s - Φ s := by
  intro s; rw [shaped_bellmanU_eq]
  have : (fun s' => (M.vStar s' - Φ s') + Φ s') = M.vStar := by ext s'; ring
  rw [this, M.h_vstar_fixed s]

theorem shaped_bellmanU_unique (V : M.S → ℝ)
    (h : ∀ s, bellmanOptOpU M.T (shapedRU M Φ) V s = V s) :
    V = fun s => M.vStar s - Φ s := by
  have h_shifted : ∀ s, bellmanOptOpU M.T M.R (fun s' => V s' + Φ s') s =
      V s + Φ s := by
    intro s
    have := shaped_bellmanU_eq M Φ V s
    rw [h s] at this; linarith
  have := M.h_bellman_unique (fun s => V s + Φ s) h_shifted
  ext s; have := congr_fun this s; linarith

/-- The shaped MDP is proper: V*_M - Φ is its unique fixed point.
    This IS the Ng 1999 Theorem 1 for γ = 1. -/
def shapedProperMDPU (hΦ : Φ M.s₀ = 0) : ProperUndiscountedMDP where
  S := M.S; A := M.A; T := M.T
  R := shapedRU M Φ
  s₀ := M.s₀
  h_absorbing := M.h_absorbing
  h_reward_s0 := shapedRU_s0 M Φ hΦ
  vStar := fun s => M.vStar s - Φ s
  h_vstar_fixed := shaped_bellman_fixed_pointU M Φ
  h_bellman_unique := shaped_bellmanU_unique M Φ

/-- Policy invariance under γ = 1 shaping. -/
theorem ng_shaping_preserves_optimalU (s : M.S) (a₁ a₂ : M.A) :
    qValueU M.T M.R M.vStar s a₁ ≤ qValueU M.T M.R M.vStar s a₂ ↔
    qValueU M.T (shapedRU M Φ) (fun s => M.vStar s - Φ s) s a₁ ≤
    qValueU M.T (shapedRU M Φ) (fun s => M.vStar s - Φ s) s a₂ := by
  rw [shaped_qValueU_eq, shaped_qValueU_eq]
  have : (fun s' => (M.vStar s' - Φ s') + Φ s') = M.vStar := by ext s'; ring
  rw [this]; constructor <;> intro h <;> linarith

/-!
## Summary

Ng 1999 Theorem 1 is now complete for γ ∈ (0, 1]:
- γ ∈ (0, 1): Shaping.lean — V* constructed via Banach FPT
- γ = 1: this file — V* assumed via structure fields (properness
  guarantees existence but is not formally proved; the structure
  fields serve as axioms for the shaping theorem)

Fidelity to Ng 1999 (γ = 1 case):
- Absorbing s₀ with T(s₀,a) = pure(s₀): exact match ✓
- Φ(s₀) = 0 convention: explicit hypothesis ✓
- F(s,a,s') = Φ(s') - Φ(s): exact match (Eq. 2 with γ = 1) ✓
- V*_{M'} = V*_M - Φ: proved via fixed point + uniqueness ✓
- Shaped MDP is proper: proved (shapedProperMDPU) ✓
- Policy invariance: proved (ng_shaping_preserves_optimalU) ✓
-/

end MDP

end
