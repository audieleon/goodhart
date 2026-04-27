/-
  Skalse et al. 2022 — Theorem 1: Impossibility on Open Sets
  arXiv: 2209.13085, Section 5.1

  WHAT THIS PROVES: If the policy set is an open set in R^d (or any
  open subset), then no non-trivial unhackable reward pair exists.
  In other words: for "generic" policy sets, reward hacking is inevitable.

  WHY IT MATTERS: This is the pessimistic result. It says that for
  MOST reward function pairs, there exist policies that score high
  on the proxy but low on the true reward. The only escape is:
  (a) the policy set is NOT generic (finite, structured), or
  (b) the proxy is equivalent to the true reward (Theorem 3).

  PROOF STRATEGY: Perturbation argument. Given any non-trivially
  unhackable pair (R₁, R₂), we find a nearby policy F' that breaks
  the ordering. Since the policy set is open, we can perturb in any
  direction — and the "non-equivalent" condition guarantees a
  direction exists that reverses R₂'s ordering while preserving R₁'s.

  This uses Mathlib's metric space and topology libraries (hence
  the additional imports).
-/

import GoodhartProofs.Skalse
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Topology.Algebra.Ring.Basic

open Finset Metric Filter

noncomputable section

namespace Skalse

variable {d : ℕ}

def unhackableS (R₁ R₂ : Fin d → ℝ) (U : Set (Fin d → ℝ)) : Prop :=
  ∀ F₁ ∈ U, ∀ F₂ ∈ U,
    value R₁ F₁ < value R₁ F₂ → value R₂ F₁ ≤ value R₂ F₂

def rewardEquivS (R₁ R₂ : Fin d → ℝ) (U : Set (Fin d → ℝ)) : Prop :=
  ∀ F₁ ∈ U, ∀ F₂ ∈ U,
    (value R₁ F₁ ≤ value R₁ F₂ ↔ value R₂ F₁ ≤ value R₂ F₂)

def trivialRewardS (R : Fin d → ℝ) (U : Set (Fin d → ℝ)) : Prop :=
  ∀ F₁ ∈ U, ∀ F₂ ∈ U, value R F₁ = value R F₂

theorem unhackableS_symm (R₁ R₂ : Fin d → ℝ) (U : Set (Fin d → ℝ))
    (h : unhackableS R₁ R₂ U) : unhackableS R₂ R₁ U := by
  intro F₁ hF₁ F₂ hF₂ hlt
  by_contra hgt; push_neg at hgt
  linarith [h F₂ hF₂ F₁ hF₁ hgt]

theorem value_add_point (R x v : Fin d → ℝ) (δ : ℝ) :
    value R (fun i => x i + δ * v i) = value R x + δ * value R v := by
  simp only [value]
  simp_rw [show ∀ i, R i * (x i + δ * v i) = R i * x i + δ * (R i * v i)
    from fun i => by ring]
  rw [Finset.sum_add_distrib, Finset.mul_sum]

/-- Small perturbations of a point in an open set stay inside. -/
theorem exists_small_perturbation
    (U : Set (Fin d → ℝ)) (hU : IsOpen U)
    (x₀ : Fin d → ℝ) (hx₀ : x₀ ∈ U)
    (w : Fin d → ℝ) (c : ℝ) (hc : 0 < c) :
    ∃ δ : ℝ, 0 < δ ∧ δ < c ∧
      (fun i => x₀ i + δ * w i) ∈ U ∧
      (fun i => x₀ i - δ * w i) ∈ U := by
  -- The maps δ ↦ x₀ + δw and δ ↦ x₀ - δw are continuous
  have h1 : Continuous (fun δ : ℝ => (fun i : Fin d => x₀ i + δ * w i)) :=
    continuous_pi fun i => continuous_const.add (continuous_id.mul continuous_const)
  have h2 : Continuous (fun δ : ℝ => (fun i : Fin d => x₀ i - δ * w i)) :=
    continuous_pi fun i => continuous_const.sub (continuous_id.mul continuous_const)
  -- At δ = 0, both give x₀ ∈ U
  have h10 : (fun i : Fin d => x₀ i + (0 : ℝ) * w i) = x₀ := by ext i; simp
  have h20 : (fun i : Fin d => x₀ i - (0 : ℝ) * w i) = x₀ := by ext i; simp
  -- Preimages of U contain 0, so they're neighborhoods of 0
  have n1 : (fun δ : ℝ => (fun i => x₀ i + δ * w i)) ⁻¹' U ∈ nhds (0 : ℝ) := by
    apply h1.continuousAt.preimage_mem_nhds
    rw [h10]; exact hU.mem_nhds hx₀
  have n2 : (fun δ : ℝ => (fun i => x₀ i - δ * w i)) ⁻¹' U ∈ nhds (0 : ℝ) := by
    apply h2.continuousAt.preimage_mem_nhds
    rw [h20]; exact hU.mem_nhds hx₀
  -- Intersection is still a neighborhood of 0
  rw [Metric.mem_nhds_iff] at n1 n2
  obtain ⟨ε₁, hε₁, hb₁⟩ := n1
  obtain ⟨ε₂, hε₂, hb₂⟩ := n2
  -- Pick δ in both balls and < c
  set δ := min (min (ε₁ / 2) (ε₂ / 2)) (c / 2)
  have hδ_pos : 0 < δ := by positivity
  refine ⟨δ, hδ_pos, ?_, ?_, ?_⟩
  · -- δ < c
    exact lt_of_le_of_lt (min_le_right _ _) (by linarith)
  · -- x₀ + δw ∈ U
    apply hb₁
    rw [Metric.mem_ball, Real.dist_eq, sub_zero, abs_of_pos hδ_pos]
    calc δ ≤ min (ε₁ / 2) (ε₂ / 2) := min_le_left _ _
      _ ≤ ε₁ / 2 := min_le_left _ _
      _ < ε₁ := by linarith
  · -- x₀ - δw ∈ U
    apply hb₂
    rw [Metric.mem_ball, Real.dist_eq, sub_zero, abs_of_pos hδ_pos]
    calc δ ≤ min (ε₁ / 2) (ε₂ / 2) := min_le_left _ _
      _ ≤ ε₂ / 2 := min_le_right _ _
      _ < ε₂ := by linarith

/-- Skalse 2022 Theorem 1.
    On open sets, unhackable + both non-trivial → equivalent. -/
theorem skalse_theorem1
    (R₁ R₂ : Fin d → ℝ) (U : Set (Fin d → ℝ))
    (hU : IsOpen U)
    (h_unhack : unhackableS R₁ R₂ U)
    (hR₁_nt : ¬trivialRewardS R₁ U)
    (hR₂_nt : ¬trivialRewardS R₂ U) :
    rewardEquivS R₁ R₂ U := by
  have h_unhack' := unhackableS_symm R₁ R₂ U h_unhack
  by_contra h_ne
  simp only [rewardEquivS] at h_ne; push_neg at h_ne
  obtain ⟨x₀, hx₀, y₀, hy₀, h_bic⟩ := h_ne
  -- h_bic : (R₁ ≤ ∧ R₂ >) ∨ (R₁ > ∧ R₂ ≤)
  rcases h_bic with ⟨h1, hR2⟩ | ⟨h1, hR2⟩
  ·
    -- Must be R₁ equality (strict < would give R₂ ≤ by unhackability)
    have heq : value R₁ x₀ = value R₁ y₀ := by
      rcases eq_or_lt_of_le h1 with h | h; exact h
      exact absurd (h_unhack x₀ hx₀ y₀ hy₀ h) (not_le.mpr hR2)
    -- R₁ non-trivial: find z with different R₁ value
    simp only [trivialRewardS] at hR₁_nt; push_neg at hR₁_nt
    obtain ⟨p, hp, q, hq, hpq⟩ := hR₁_nt
    obtain ⟨z, hz, hzne⟩ : ∃ z ∈ U, value R₁ z ≠ value R₁ y₀ := by
      by_cases h : value R₁ p = value R₁ y₀
      · exact ⟨q, hq, by rw [← h]; exact Ne.symm hpq⟩
      · exact ⟨p, hp, h⟩
    -- w = z - y₀ has nonzero R₁ value
    set w := fun i => z i - y₀ i
    have hw : value R₁ w ≠ 0 := by
      intro heq2; apply hzne
      have h1 := value_add_point R₁ y₀ w 1
      simp only [one_mul] at h1
      have hw2 : (fun i => y₀ i + w i) = z := funext fun i => by simp [w]
      rw [hw2] at h1; linarith
    set gap := value R₂ x₀ - value R₂ y₀
    have hgap : 0 < gap := by linarith
    -- Get perturbation (need c > 0)
    have hc : 0 < gap / (|value R₂ w| + 1) := div_pos hgap (by positivity)
    obtain ⟨δ, hδ, hδc, hy_plus, hy_minus⟩ :=
      exists_small_perturbation U hU y₀ hy₀ w _ hc
    -- The key bound: δ · |value R₂ w| < gap
    have hδ_bound : δ * |value R₂ w| < gap := by
      calc δ * |value R₂ w| ≤ δ * (|value R₂ w| + 1) :=
            mul_le_mul_of_nonneg_left (by linarith) (le_of_lt hδ)
        _ < gap := by
            have h_pos : (0 : ℝ) < |value R₂ w| + 1 := by positivity
            calc δ * (|value R₂ w| + 1) < gap / (|value R₂ w| + 1) * (|value R₂ w| + 1) :=
                  mul_lt_mul_of_pos_right hδc h_pos
              _ = gap := div_mul_cancel₀ gap (ne_of_gt h_pos)
    -- Choose direction based on sign of value R₁ w
    by_cases hs : 0 < value R₁ w
    · -- +δ direction
      have hR1 : value R₁ x₀ < value R₁ (fun i => y₀ i + δ * w i) := by
        rw [value_add_point, heq]; linarith [mul_pos hδ hs]
      have hR2 : value R₂ (fun i => y₀ i + δ * w i) < value R₂ x₀ := by
        rw [value_add_point]; nlinarith [abs_nonneg (value R₂ w),
          neg_abs_le (value R₂ w), le_abs_self (value R₂ w)]
      exact absurd (h_unhack x₀ hx₀ _ hy_plus hR1) (not_le.mpr hR2)
    · -- -δ direction
      push_neg at hs
      have hlt : value R₁ w < 0 := lt_of_le_of_ne hs hw
      have hR1 : value R₁ x₀ < value R₁ (fun i => y₀ i - δ * w i) := by
        have : (fun i => y₀ i - δ * w i) = (fun i => y₀ i + (-δ) * w i) := by ext i; ring
        rw [this, value_add_point, heq]; nlinarith
      have hR2 : value R₂ (fun i => y₀ i - δ * w i) < value R₂ x₀ := by
        have : (fun i => y₀ i - δ * w i) = (fun i => y₀ i + (-δ) * w i) := by ext i; ring
        rw [this, value_add_point]
        nlinarith [abs_nonneg (value R₂ w), neg_abs_le (value R₂ w), le_abs_self (value R₂ w)]
      exact absurd (h_unhack x₀ hx₀ _ hy_minus hR1) (not_le.mpr hR2)
  · -- Symmetric: R₁ x₀ > y₀, R₂ x₀ ≤ y₀
    -- h1 : value R₁ y₀ < value R₁ x₀, hR2 : value R₂ x₀ ≤ value R₂ y₀
    have heq : value R₂ x₀ = value R₂ y₀ := by
      rcases eq_or_lt_of_le hR2 with h | h; exact h
      exact absurd (h_unhack' x₀ hx₀ y₀ hy₀ h) (not_le.mpr h1)
    simp only [trivialRewardS] at hR₂_nt; push_neg at hR₂_nt
    obtain ⟨p, hp, q, hq, hpq⟩ := hR₂_nt
    obtain ⟨z, hz, hzne⟩ : ∃ z ∈ U, value R₂ z ≠ value R₂ y₀ := by
      by_cases h : value R₂ p = value R₂ y₀
      · exact ⟨q, hq, by rw [← h]; exact Ne.symm hpq⟩
      · exact ⟨p, hp, h⟩
    set w := fun i => z i - y₀ i
    have hw : value R₂ w ≠ 0 := by
      intro heq2; apply hzne
      have h1' := value_add_point R₂ y₀ w 1
      simp only [one_mul] at h1'
      have h2 : (fun i => y₀ i + w i) = z := funext fun i => by simp [w]
      rw [h2] at h1'; linarith
    set gap := value R₁ x₀ - value R₁ y₀
    have hgap : 0 < gap := by linarith
    have hc : 0 < gap / (|value R₁ w| + 1) := div_pos hgap (by positivity)
    obtain ⟨δ, hδ, hδc, hy_plus, hy_minus⟩ :=
      exists_small_perturbation U hU y₀ hy₀ w _ hc
    have hδ_bound : δ * |value R₁ w| < gap := by
      calc δ * |value R₁ w| ≤ δ * (|value R₁ w| + 1) :=
            mul_le_mul_of_nonneg_left (by linarith) (le_of_lt hδ)
        _ < gap := by
            have h_pos : (0 : ℝ) < |value R₁ w| + 1 := by positivity
            calc δ * (|value R₁ w| + 1) < gap / (|value R₁ w| + 1) * (|value R₁ w| + 1) :=
                  mul_lt_mul_of_pos_right hδc h_pos
              _ = gap := div_mul_cancel₀ gap (ne_of_gt h_pos)
    by_cases hs : 0 < value R₂ w
    · have hR2 : value R₂ x₀ < value R₂ (fun i => y₀ i + δ * w i) := by
        rw [value_add_point, heq]; linarith [mul_pos hδ hs]
      have hR1 : value R₁ (fun i => y₀ i + δ * w i) < value R₁ x₀ := by
        rw [value_add_point]
        nlinarith [abs_nonneg (value R₁ w), neg_abs_le (value R₁ w), le_abs_self (value R₁ w)]
      exact absurd (h_unhack' x₀ hx₀ _ hy_plus hR2) (not_le.mpr hR1)
    · push_neg at hs
      have hlt : value R₂ w < 0 := lt_of_le_of_ne hs hw
      have hR2 : value R₂ x₀ < value R₂ (fun i => y₀ i - δ * w i) := by
        have : (fun i => y₀ i - δ * w i) = (fun i => y₀ i + (-δ) * w i) := by ext i; ring
        rw [this, value_add_point, heq]; nlinarith
      have hR1 : value R₁ (fun i => y₀ i - δ * w i) < value R₁ x₀ := by
        have : (fun i => y₀ i - δ * w i) = (fun i => y₀ i + (-δ) * w i) := by ext i; ring
        rw [this, value_add_point]
        nlinarith [abs_nonneg (value R₁ w), neg_abs_le (value R₁ w), le_abs_self (value R₁ w)]
      exact absurd (h_unhack' x₀ hx₀ _ hy_minus hR2) (not_le.mpr hR1)

end Skalse

end
