/-
  Skalse 2022 Theorem 3 — Finrank Formulation

  The dimension characterization: non-trivial simplification exists
  iff finrank(Z⊥) ≥ 2, where Z is the span of within-class difference
  vectors and Z⊥ is the equality-preserving subspace.

  References: Skalse et al. 2022, Theorem 3 (Section 5.2)
-/

import GoodhartProofs.Skalse
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.InnerProductSpace.Projection.FiniteDimensional

open Module Submodule

noncomputable section

variable {d : ℕ}

/-!
## Type Bridge: value = inner product

Our `Skalse.value R F = ∑ i, R i * F i` on `Fin d → ℝ` equals the
inner product `⟪R, F⟫` on `EuclideanSpace ℝ (Fin d)`. This connects
the abstract Skalse theorems to the finrank machinery.
-/

/-- The Skalse value function equals the EuclideanSpace inner product. -/
theorem value_eq_inner (R F : Fin d → ℝ) :
    Skalse.value R F = @inner ℝ (EuclideanSpace ℝ (Fin d)) _
      (WithLp.toLp (p := 2) R) (WithLp.toLp (p := 2) F) := by
  simp only [Skalse.value, PiLp.inner_apply]
  congr 1; ext i; simp [inner, mul_comm]

/-!
## Dimension identity
-/

/-- finrank(Z) + finrank(Z⊥) = d on EuclideanSpace ℝ (Fin d). -/
theorem eq_preserving_dim
    (Z : Submodule ℝ (EuclideanSpace ℝ (Fin d))) :
    finrank ℝ ↥Z + finrank ℝ ↥Zᗮ = d := by
  have := Submodule.finrank_add_finrank_orthogonal Z
  rwa [finrank_euclideanSpace_fin] at this

/-!
## "Only if": finrank(Z⊥) ≤ 1 → proportionality

If the equality-preserving subspace is at most 1-dimensional
and contains a nonzero R₁, then every R₂ in it is a scalar
multiple of R₁. This prevents non-trivial simplification.
-/

/-- In a subspace of dimension ≤ 1, every vector is proportional
    to any nonzero member. -/
theorem proportional_of_finrank_le_one
    (W : Submodule ℝ (EuclideanSpace ℝ (Fin d)))
    (R₁ : EuclideanSpace ℝ (Fin d))
    (hR₁_ne : R₁ ≠ 0) (hR₁_mem : R₁ ∈ W)
    (h_dim : finrank ℝ ↥W ≤ 1)
    (R₂ : EuclideanSpace ℝ (Fin d)) (hR₂_mem : R₂ ∈ W) :
    ∃ c : ℝ, R₂ = c • R₁ := by
  -- span{R₁} has finrank 1
  have h_span_dim : finrank ℝ ↥(ℝ ∙ R₁) = 1 := finrank_span_singleton hR₁_ne
  -- span{R₁} ≤ W
  have h_span_le : ℝ ∙ R₁ ≤ W :=
    span_le.mpr (Set.singleton_subset_iff.mpr hR₁_mem)
  -- finrank(W) ≤ 1 = finrank(span{R₁}), so W = span{R₁}
  have h_eq : ℝ ∙ R₁ = W :=
    eq_of_le_of_finrank_le h_span_le (by omega)
  -- R₂ ∈ W = span{R₁}, so R₂ = c • R₁
  rw [← h_eq] at hR₂_mem
  rw [mem_span_singleton] at hR₂_mem
  obtain ⟨c, hc⟩ := hR₂_mem
  exact ⟨c, hc.symm⟩

/-!
## "If": finrank(Z⊥) ≥ 2 → independent witness exists

If the equality-preserving subspace has dimension ≥ 2 and
contains nonzero R₁, then there exists w ∈ Z⊥ independent
from R₁ (i.e., w is not a scalar multiple of R₁).
-/

/-- In a subspace of dimension ≥ 2 containing nonzero R₁,
    there exists a vector not proportional to R₁. -/
theorem exists_independent_of_finrank_ge_two
    (W : Submodule ℝ (EuclideanSpace ℝ (Fin d)))
    (R₁ : EuclideanSpace ℝ (Fin d))
    (hR₁_ne : R₁ ≠ 0) (hR₁_mem : R₁ ∈ W)
    (h_dim : 2 ≤ finrank ℝ ↥W) :
    ∃ w ∈ W, ¬∃ c : ℝ, w = c • R₁ := by
  -- span{R₁} has finrank 1 < 2 ≤ finrank(W), so span{R₁} ≠ W
  have h_span_dim : finrank ℝ ↥(ℝ ∙ R₁) = 1 := finrank_span_singleton hR₁_ne
  have h_span_le : ℝ ∙ R₁ ≤ W :=
    span_le.mpr (Set.singleton_subset_iff.mpr hR₁_mem)
  have h_ne : ℝ ∙ R₁ ≠ W := by
    intro h; rw [h] at h_span_dim; omega
  -- Since span{R₁} ⊊ W, there exists w ∈ W \ span{R₁}
  obtain ⟨w, hw_mem, hw_not⟩ : ∃ w ∈ W, w ∉ ℝ ∙ R₁ := by
    by_contra h; push_neg at h
    exact h_ne (le_antisymm h_span_le (fun x hx => h x hx))
  refine ⟨w, hw_mem, ?_⟩
  rw [mem_span_singleton] at hw_not
  exact fun ⟨c, hc⟩ => hw_not ⟨c, hc.symm⟩

/-!
## Summary

The dimension characterization of Theorem 3 (Skalse et al. 2022):

  ∃ non-trivial simplification ⟺ finrank(Z⊥) ≥ 2
                               ⟺ finrank(Z) ≤ d - 2

is established by:
1. eq_preserving_dim: finrank(Z) + finrank(Z⊥) = d
2. proportional_of_finrank_le_one: finrank(Z⊥) ≤ 1 → all eq-preserving
   R₂ are proportional to R₁ → no non-trivial simplification
   (combined with skalse_theorem3_only_if from Theorem3.lean)
3. exists_independent_of_finrank_ge_two: finrank(Z⊥) ≥ 2 → ∃ independent
   witness → non-trivial simplification exists
   (combined with skalse_theorem3_if from Theorem3.lean)

Fidelity to Skalse et al. 2022 Theorem 3:
- Dimension identity finrank(Z) + finrank(W) = d: proved ✓
- "Only if" via finrank: proved (proportional_of_finrank_le_one) ✓
- "If" via finrank: proved (exists_independent_of_finrank_ge_two) ✓
- Paper's condition "dim(Z₁ ∪...∪ Z_m) ≤ dim(F(Π̂)) - 2": exact match
  via eq_preserving_dim (finrank(Z) + finrank(Z⊥) = d, so
  finrank(Z) ≤ d - 2 ⟺ finrank(Z⊥) ≥ 2) ✓

References:
- Skalse, J., Howe, N.H.R., Krasheninnikov, D., Krueger, D. (2022).
  "Defining and Characterizing Reward Hacking." NeurIPS 2022.
  arXiv: 2209.13085
-/

end
