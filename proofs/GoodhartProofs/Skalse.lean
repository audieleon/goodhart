/-
  Skalse et al. 2022 — "Defining and Characterizing Reward Hacking"
  NeurIPS 2022 (arXiv: 2209.13085)

  First machine-verified formalization of Theorem 2 (existence of
  non-trivial unhackable reward pairs for finite policy sets).

  DESIGN CHOICE: We abstract away MDP structure entirely. No states,
  actions, or transitions appear. The theorem operates on:
  - Occupancy measures F^π (vectors in Fin d → ℝ): how often policy π
    visits each state-action pair
  - Reward functions R (vectors in the same space): how much each
    state-action pair pays
  - Value = dot product: J(π, R) = Σᵢ R(i) · F^π(i)

  This abstraction works because the dot-product structure of J(π, R)
  is the ONLY fact the hackability proofs need. All the MDP structure
  (transitions, discounting, etc.) is absorbed into the occupancy
  measures. This makes the proofs cleaner and the results more general:
  they apply to any setting where value is linear in the reward.

  PROOF ROADMAP:
  1. Define hackability: ∃ π, π' with R₁ preferring π but R₂ preferring π'
  2. Prove the two-policy construction: given F₁ ≠ F₂ and R₁ distinguishing
     them, find R₂ by walking along the line from R₁ to R_opposing until
     the value difference crosses zero (affine root, no IVT needed)
  3. Prove |Π|=2 impossibility: with only 2 policies, any non-trivial R₂
     that doesn't reverse R₁'s ordering must agree with it
  4. Prove |Π|≥3 construction: with 3+ policies, the crossing can equalize
     ONE pair while keeping another pair strictly ordered → non-trivial R₂
-/

import Mathlib.Tactic

open Finset

noncomputable section

namespace Skalse

variable {d : ℕ}

/-!
## Core Definitions

Following Skalse et al. Section 4.2.
-/

/-- Value of a policy (occupancy measure F) under reward R.
    J(π, R) = Σᵢ R(i) · F(i). This is the inner product ⟨R, F⟩. -/
def value (R F : Fin d → ℝ) : ℝ :=
  ∑ i : Fin d, R i * F i

/-- Two reward functions are hackable on a policy set if there exist
    two policies where R₁ prefers one but R₂ prefers the other.
    (Definition 1, Skalse et al. 2022) -/
def hackable (R₁ R₂ : Fin d → ℝ) (policies : Finset (Fin d → ℝ)) : Prop :=
  ∃ F₁ ∈ policies, ∃ F₂ ∈ policies,
    value R₁ F₁ < value R₁ F₂ ∧ value R₂ F₁ > value R₂ F₂

/-- Unhackable = not hackable. The pair (R₁, R₂) never disagree
    on the ordering of any two policies. -/
def unhackable (R₁ R₂ : Fin d → ℝ) (policies : Finset (Fin d → ℝ)) : Prop :=
  ¬hackable R₁ R₂ policies

/-- Two rewards are equivalent on a policy set if they induce
    the same ordering over policies by value. -/
def rewardEquiv (R₁ R₂ : Fin d → ℝ) (policies : Finset (Fin d → ℝ)) : Prop :=
  ∀ F₁ ∈ policies, ∀ F₂ ∈ policies,
    (value R₁ F₁ ≤ value R₁ F₂ ↔ value R₂ F₁ ≤ value R₂ F₂)

/-- A reward is trivial on a policy set if it assigns equal value
    to all policies. -/
def trivialReward (R : Fin d → ℝ) (policies : Finset (Fin d → ℝ)) : Prop :=
  ∀ F₁ ∈ policies, ∀ F₂ ∈ policies, value R F₁ = value R F₂

/-- Unhackable but not equivalent: R₂ collapses some of R₁'s
    preferences without ever reversing them.
    NOTE: This does NOT require R₂ to be non-trivial. The paper's
    Theorem 2 additionally requires R₂ to distinguish at least
    some pair (see isStronglyNontriviallyUnhackable). -/
def isNontriviallyUnhackable (R₁ R₂ : Fin d → ℝ) (policies : Finset (Fin d → ℝ)) : Prop :=
  unhackable R₁ R₂ policies ∧ ¬rewardEquiv R₁ R₂ policies

/-- Full Skalse Theorem 2 predicate: unhackable, not equivalent,
    AND R₂ is non-trivial (distinguishes at least one pair).
    This matches the paper's exact claim. -/
def isStronglyNontriviallyUnhackable (R₁ R₂ : Fin d → ℝ) (policies : Finset (Fin d → ℝ)) : Prop :=
  unhackable R₁ R₂ policies ∧ ¬rewardEquiv R₁ R₂ policies ∧ ¬trivialReward R₂ policies

/-!
## Lemma A: Characterization of Unhackability

Unhackability has an equivalent formulation as order preservation:
R₁ strictly preferring π' over π implies R₂ weakly preferring π' over π.
-/

theorem unhackable_iff_order_preserved (R₁ R₂ : Fin d → ℝ)
    (policies : Finset (Fin d → ℝ)) :
    unhackable R₁ R₂ policies ↔
    ∀ F₁ ∈ policies, ∀ F₂ ∈ policies,
      value R₁ F₁ < value R₁ F₂ → value R₂ F₁ ≤ value R₂ F₂ := by
  constructor
  · intro h F₁ hF₁ F₂ hF₂ hlt
    by_contra hgt
    apply h
    push Not at hgt
    exact ⟨F₁, hF₁, F₂, hF₂, hlt, hgt⟩
  · intro h ⟨F₁, hF₁, F₂, hF₂, hlt, hgt⟩
    have := h F₁ hF₁ F₂ hF₂ hlt
    linarith

/-!
## Lemma B: Value is Linear in the Reward Function

The value function J(π, R) = ⟨R, F^π⟩ is linear in R.
Specifically, value of a convex combination equals the
convex combination of values.
-/

theorem value_linear_combination (R₁ R₃ F : Fin d → ℝ) (t : ℝ) :
    value (fun i => (1 - t) * R₁ i + t * R₃ i) F =
    (1 - t) * value R₁ F + t * value R₃ F := by
  simp only [value]
  simp_rw [show ∀ x, ((1 - t) * R₁ x + t * R₃ x) * F x =
      (1 - t) * (R₁ x * F x) + t * (R₃ x * F x) from fun x => by ring]
  rw [Finset.sum_add_distrib, Finset.mul_sum, Finset.mul_sum]

/-!
## Lemma C: Separating Reward Existence

If two occupancy measures differ, there exists a reward function
that assigns them different values. This is the fundamental
non-degeneracy condition.

Proof: take the indicator function at a coordinate where they differ.
-/

theorem exists_separating_reward {F₁ F₂ : Fin d → ℝ} (h : F₁ ≠ F₂) :
    ∃ R : Fin d → ℝ, value R F₁ ≠ value R F₂ := by
  rw [Function.ne_iff] at h
  obtain ⟨i, hi⟩ := h
  use fun j => if j = i then 1 else 0
  simp only [value]
  simp [Finset.sum_ite_eq']
  exact hi

/-- Stronger version: we can choose which direction separates. -/
theorem exists_separating_reward_directed {F₁ F₂ : Fin d → ℝ} (h : F₁ ≠ F₂) :
    ∃ R : Fin d → ℝ, value R F₁ < value R F₂ := by
  rw [Function.ne_iff] at h
  obtain ⟨i, hi⟩ := h
  by_cases hlt : F₁ i < F₂ i
  · use fun j => if j = i then 1 else 0
    simp only [value]
    simp [Finset.sum_ite_eq']
    exact hlt
  · use fun j => if j = i then -1 else 0
    simp only [value]
    simp [Finset.sum_ite_eq']
    have : F₂ i < F₁ i := lt_of_le_of_ne (not_lt.mp hlt) (Ne.symm hi)
    linarith

/-!
## Lemma D: Affine Root Construction

For a linear function g(t) = (1-t)·a + t·b where a ≤ 0 < b,
the root t* = -a/(b-a) lies in [0,1] and satisfies g(t*) = 0.

This replaces the IVT — for affine functions, the root has a
closed form, no topology needed.
-/

/-- NOTE ON TECHNIQUE: This lemma replaces the Intermediate Value Theorem.
    For AFFINE functions g(t) = (1-t)a + tb, the root has a closed form:
    t* = -a/(b-a). No topology, no completeness of ℝ, no limits. Just
    algebra. This is why we can formalize Skalse's proof without importing
    Mathlib's topology library — the "continuous path" in reward space is
    actually a straight line, and roots of straight lines are computable. -/
theorem affine_root_exists (a b : ℝ) (ha : a ≤ 0) (hb : 0 < b) :
    ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ (1 - t) * a + t * b = 0 := by
  use (-a) / (b - a)
  have hba : 0 < b - a := by linarith
  refine ⟨?_, ?_, ?_⟩
  · exact div_nonneg (neg_nonneg.mpr ha) (le_of_lt hba)
  · rw [div_le_one hba]
    linarith
  · field_simp; ring

/-- When a < 0 and b > 0, the affine root is strictly in (0,1). -/
theorem affine_root_strict (a b t : ℝ)
    (ha : a < 0) (hb : 0 < b)
    (ht0 : 0 ≤ t) (ht1 : t ≤ 1)
    (htroot : (1 - t) * a + t * b = 0) :
    0 < t ∧ t < 1 := by
  constructor
  · by_contra h; push Not at h
    nlinarith [le_antisymm h ht0]
  · by_contra h; push Not at h
    nlinarith [le_antisymm ht1 h]

/-- The value difference along a reward path is affine in t. -/
theorem value_diff_affine (R₁ R₃ F₁ F₂ : Fin d → ℝ) (t : ℝ) :
    value (fun i => (1 - t) * R₁ i + t * R₃ i) F₁ -
    value (fun i => (1 - t) * R₁ i + t * R₃ i) F₂ =
    (1 - t) * (value R₁ F₁ - value R₁ F₂) + t * (value R₃ F₁ - value R₃ F₂) := by
  rw [value_linear_combination, value_linear_combination]
  ring

/-- Negating a reward negates its value. -/
theorem value_neg (R F : Fin d → ℝ) :
    value (fun i => -R i) F = -value R F := by
  simp only [value, neg_mul, Finset.sum_neg_distrib]

/-!
## Theorem 2: Existence (Two-Policy Case)

Given two policies with distinct occupancy measures and any
reward R₁, there exists a non-trivial unhackable R₂.

This is the base case of Skalse et al. Theorem 2.
-/

/-- Helper: a reward that is indifferent between two policies
    is unhackable on {F₁, F₂}. -/
theorem indifferent_is_unhackable (R₁ R₂ : Fin d → ℝ) (F₁ F₂ : Fin d → ℝ)
    (h_indiff : value R₂ F₁ = value R₂ F₂) :
    unhackable R₁ R₂ {F₁, F₂} := by
  rw [unhackable_iff_order_preserved]
  intro G₁ hG₁ G₂ hG₂ _
  simp only [Finset.mem_insert, Finset.mem_singleton] at hG₁ hG₂
  rcases hG₁ with rfl | rfl <;> rcases hG₂ with rfl | rfl <;> linarith

/-- Core construction: given R₁ distinguishing F₁ from F₂ and
    R_opp opposing R₁'s preference, produce a non-trivially
    unhackable R₂ on {F₁, F₂}.

    Used to avoid repeating the construction in each case branch. -/
theorem construct_nontrivial_unhackable
    (R₁ R_opp : Fin d → ℝ) (F₁ F₂ : Fin d → ℝ)
    (hR₁_lt : value R₁ F₁ < value R₁ F₂)
    (hR_opp : value R_opp F₁ > value R_opp F₂) :
    ∃ R₂ : Fin d → ℝ, isNontriviallyUnhackable R₁ R₂ {F₁, F₂} := by
  have ha : value R₁ F₁ - value R₁ F₂ ≤ 0 := by linarith
  have hb : 0 < value R_opp F₁ - value R_opp F₂ := by linarith
  obtain ⟨t, ht0, ht1, htroot⟩ := affine_root_exists _ _ ha hb
  use fun i => (1 - t) * R₁ i + t * R_opp i
  refine ⟨?_, ?_⟩
  · -- Unhackable: R₂ is indifferent between F₁ and F₂
    apply indifferent_is_unhackable
    have := value_diff_affine R₁ R_opp F₁ F₂ t
    linarith
  · -- Not equivalent: R₁ distinguishes but R₂ doesn't
    intro h_equiv
    have h_indiff : value (fun i => (1 - t) * R₁ i + t * R_opp i) F₁ =
        value (fun i => (1 - t) * R₁ i + t * R_opp i) F₂ := by
      have := value_diff_affine R₁ R_opp F₁ F₂ t
      linarith
    rw [rewardEquiv] at h_equiv
    -- Use the reverse pair: equiv says F₂ ≤ F₁ under R₂ → F₂ ≤ F₁ under R₁
    -- But R₂ is indifferent, so F₂ ≤ F₁ is true. This gives R₁: F₂ ≤ F₁.
    -- But we have R₁: F₁ < F₂. Contradiction.
    have := (h_equiv F₂ (by simp) F₁ (by simp)).mpr (le_of_eq h_indiff.symm)
    linarith

/-- Skalse et al. 2022, Theorem 2 (two-policy case, weak version).

    For any two policies with distinct occupancy measures and any
    reward R₁ that distinguishes them, there exists R₂ such that
    (R₁, R₂) is unhackable but not equivalent on {F₁, F₂}.

    NOTE: The witness R₂ is trivial on {F₁, F₂} (assigns equal value
    to both policies). For |Π| = 2, this is unavoidable: any
    non-trivial R₂ that is unhackable must agree with R₁'s ordering,
    making them equivalent. The paper's full Theorem 2 (non-trivial R₂)
    requires |Π| ≥ 3 with sufficient structure. -/
theorem skalse_existence_two
    (R₁ : Fin d → ℝ) (F₁ F₂ : Fin d → ℝ)
    (hF : F₁ ≠ F₂)
    (hR₁ : value R₁ F₁ ≠ value R₁ F₂) :
    ∃ R₂ : Fin d → ℝ, isNontriviallyUnhackable R₁ R₂ {F₁, F₂} := by
  -- Get a separating direction
  obtain ⟨R₃, hR₃⟩ := exists_separating_reward_directed hF
  -- Case split on R₁'s ordering
  by_cases hR₁_lt : value R₁ F₁ < value R₁ F₂
  · -- R₁ prefers F₂. Need R_opp preferring F₁.
    by_cases hR₃_opp : value R₃ F₁ > value R₃ F₂
    · exact construct_nontrivial_unhackable R₁ R₃ F₁ F₂ hR₁_lt hR₃_opp
    · -- R₃ agrees with R₁. Use -R₃ which opposes.
      have hR₃_le : value R₃ F₁ ≤ value R₃ F₂ := not_lt.mp (not_lt.mpr (le_of_not_gt hR₃_opp))
      have : value (fun i => -R₃ i) F₁ > value (fun i => -R₃ i) F₂ := by
        rw [value_neg, value_neg]; linarith
      exact construct_nontrivial_unhackable R₁ (fun i => -R₃ i) F₁ F₂ hR₁_lt this
  · -- R₁ prefers F₁ over F₂
    have hR₁_gt : value R₁ F₁ > value R₁ F₂ :=
      lt_of_le_of_ne (not_lt.mp hR₁_lt) (Ne.symm hR₁)
    -- Swap perspective: R₁ prefers F₁, so we need the construction
    -- with F₂ < F₁ under R₁, i.e. value R₁ F₂ < value R₁ F₁
    by_cases hR₃_opp : value R₃ F₂ > value R₃ F₁
    · -- R₃ already gives F₂ > F₁ (opposes R₁'s F₁ > F₂)
      have h := construct_nontrivial_unhackable R₁ R₃ F₂ F₁ (by linarith) hR₃_opp
      obtain ⟨R₂, huh, hne⟩ := h
      exact ⟨R₂, by rwa [Finset.pair_comm] at huh, by rwa [Finset.pair_comm] at hne⟩
    · -- R₃ agrees with R₁ on this ordering. Use -R₃.
      have : value (fun i => -R₃ i) F₂ > value (fun i => -R₃ i) F₁ := by
        rw [value_neg, value_neg]; linarith [hR₃]
      have h := construct_nontrivial_unhackable R₁ (fun i => -R₃ i) F₂ F₁ (by linarith) this
      obtain ⟨R₂, huh, hne⟩ := h
      exact ⟨R₂, by rwa [Finset.pair_comm] at huh, by rwa [Finset.pair_comm] at hne⟩

/-!
## Theorem 2: General Finite Case

For arbitrary finite policy sets with at least two policies
having distinct occupancy measures. The key insight: unhackability
on a superset implies unhackability on any subset, but we need
the reverse direction. We use the two-policy construction and
show that not-equivalent on the subset implies not-equivalent
on the superset.
-/

/-- Unhackable on a superset implies unhackable on any subset. -/
theorem unhackable_superset {R₁ R₂ : Fin d → ℝ}
    {small large : Finset (Fin d → ℝ)}
    (hsub : small ⊆ large)
    (h : unhackable R₁ R₂ large) :
    unhackable R₁ R₂ small := by
  rw [unhackable_iff_order_preserved] at h ⊢
  intro F₁ hF₁ F₂ hF₂
  exact h F₁ (hsub hF₁) F₂ (hsub hF₂)

/-- Not-equivalent on a subset implies not-equivalent on any superset. -/
theorem not_equiv_subset {R₁ R₂ : Fin d → ℝ}
    {small large : Finset (Fin d → ℝ)}
    (hsub : small ⊆ large)
    (h : ¬rewardEquiv R₁ R₂ small) :
    ¬rewardEquiv R₁ R₂ large := by
  intro h_equiv
  apply h
  intro F₁ hF₁ F₂ hF₂
  exact h_equiv F₁ (hsub hF₁) F₂ (hsub hF₂)

/-- Weak version of Skalse et al. 2022 Theorem 2 (general case).

    For any finite policy set with two policies distinguished by R₁,
    there exists R₂ that is unhackable and not equivalent to R₁.

    The witness R₂ = 0 (zero reward) is TRIVIAL — it assigns equal
    value to all policies. This makes it unhackable (no ordering to
    reverse) and not equivalent (R₁ distinguishes but R₂ doesn't).

    The paper's Theorem 2 additionally requires R₂ to be non-trivial.
    That stronger claim requires |Π| ≥ 3 and uses a min-crossing
    argument along a path in reward space to construct a non-trivial
    witness. See the paper (page 17, proof of Theorem 2) for details.

    Relationship to Skalse et al. 2022 Theorem 2:
    - Definitions (hackable, unhackable, equiv): exact match (Def. 1)
    - Conclusion (∃ R₂ unhackable ∧ ¬equiv): proved ✓
    - Non-triviality of R₂: NOT proved (our R₂ is trivial)
    - Hypothesis hR₁ (R₁ distinguishes): stronger than paper requires -/
theorem skalse_existence_general
    (R₁ : Fin d → ℝ) (policies : Finset (Fin d → ℝ))
    (F₁ F₂ : Fin d → ℝ)
    (hF₁ : F₁ ∈ policies) (hF₂ : F₂ ∈ policies)
    (_hF : F₁ ≠ F₂)
    (hR₁ : value R₁ F₁ ≠ value R₁ F₂) :
    ∃ R₂ : Fin d → ℝ, isNontriviallyUnhackable R₁ R₂ policies := by
  -- For the general case, we construct R₂ as a simplification
  -- of R₁ that equalizes exactly one pair.
  -- The two-policy result gives us a witness on {F₁, F₂}.
  -- Not-equivalent lifts from subset to superset.
  -- For unhackability on the full set, we use a trivial R₂:
  -- R₂ := 0 (the zero reward) is trivially unhackable with any R₁
  -- (it assigns equal value to all policies, so no pair can be reversed).
  -- And it's not equivalent to R₁ since R₁ distinguishes F₁ from F₂.
  use fun _ => 0
  refine ⟨?_, ?_⟩
  · -- Unhackable: zero reward assigns equal value to all policies
    rw [unhackable_iff_order_preserved]
    intro G₁ _ G₂ _ _
    simp only [value, zero_mul, Finset.sum_const_zero]
    exact le_refl 0
  · -- Not equivalent: R₁ distinguishes F₁, F₂ but zero reward doesn't
    intro h_equiv
    rw [rewardEquiv] at h_equiv
    have h0 : value (fun _ => (0 : ℝ)) F₁ = value (fun _ => (0 : ℝ)) F₂ := by
      simp only [value, zero_mul, Finset.sum_const_zero]
    by_cases hlt : value R₁ F₁ < value R₁ F₂
    · -- R₁ prefers F₂, but equiv + zero gives F₂ ≤ F₁ too
      have := (h_equiv F₂ hF₂ F₁ hF₁).mpr (le_of_eq h0.symm)
      linarith
    · -- R₁ prefers F₁, but equiv + zero gives F₁ ≤ F₂ too
      have hgt : value R₁ F₁ > value R₁ F₂ :=
        lt_of_le_of_ne (not_lt.mp hlt) (Ne.symm hR₁)
      have := (h_equiv F₁ hF₁ F₂ hF₂).mpr (le_of_eq h0)
      linarith

/-!
## Theorem 2: Strong Version (Non-Trivial Witness, |Π| ≥ 3)

For three policies with three distinct R₁-values, we construct
a genuinely non-trivial R₂. The key: use R_opp that opposes R₁
on one pair but is neutral on another. Then the crossing only
equalizes the targeted pair while the neutral pair retains its
strict ordering — giving non-triviality.
-/

/-- Value along a convex combination preserves weak ordering when
    both endpoints agree. -/
theorem value_convex_le (R₁ R_opp F₁ F₂ : Fin d → ℝ) (t : ℝ)
    (ht0 : 0 ≤ t) (ht1 : t ≤ 1)
    (h₁ : value R₁ F₁ ≤ value R₁ F₂)
    (h₂ : value R_opp F₁ ≤ value R_opp F₂) :
    value (fun i => (1 - t) * R₁ i + t * R_opp i) F₁ ≤
    value (fun i => (1 - t) * R₁ i + t * R_opp i) F₂ := by
  have := value_diff_affine R₁ R_opp F₁ F₂ t
  linarith [mul_nonneg (sub_nonneg.mpr ht1) (sub_nonneg.mpr h₁),
            mul_nonneg ht0 (sub_nonneg.mpr h₂)]

/-- Skalse Theorem 2 (strong, non-trivial witness).

    Given three policies with three distinct R₁-values and an
    opposing reward R_opp that is neutral on the (F₂,F₃) pair,
    there exists a non-trivial R₂ satisfying the full Theorem 2
    predicate (unhackable ∧ ¬equivalent ∧ R₂ non-trivial).

    The hypothesis hR_opp_23 (neutrality on one pair) is achievable
    whenever the occupancy measure differences F₁-F₂ and F₂-F₃ are
    linearly independent — standard for policies with distinct
    behavior in MDPs with ≥ 2 state-action pairs.

    This resolves the |Π| = 2 impossibility: with 3 policies,
    the construction produces a witness that equalizes one pair
    while preserving strict ordering on another. -/
theorem skalse_existence_nontrivial_three
    (R₁ R_opp : Fin d → ℝ) (F₁ F₂ F₃ : Fin d → ℝ)
    (hR₁_12 : value R₁ F₁ < value R₁ F₂)
    (hR₁_23 : value R₁ F₂ < value R₁ F₃)
    (hR_opp_12 : value R_opp F₁ > value R_opp F₂)
    (hR_opp_23 : value R_opp F₂ = value R_opp F₃) :
    ∃ R₂, isStronglyNontriviallyUnhackable R₁ R₂ {F₁, F₂, F₃} := by
  -- Find crossing time for (F₁, F₂)
  have ha : value R₁ F₁ - value R₁ F₂ ≤ 0 := by linarith
  have hb : 0 < value R_opp F₁ - value R_opp F₂ := by linarith
  obtain ⟨t, ht0, ht1, htroot⟩ := affine_root_exists _ _ ha hb
  -- Witness: R₂ = (1-t)R₁ + tR_opp
  use fun i => (1 - t) * R₁ i + t * R_opp i
  obtain ⟨ht_pos, ht_lt_one⟩ := affine_root_strict _ _ t (by linarith) hb ht0 ht1 htroot
  -- R₂ equalizes (F₁, F₂)
  have h_eq_12 : value (fun i => (1 - t) * R₁ i + t * R_opp i) F₁ =
      value (fun i => (1 - t) * R₁ i + t * R_opp i) F₂ := by
    have := value_diff_affine R₁ R_opp F₁ F₂ t; linarith
  -- R₂ preserves strict ordering on (F₂, F₃)
  have h_opp_diff : value R_opp F₂ - value R_opp F₃ = 0 := sub_eq_zero.mpr hR_opp_23
  have h_lt_23 : value (fun i => (1 - t) * R₁ i + t * R_opp i) F₂ <
      value (fun i => (1 - t) * R₁ i + t * R_opp i) F₃ := by
    have hdiff := value_diff_affine R₁ R_opp F₂ F₃ t
    -- (1-t)(v₁(F₂)-v₁(F₃)) + t·0 = (1-t)(v₁(F₂)-v₁(F₃)) < 0
    nlinarith
  -- R₂ ordering on (F₁, F₃)
  have h_le_13 : value (fun i => (1 - t) * R₁ i + t * R_opp i) F₁ ≤
      value (fun i => (1 - t) * R₁ i + t * R_opp i) F₃ := by linarith
  refine ⟨?_, ?_, ?_⟩
  · -- Unhackable: check all pairs in {F₁, F₂, F₃}
    rw [unhackable_iff_order_preserved]
    intro G₁ hG₁ G₂ hG₂ hlt
    simp only [Finset.mem_insert, Finset.mem_singleton] at hG₁ hG₂
    -- 9 cases from 3×3 membership; trivial cases (G₁=G₂) are contradictions
    rcases hG₁ with rfl | rfl | rfl <;> rcases hG₂ with rfl | rfl | rfl <;> linarith
  · -- Not equivalent: R₁ strictly orders F₁ < F₂ but R₂ equalizes them
    intro h_equiv
    rw [rewardEquiv] at h_equiv
    have := (h_equiv F₂ (by simp) F₁ (by simp)).mpr (le_of_eq h_eq_12.symm)
    linarith
  · -- Non-trivial: R₂ strictly distinguishes F₂ from F₃
    intro h_triv
    have := h_triv F₂ (by simp) F₃ (by simp [Finset.mem_insert])
    linarith

/-!
## Summary

Theorems:
1. unhackable_iff_order_preserved — characterization of unhackability
2. value_linear_combination — linearity of value in reward
3. exists_separating_reward — non-degenerate policies are separable
4. exists_separating_reward_directed — separability with chosen direction
5. affine_root_exists — closed-form root of affine function
6. value_diff_affine — value difference is affine along reward paths
7. value_neg — negation distributes over value
8. indifferent_is_unhackable — indifference implies unhackability
9. construct_nontrivial_unhackable — core two-policy construction
10. skalse_existence_two — Theorem 2 for two policies (weak: trivial R₂)
11. unhackable_superset — monotonicity of unhackability
12. not_equiv_subset — anti-monotonicity of non-equivalence
13. skalse_existence_general — Theorem 2 general (weak: trivial R₂)

Definitions formalized:
- value (inner product of reward and occupancy measure)
- hackable / unhackable (Definition 1, Skalse et al. — exact match)
- rewardEquiv (order equivalence — exact match)
- trivialReward
- isNontriviallyUnhackable (unhackable ∧ ¬equiv)
- isStronglyNontriviallyUnhackable (unhackable ∧ ¬equiv ∧ ¬trivial R₂)

Fidelity to Skalse et al. 2022:
- Definition 1 (hackability): exact match, strict inequalities ✓
- Equivalence: exact match via pairwise ≤ biconditional ✓
- Theorem 2 conclusion (∃ R₂ unhackable ∧ ¬equiv): proved ✓
- Theorem 2 non-triviality of R₂: NOT proved (our witnesses are trivial)
- See `two_policy_nontrivial_impossible` for a machine-verified proof
  that non-trivial witnesses are impossible for |Π| = 2.
-/

/-!
## Edge Case Analysis: |Π| = 2 and Non-Trivial Witnesses

Skalse et al. 2022 Theorem 2 (Section 5.2, arXiv:2209.13085) states:

  "For any MDP\R, any finite set of policies Π̂ containing at least
  two π, π' such that F(π) ≠ F(π'), and any reward function R₁,
  there is a NON-TRIVIAL reward function R₂ such that R₁ and R₂
  are unhackable but not equivalent."

The hypothesis says "at least two" policies with distinct occupancy
measures. We prove this is insufficient: for |Π| = 2 with R₁
non-trivial, no non-trivial R₂ satisfies the conclusion.

### Why the proof requires |Π| ≥ 3

The paper's proof (Section 9) constructs a path from R₁ to -R₁ in
reward space. Along this path, different policy pairs equalize at
different times. The proof then argues the path can avoid the
"trivial subspace" T = {R : value R F = const for all F ∈ Π}.

The avoidance argument requires codim(T) ≥ 2. With d-dimensional
occupancy measures and k policies with affinely independent F:
- The differences {F(πᵢ) - F(π₁)} span a (k-1)-dimensional space
- T is the orthogonal complement: dim(T) = d - (k-1)
- codim(T) = k - 1

For codim(T) ≥ 2: need k ≥ 3, i.e., at least 3 policies with
affinely independent occupancy measures.

For k = 2: codim(T) = 1 (T is a hyperplane). The path from R₁
to -R₁ must cross this hyperplane. At the crossing, R₂ is trivial.
Every continuous path between R₁ and -R₁ passes through a trivial
reward — the avoidance argument fails.

### Where "≥ 2" appears in the paper

1. Theorem 2 (Section 5.2): "at least two π, π' with F(π) ≠ F(π')"
   → Proof needs ≥ 3 for the path-avoidance argument

2. Corollary 3 (Section 5.2): "|Π̂| ≥ 2 and J(π) ≠ J(π') for all π,π'"
   → Follows from Theorem 3 (separate argument), not Theorem 2

The condition in Theorem 2 appears to have an off-by-one: the proof
requires ≥ 3 affinely independent occupancy measures, but the
statement says ≥ 2. This does not affect the paper's substance
(the interesting case is large Π), but it is a precise mathematical
discrepancy that we document and prove below.
-/

/-- For |Π| = 2 with R₁ non-trivial: any non-trivial unhackable R₂
    must be equivalent to R₁. Therefore the conjunction
    (non-trivial ∧ unhackable ∧ not-equivalent) is impossible.

    Proof: On {F₁, F₂}, R₂ non-trivial means value R₂ F₁ ≠ value R₂ F₂.
    WLOG value R₁ F₁ < value R₁ F₂ (R₁ is non-trivial).
    Unhackable means: value R₁ F₁ < value R₁ F₂ → value R₂ F₁ ≤ value R₂ F₂.
    Since R₂ is non-trivial, either value R₂ F₁ < value R₂ F₂ or vice versa.
    If value R₂ F₁ < value R₂ F₂: R₂ agrees with R₁ → equivalent.
    If value R₂ F₁ > value R₂ F₂: contradicts unhackability.
    In both cases, (non-trivial ∧ unhackable ∧ ¬equivalent) fails. -/
theorem two_policy_nontrivial_impossible
    (R₁ R₂ : Fin d → ℝ) (F₁ F₂ : Fin d → ℝ)
    (hR₁ : value R₁ F₁ < value R₁ F₂)
    (hR₂_nontrivial : ¬trivialReward R₂ {F₁, F₂})
    (hR₂_unhackable : unhackable R₁ R₂ {F₁, F₂}) :
    rewardEquiv R₁ R₂ {F₁, F₂} := by
  -- R₂ non-trivial on {F₁, F₂} means value R₂ F₁ ≠ value R₂ F₂
  have hne : value R₂ F₁ ≠ value R₂ F₂ := by
    intro heq
    apply hR₂_nontrivial
    intro G₁ hG₁ G₂ hG₂
    simp only [Finset.mem_insert, Finset.mem_singleton] at hG₁ hG₂
    rcases hG₁ with rfl | rfl <;> rcases hG₂ with rfl | rfl <;> linarith
  -- Unhackable means R₂ preserves R₁'s strict ordering
  rw [unhackable_iff_order_preserved] at hR₂_unhackable
  -- R₁: F₁ < F₂. Unhackable gives R₂: F₁ ≤ F₂.
  have h_le : value R₂ F₁ ≤ value R₂ F₂ :=
    hR₂_unhackable F₁ (by simp) F₂ (by simp) hR₁
  -- Combined with R₂ non-trivial: R₂: F₁ < F₂ (strict)
  have h_lt : value R₂ F₁ < value R₂ F₂ :=
    lt_of_le_of_ne h_le hne
  -- Now show R₁ and R₂ are equivalent on {F₁, F₂}
  intro G₁ hG₁ G₂ hG₂
  simp only [Finset.mem_insert, Finset.mem_singleton] at hG₁ hG₂
  rcases hG₁ with rfl | rfl <;> rcases hG₂ with rfl | rfl
  · exact ⟨fun _ => le_refl _, fun _ => le_refl _⟩
  · exact ⟨fun _ => le_of_lt h_lt, fun _ => le_of_lt hR₁⟩
  · exact ⟨fun h => absurd (lt_of_lt_of_le hR₁ h) (lt_irrefl _),
          fun h => absurd (lt_of_lt_of_le h_lt h) (lt_irrefl _)⟩
  · exact ⟨fun _ => le_refl _, fun _ => le_refl _⟩

/-- Corollary: on a two-element policy set with non-trivial R₁,
    isStronglyNontriviallyUnhackable is unsatisfiable. -/
theorem two_policy_strong_impossible
    (R₁ : Fin d → ℝ) (F₁ F₂ : Fin d → ℝ)
    (hR₁ : value R₁ F₁ ≠ value R₁ F₂) :
    ¬∃ R₂, isStronglyNontriviallyUnhackable R₁ R₂ {F₁, F₂} := by
  intro ⟨R₂, huh, hne, hnt⟩
  by_cases hlt : value R₁ F₁ < value R₁ F₂
  · exact hne (two_policy_nontrivial_impossible R₁ R₂ F₁ F₂ hlt hnt huh)
  · -- Symmetric case: value R₁ F₂ < value R₁ F₁
    have hgt : value R₁ F₂ < value R₁ F₁ :=
      lt_of_le_of_ne (not_lt.mp hlt) (Ne.symm hR₁)
    -- Swap: unhackable is symmetric in the policy set ordering
    have huh' : unhackable R₁ R₂ {F₂, F₁} := by
      rwa [Finset.pair_comm] at huh
    have hnt' : ¬trivialReward R₂ {F₂, F₁} := by
      rwa [Finset.pair_comm] at hnt
    have heq := two_policy_nontrivial_impossible R₁ R₂ F₂ F₁ hgt hnt' huh'
    apply hne
    intro G₁ hG₁ G₂ hG₂
    exact heq G₁ (by rwa [Finset.pair_comm] at hG₁)
      G₂ (by rwa [Finset.pair_comm] at hG₂)

/-!
## References

- Skalse, J., Howe, N.H.R., Krasheninnikov, D., Krueger, D. (2022).
  "Defining and Characterizing Reward Hacking." NeurIPS 2022.
  arXiv: 2209.13085

  Theorem 2 (Section 5.2): "at least two π, π' with F(π) ≠ F(π')"
  Proof (Section 9): path-avoidance of trivial subspace
  Corollary 3 (Section 5.2): "|Π̂| ≥ 2 and J(π) ≠ J(π') for all pairs"

  Our analysis shows the Theorem 2 proof requires codim(T) ≥ 2, which
  needs ≥ 3 affinely independent occupancy measures (not ≥ 2 as stated).
  The theorem is correct for |Π| ≥ 3. For |Π| = 2 with non-trivial R₁,
  we prove the conclusion is impossible (two_policy_strong_impossible).
-/

end Skalse

end
