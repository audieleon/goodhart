/-
  Bellman Equations — Operators, Contraction, and Fixed Points

  Defines the Bellman optimality operator for finite MDPs and proves
  it is a contraction mapping on the sup-norm. The unique fixed point
  is the optimal value function V*.

  Key results:
  - bellmanOptOp is γ-Lipschitz (contraction for γ < 1)
  - V* exists and is unique (Banach fixed point)
  - Q* derived from V*

  PROOF ROADMAP (plain language):
  1. Define Q-value: Q(s,a,V) = Σ T(s,a,s') · [R(s,a,s') + γ·V(s')]
  2. Define Bellman operator: (T*V)(s) = max_a Q(s,a,V)
  3. Prove Q-value contraction: changing V by at most C changes Q by at most γC
     (because the γ multiplier on V shrinks the difference, and the T weights
     sum to 1 so they can't amplify it)
  4. Prove Bellman operator contraction: max preserves the Lipschitz bound
  5. Apply Banach's fixed point theorem: γ < 1 means T* is a contraction on a
     complete metric space, so it has a unique fixed point. That fixed point is V*.

  The heavy mathematical lifting is in step 3 (triangle inequality + PMF sums to 1)
  and step 5 (Banach's theorem from Mathlib, invoked in one line).
-/

import GoodhartProofs.MDP.Defs

open Finset ENNReal

noncomputable section

namespace MDP

variable {M : FiniteMDP}

/-!
## PMF Bridge Lemma

PMF values live in ℝ≥0∞ (extended non-negative reals), but we need ℝ for
arithmetic. These lemmas bridge the gap. The critical fact: for a PMF over
a finite type, the real-valued sum Σ toReal(p(s)) = 1. This is used
repeatedly in the contraction proof (steps weighted by transition probs sum to 1).
-/

/-- Each PMF value is finite (not ⊤).
    PROOF: each p(s) ≤ Σ p(s) = 1 < ⊤. -/
theorem pmf_apply_ne_top (p : PMF M.S) (s : M.S) : p s ≠ ⊤ :=
  ne_top_of_le_ne_top one_ne_top
    (PMF.tsum_coe p ▸ ENNReal.le_tsum s)

/-- PMF values converted to ℝ are non-negative.
    Follows from ENNReal.toReal being non-negative by definition. -/
theorem pmf_toReal_nonneg (p : PMF M.S) (s : M.S) : 0 ≤ (p s).toReal :=
  ENNReal.toReal_nonneg

/-- The sum of PMF toReal values over a Fintype is 1.
    This is the CRITICAL bridge: it lets us treat transition probabilities
    as real-valued weights that sum to 1. Without this, the contraction
    proof can't conclude that weighted averages don't amplify differences.

    PROOF: convert the finite sum to a tsum (they agree on Fintype), then
    use PMF.tsum_coe which says Σ p(s) = 1 in ℝ≥0∞, then convert to ℝ. -/
theorem pmf_sum_toReal_eq_one (p : PMF M.S) :
    ∑ s : M.S, (p s).toReal = 1 := by
  rw [← ENNReal.toReal_sum (fun s _ => pmf_apply_ne_top p s)]
  have h : ∑ s ∈ Finset.univ, p s = 1 := by
    rw [← PMF.tsum_coe p]
    exact (tsum_eq_sum (fun s hs => absurd (Finset.mem_univ s) hs)).symm
  rw [h]
  rfl

/-!
## Bellman Optimality Operator

The operator that encodes "pick the best action at each state":
  (T* V)(s) = max_a Σ_{s'} T(s,a)(s') · [R(s,a,s') + γ·V(s')]

V is a candidate value function. T* maps value functions to value functions.
When T*V = V (fixed point), V is the optimal value function V*.
-/

/-- The Q-value for a specific state-action pair given a value function.
    Q(s,a,V) = E_{s'~T(s,a)}[R(s,a,s') + γ·V(s')]
    This is a finite weighted sum (no integrals needed for finite MDPs). -/
def qValue (M : FiniteMDP) (V : M.S → ℝ) (s : M.S) (a : M.A) : ℝ :=
  ∑ s' : M.S, (M.T s a s').toReal * (M.R s a s' + M.γ * V s')

/-- The Bellman optimality operator.
    (T* V)(s) = max_a Q(s, a, V).
    Uses Finset.sup' (supremum over a nonempty finite set) because the
    action space is finite and inhabited. -/
def bellmanOptOp (M : FiniteMDP) (V : M.S → ℝ) (s : M.S) : ℝ :=
  Finset.sup' Finset.univ ⟨default, Finset.mem_univ _⟩ (qValue M V s)

/-!
## Contraction Property

THE KEY THEOREM: changing V by at most C changes T*V by at most γC.

Why does this work? Expanding Q(s,a,V₁) - Q(s,a,V₂):
  = Σ T(s,a,s') · [R + γV₁(s')] - Σ T(s,a,s') · [R + γV₂(s')]
  = Σ T(s,a,s') · γ · (V₁(s') - V₂(s'))       ← R cancels!
  ≤ Σ T(s,a,s') · γ · C                         ← bound on |V₁-V₂|
  = γC · Σ T(s,a,s')                             ← factor out γC
  = γC · 1                                        ← probabilities sum to 1
  = γC                                             ← done!

The R cancels because it doesn't depend on V. The γ < 1 shrinks
the difference. The transition weights summing to 1 prevents amplification.
This is why the Bellman operator contracts: it takes a weighted average
(can't amplify) after multiplying by γ (shrinks).
-/

/-- The Q-value difference is bounded by γ times the value function difference.
    This is the core contraction step. See the algebraic expansion above. -/
theorem qValue_sub_le (V₁ V₂ : M.S → ℝ) (s : M.S) (a : M.A)
    (hbound : ∀ s', |V₁ s' - V₂ s'| ≤ C) :
    |qValue M V₁ s a - qValue M V₂ s a| ≤ M.γ * C := by
  simp only [qValue]
  -- Step 1: The R terms cancel, leaving γ·(V₁-V₂) weighted by T
  have : (∑ s' : M.S, (M.T s a s').toReal * (M.R s a s' + M.γ * V₁ s')) -
         (∑ s' : M.S, (M.T s a s').toReal * (M.R s a s' + M.γ * V₂ s')) =
         ∑ s' : M.S, (M.T s a s').toReal * (M.γ * (V₁ s' - V₂ s')) := by
    rw [← Finset.sum_sub_distrib]
    congr 1; ext s'; ring
  rw [this]
  -- Step 2: Triangle inequality (|Σ| ≤ Σ|·|), then factor |T·γ·diff|
  calc |∑ s', (M.T s a s').toReal * (M.γ * (V₁ s' - V₂ s'))|
      ≤ ∑ s', |(M.T s a s').toReal * (M.γ * (V₁ s' - V₂ s'))| :=
        Finset.abs_sum_le_sum_abs _ _
    -- Step 3: T values are non-negative, so |T·x| = T·|x|
    _ = ∑ s', (M.T s a s').toReal * |M.γ * (V₁ s' - V₂ s')| := by
        congr 1; ext s'
        rw [abs_mul, abs_of_nonneg (pmf_toReal_nonneg (M.T s a) s')]
    -- Step 4: γ is positive, so |γ·x| = γ·|x|
    _ = ∑ s', (M.T s a s').toReal * (M.γ * |V₁ s' - V₂ s'|) := by
        congr 1; ext s'
        rw [abs_mul, abs_of_nonneg (le_of_lt M.hγ_pos)]
    -- Step 5: Bound |V₁(s') - V₂(s')| ≤ C (our hypothesis)
    _ ≤ ∑ s', (M.T s a s').toReal * (M.γ * C) := by
        apply Finset.sum_le_sum; intro s' _
        apply mul_le_mul_of_nonneg_left _ (pmf_toReal_nonneg (M.T s a) s')
        apply mul_le_mul_of_nonneg_left (hbound s') (le_of_lt M.hγ_pos)
    -- Step 6: Factor γC out of the sum, sum of T values = 1
    _ = M.γ * C * ∑ s', (M.T s a s').toReal := by
        rw [Finset.mul_sum]; congr 1; ext s'; ring
    _ = M.γ * C := by rw [pmf_sum_toReal_eq_one (M.T s a)]; ring

/-- The Bellman operator difference is bounded by γC.
    Since T* takes the max over Q-values, and each Q-value differs
    by at most γC, the max differs by at most γC.

    PROOF STRATEGY: show |sup f - sup g| ≤ γC by proving both
    sup f ≤ sup g + γC and sup g ≤ sup f + γC. Each direction
    works because every f(a) is within γC of some g(a) (namely, the same a). -/
theorem bellmanOptOp_sub_le (V₁ V₂ : M.S → ℝ) (s : M.S)
    (hbound : ∀ s', |V₁ s' - V₂ s'| ≤ C) :
    |bellmanOptOp M V₁ s - bellmanOptOp M V₂ s| ≤ M.γ * C := by
  simp only [bellmanOptOp]
  have hne : (Finset.univ : Finset M.A).Nonempty :=
    ⟨(default : M.A), Finset.mem_univ _⟩
  -- Show both directions: sup q₁ - sup q₂ ∈ [-γC, γC]
  apply abs_le.mpr
  constructor
  · -- Direction 1: -(γC) ≤ sup q₁ - sup q₂
    -- Equivalently: sup q₂ ≤ sup q₁ + γC
    -- For any action a: q₂(a) ≤ q₁(a) + γC ≤ sup q₁ + γC
    have h : Finset.sup' Finset.univ hne (qValue M V₂ s) ≤
        Finset.sup' Finset.univ hne (qValue M V₁ s) + M.γ * C := by
      apply Finset.sup'_le
      intro a ha
      have := (abs_le.mp (qValue_sub_le V₁ V₂ s a hbound)).1
      linarith [Finset.le_sup' (qValue M V₁ s) ha]
    linarith
  · -- Direction 2: sup q₁ - sup q₂ ≤ γC (symmetric argument)
    have h : Finset.sup' Finset.univ hne (qValue M V₁ s) ≤
        Finset.sup' Finset.univ hne (qValue M V₂ s) + M.γ * C := by
      apply Finset.sup'_le
      intro a ha
      have := (abs_le.mp (qValue_sub_le V₁ V₂ s a hbound)).2
      linarith [Finset.le_sup' (qValue M V₂ s) ha]
    linarith

/-!
## Bellman Policy Evaluation Operator

For a FIXED deterministic policy π (not optimizing, just evaluating),
the Bellman operator T_π is simpler: no max, just Q(s, π(s), V).
It's also a γ-contraction, so its fixed point V^π exists and is unique.

This is needed for Ng 1999: we need V^π for every policy π to show
that potential-based shaping preserves not just V* but all V^π.
-/

/-- Bellman policy evaluation operator for a deterministic policy.
    (T_π V)(s) = Q(s, π(s), V) — no max, policy is fixed. -/
def bellmanPolicyOp (M : FiniteMDP) (π : DetPolicy M) (V : M.S → ℝ) (s : M.S) : ℝ :=
  qValue M V s (π s)

/-- The policy Bellman operator is γ-Lipschitz.
    Same proof as for Q-value — plugging in π(s) for the action just
    restricts to one action instead of taking the max.

    TECHNICAL NOTE: LipschitzWith uses NNReal for the constant.
    ⟨M.γ, M.hγ_nonneg⟩ packages γ with its non-negativity proof. -/
theorem bellmanPolicy_lipschitz (π : DetPolicy M) :
    LipschitzWith ⟨M.γ, M.hγ_nonneg⟩ (bellmanPolicyOp M π) := by
  rw [lipschitzWith_iff_dist_le_mul]
  intro V₁ V₂
  -- dist in function space (M.S → ℝ) is the sup-norm: sup_s |V₁(s) - V₂(s)|
  rw [dist_pi_le_iff (mul_nonneg (NNReal.coe_nonneg _) dist_nonneg)]
  intro s; rw [Real.dist_eq]
  exact qValue_sub_le V₁ V₂ s (π s) (fun s' => by
    rw [← Real.dist_eq]; exact dist_le_pi_dist V₁ V₂ s')

/-- The policy Bellman operator is a γ-contraction.
    "Contraction" = Lipschitz with constant < 1. Since γ < 1, done. -/
theorem bellmanPolicy_contracting (π : DetPolicy M) :
    ContractingWith ⟨M.γ, M.hγ_nonneg⟩ (bellmanPolicyOp M π) :=
  ⟨by exact_mod_cast M.hγ_lt, bellmanPolicy_lipschitz π⟩

/-!
### V^π: Value Function Under a Fixed Policy

**HERE IS WHERE BANACH'S FIXED POINT THEOREM IS INVOKED.**

ContractingWith.fixedPoint takes a contraction mapping on a complete
metric space and returns its unique fixed point. Mathlib proves this
using the standard iterative construction: start anywhere, apply T_π
repeatedly, the sequence converges geometrically to V^π.

We don't prove convergence ourselves — Banach did it in 1922 and
Mathlib formalized it. We just verify the premises (T_π is a
contraction on a complete metric space) and invoke the conclusion.
-/

/-- V^π: value function under a fixed deterministic policy.
    Defined as the unique fixed point of the γ-contraction T_π.

    MATHEMATICAL CONTENT: this single line invokes the Banach Fixed
    Point Theorem. Mathlib's proof constructs V^π as the limit of
    the sequence V₀ = 0, V_{n+1} = T_π(V_n), which converges because
    T_π shrinks distances by factor γ < 1 at each step. -/
noncomputable def vPi (M : FiniteMDP) (π : DetPolicy M) : M.S → ℝ :=
  ContractingWith.fixedPoint (bellmanPolicyOp M π) (bellmanPolicy_contracting π)

/-- V^π is a fixed point: T_π(V^π) = V^π.
    This is the Bellman equation: V^π(s) = Σ T(s,π(s),s') · [R + γ·V^π(s')]. -/
theorem vPi_is_fixed (M : FiniteMDP) (π : DetPolicy M) :
    bellmanPolicyOp M π (vPi M π) = vPi M π :=
  (bellmanPolicy_contracting (M := M) π).fixedPoint_isFixedPt

/-- The fixed point is unique: if T_π(V) = V, then V = V^π.
    There can't be two different self-consistent value functions for the
    same policy. (This is a standard consequence of contraction mappings.) -/
theorem bellmanPolicy_unique (M : FiniteMDP) (π : DetPolicy M) (V : M.S → ℝ) :
    bellmanPolicyOp M π V = V → V = vPi M π :=
  fun h => (bellmanPolicy_contracting (M := M) π).fixedPoint_unique h

/-!
## Bellman Optimality Operator is a Contraction (→ V* exists)

Same argument as for T_π, but now with the max over actions.
The Lipschitz constant is the same γ because max preserves
the bound (shown in bellmanOptOp_sub_le above).
-/

/-- The Bellman optimality operator is γ-Lipschitz on the sup-norm. -/
theorem bellman_lipschitz :
    LipschitzWith ⟨M.γ, M.hγ_nonneg⟩ (bellmanOptOp M) := by
  rw [lipschitzWith_iff_dist_le_mul]
  intro V₁ V₂
  rw [dist_pi_le_iff (mul_nonneg (NNReal.coe_nonneg _) dist_nonneg)]
  intro s
  rw [Real.dist_eq]
  exact bellmanOptOp_sub_le V₁ V₂ s (fun s' => by
    rw [← Real.dist_eq]; exact dist_le_pi_dist V₁ V₂ s')

/-- The Bellman optimality operator is a γ-contraction (γ < 1). -/
theorem bellman_contracting :
    ContractingWith ⟨M.γ, M.hγ_nonneg⟩ (bellmanOptOp M) :=
  ⟨by exact_mod_cast M.hγ_lt, @bellman_lipschitz M⟩

/-!
## Optimal Value Function V*

**BANACH'S FIXED POINT THEOREM, INVOKED AGAIN.**

Same one-line construction as V^π, but now for the optimality operator T*.
The fixed point V* satisfies the Bellman optimality equation:
  V*(s) = max_a Σ T(s,a,s') · [R(s,a,s') + γ·V*(s')]

This is the fundamental result of MDP theory: the optimal value function
exists, is unique, and can be computed by value iteration.
-/

instance : Nonempty (M.S → ℝ) := ⟨fun _ => 0⟩

/-- The optimal value function V*.
    Defined as the unique fixed point of the Bellman optimality operator
    via the Banach fixed point theorem.

    In practice, V* is computed by value iteration:
    start with V₀ = 0, compute V_{n+1} = T*(V_n), repeat until convergence.
    The convergence rate is geometric with ratio γ. -/
noncomputable def vStar (M : FiniteMDP) : M.S → ℝ :=
  ContractingWith.fixedPoint (bellmanOptOp M) (bellman_contracting (M := M))

/-- V* satisfies the Bellman optimality equation: T*(V*) = V*. -/
theorem vStar_is_fixed (M : FiniteMDP) :
    bellmanOptOp M (vStar M) = vStar M :=
  (bellman_contracting (M := M)).fixedPoint_isFixedPt

/-- V* is the UNIQUE fixed point: if T*(V) = V, then V = V*.
    There is exactly one value function that is self-consistent
    under optimal play. -/
theorem bellman_unique (M : FiniteMDP) (V : M.S → ℝ) :
    bellmanOptOp M V = V → V = vStar M :=
  fun h => (bellman_contracting (M := M)).fixedPoint_unique h

/-!
## Q* and Optimal Policy

With V* in hand, Q* is just Q(s,a,V*). The optimal policy is
the argmax of Q* over actions (not formalized here, but the
existence of V* is what matters for the shaping proofs).
-/

/-- The optimal Q-function, derived from V*. -/
def qStar (M : FiniteMDP) (s : M.S) (a : M.A) : ℝ :=
  qValue M (vStar M) s a

/-- V* equals the max over Q*: V*(s) = max_a Q*(s,a).
    This follows directly from V* being a fixed point of T*. -/
theorem vStar_eq_max_qStar (s : M.S) :
    vStar M s = Finset.sup' Finset.univ ⟨default, Finset.mem_univ _⟩ (qStar M s) := by
  -- V*(s) = T*(V*)(s) = max_a Q(s,a,V*) = max_a Q*(s,a)
  have := congr_fun (vStar_is_fixed M) s
  simp only [bellmanOptOp, qStar] at this ⊢
  exact this.symm

end MDP

end
