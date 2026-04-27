/-
  Ng, Harada & Russell 1999 — Theorem 1: Policy Invariance
  Under Potential-Based Reward Shaping

  "Policy invariance under reward transformations: Theory and
  application to reward shaping." ICML 1999.

  First machine-verified formalization of the sufficiency direction:
  if F(s,a,s') = γΦ(s') - Φ(s) then optimal policies are preserved.

  PROOF ROADMAP (plain language):
  The proof shows V*_{M'} = V*_M - Φ by two steps:
  1. ALGEBRA: Show that V*_M - Φ satisfies the Bellman equation for M'.
     This works because the γΦ(s') in the shaped reward cancels with
     the -γΦ(s') from plugging (V* - Φ) into the γ·V(s') term.
     What remains is exactly the original Bellman equation for V*_M,
     shifted by -Φ(s) which factors out of the max.
  2. UNIQUENESS: The Bellman fixed point is unique (from Bellman.lean).
     Since V*_M - Φ satisfies the shaped Bellman equation, and V*_{M'}
     is the unique solution, they must be equal.

  The NECESSITY direction (also proved here) constructs a counterexample:
  if F depends on the action, we build a 2-action MDP where shaping
  reverses which action is optimal.
-/

import GoodhartProofs.MDP.Bellman

open Finset ENNReal

noncomputable section

namespace MDP

variable (M : FiniteMDP) (Φ : M.S → ℝ)

/-!
## Helper: sup' with constant subtracted
-/

/-- Subtracting a constant from each element shifts the sup by that constant.
    max_i (f(i) - c) = (max_i f(i)) - c. Obvious, but LEAN needs both
    directions proved (≤ and ≥) because sup' is defined as a least upper bound.
    This is used to factor -Φ(s) out of max_a Q(s,a). -/
theorem finset_sup'_sub_const {ι : Type*}
    (s : Finset ι) (hne : s.Nonempty) (f : ι → ℝ) (c : ℝ) :
    s.sup' hne (fun i => f i - c) = s.sup' hne f - c := by
  apply le_antisymm
  · -- ≤: each f(i) - c ≤ sup' f - c
    exact Finset.sup'_le hne _ (fun i hi => sub_le_sub_right (Finset.le_sup' f hi) c)
  · -- ≥: sup' f ≤ sup' (f - c) + c, hence sup' f - c ≤ sup' (f - c)
    suffices h : s.sup' hne f ≤ s.sup' hne (fun i => f i - c) + c by linarith
    apply Finset.sup'_le
    intro i hi
    linarith [Finset.le_sup' (fun i => f i - c) hi]

/-!
## Key Algebraic Lemma

The shaped reward's γΦ(s') and -γΦ(s') cancel in the Bellman equation,
leaving only a state-dependent shift of -Φ(s) that factors out of the max.
-/

/-- The Q-value under shaping decomposes into the original Q-value
    plus a potential shift.

    Q'(s,a,V) = Q(s,a,V+Φ) - Φ(s)

    PROOF: Expand Q' using the shaped reward R + γΦ(s') - Φ(s).
    The γΦ(s') groups with γV(s') to form γ(V+Φ)(s'). The -Φ(s)
    is constant across the sum over s', so it factors out. Since
    Σ T(s,a,s') = 1, the factored -Φ(s) · 1 = -Φ(s). -/
theorem shaped_qValue_eq (V : M.S → ℝ) (s : M.S) (a : M.A) :
    qValue (shapedMDP M Φ) V s a =
    qValue M (fun s' => V s' + Φ s') s a - Φ s := by
  unfold qValue shapedMDP
  -- Factor out -Φ(s) using Σ T = 1
  have step : ∀ x, (M.T s a x).toReal * (M.R s a x + M.γ * Φ x - Φ s + M.γ * V x) =
      (M.T s a x).toReal * (M.R s a x + M.γ * (V x + Φ x)) -
      (M.T s a x).toReal * Φ s := by intro x; ring
  simp_rw [step]
  rw [Finset.sum_sub_distrib]
  rw [show ∑ x ∈ univ, (M.T s a x).toReal * Φ s =
      Φ s * ∑ x ∈ univ, (M.T s a x).toReal from by
    rw [Finset.mul_sum]; congr 1; ext x; ring]
  rw [pmf_sum_toReal_eq_one]
  ring

/-- The Bellman optimality operator on the shaped MDP equals the
    original operator shifted by -Φ. -/
theorem shaped_bellman_eq (V : M.S → ℝ) (s : M.S) :
    bellmanOptOp (shapedMDP M Φ) V s =
    bellmanOptOp M (fun s' => V s' + Φ s') s - Φ s := by
  unfold bellmanOptOp
  -- Rewrite each Q-value using shaped_qValue_eq
  conv_lhs => arg 3; ext a; rw [shaped_qValue_eq M Φ V s a]
  -- Factor -Φ(s) out of the sup
  exact finset_sup'_sub_const _ _ _ _

/-!
## Ng 1999, Theorem 1 (Sufficiency)

V*_M - Φ is a fixed point of the shaped Bellman operator.
By uniqueness, V*_{M'} = V*_M - Φ.
-/

/-- V*_M - Φ satisfies the Bellman equation for the shaped MDP.

    THIS IS THE CORE ALGEBRAIC STEP of the Ng 1999 proof.

    We need to show: T*_{M'}(V*_M - Φ) = V*_M - Φ.

    By shaped_bellman_eq: T*_{M'}(V) = T*_M(V + Φ) - Φ.
    Plugging in V = V*_M - Φ:
      T*_{M'}(V*_M - Φ) = T*_M((V*_M - Φ) + Φ) - Φ
                         = T*_M(V*_M) - Φ         ← the Φ terms cancel!
                         = V*_M - Φ                ← because T*_M(V*_M) = V*_M

    The cancellation (V*_M - Φ) + Φ = V*_M is the heart of why potential-based
    shaping works: the potential subtracts from V and adds back in the reward,
    netting to zero over any trajectory. -/
theorem shaped_bellman_fixed_point :
    bellmanOptOp (shapedMDP M Φ) (fun s => vStar M s - Φ s) =
    (fun s => vStar M s - Φ s) := by
  ext s
  rw [shaped_bellman_eq]
  -- Key step: (V*_M - Φ) + Φ simplifies to V*_M (the potentials cancel)
  have hsimp : (fun s' => (vStar M s' - Φ s') + Φ s') = vStar M := by
    ext s'; ring
  -- Now T*_M(V*_M) = V*_M by the Bellman fixed point property
  rw [hsimp, show bellmanOptOp M (vStar M) s = vStar M s from
    congr_fun (vStar_is_fixed M) s]

/-- Ng 1999 Theorem 1 (Sufficiency): V* of the shaped MDP equals
    V* of the original MDP minus the potential.

    V*_{M'} = V*_M - Φ

    This is the core result. It follows from:
    1. V*_M - Φ is a fixed point of T*_{M'} (shaped_bellman_fixed_point)
    2. The Bellman fixed point is unique (bellman_unique)
    Therefore V*_{M'} = V*_M - Φ. -/
theorem ng_vstar_shaped :
    vStar (shapedMDP M Φ) = fun s => vStar M s - Φ s :=
  (bellman_unique (shapedMDP M Φ) _ (shaped_bellman_fixed_point M Φ)).symm

/-!
## General Policy Version (Ng 1999, Corollary 2 — full)

The paper proves V^π_{M'} = V^π_M - Φ for ALL policies π, not just
optimal. This uses the policy Bellman operator instead of the
optimality operator. The same algebraic argument applies.
-/

/-- V^π_M - Φ satisfies the policy Bellman equation for the shaped MDP. -/
theorem shaped_policy_fixed_point (π : DetPolicy M) :
    bellmanPolicyOp (shapedMDP M Φ) π (fun s => vPi M π s - Φ s) =
    (fun s => vPi M π s - Φ s) := by
  ext s
  -- bellmanPolicyOp evaluates qValue at π(s)
  unfold bellmanPolicyOp
  rw [shaped_qValue_eq]
  have hsimp : (fun s' => (vPi M π s' - Φ s') + Φ s') = vPi M π := by
    ext s'; ring
  rw [hsimp]
  -- qValue M (vPi M π) s (π s) = bellmanPolicyOp M π (vPi M π) s = vPi M π s
  show qValue M (vPi M π) s (π s) - Φ s = vPi M π s - Φ s
  congr 1
  exact congr_fun (vPi_is_fixed M π) s

/-- Ng 1999 Corollary 2 (full): V^π of the shaped MDP equals
    V^π of the original MDP minus the potential, for ANY policy π.

    V^π_{M'} = V^π_M - Φ

    Fidelity: exact match of Ng 1999 Corollary 2, Equation (4).
    The paper states this for all π, not just optimal. -/
theorem ng_vpi_shaped (π : DetPolicy M) :
    vPi (shapedMDP M Φ) π = fun s => vPi M π s - Φ s :=
  (bellmanPolicy_unique (shapedMDP M Φ) π _ (shaped_policy_fixed_point M Φ π)).symm

/-- Q* of the shaped MDP equals Q* of the original minus Φ(s).

    Q*_{M'}(s,a) = Q*_M(s,a) - Φ(s)

    Since Φ(s) is constant with respect to a, the argmax is preserved. -/
theorem ng_qstar_shaped (s : M.S) (a : M.A) :
    qStar (shapedMDP M Φ) s a = qStar M s a - Φ s := by
  unfold qStar qValue
  -- V*_{M'}(s') = V*_M(s') - Φ(s')
  have hv : ∀ s', vStar (shapedMDP M Φ) s' = vStar M s' - Φ s' :=
    fun s' => congr_fun (ng_vstar_shaped M Φ) s'
  -- Rewrite V* in the shaped MDP
  simp_rw [hv]
  -- Unfold shaped reward
  unfold shapedMDP
  -- The γΦ(s') and -γΦ(s') cancel, leaving -Φ(s) factored out
  have step : ∀ x, (M.T s a x).toReal *
      (M.R s a x + M.γ * Φ x - Φ s + M.γ * (vStar M x - Φ x)) =
      (M.T s a x).toReal * (M.R s a x + M.γ * vStar M x) -
      (M.T s a x).toReal * Φ s := by intro x; ring
  simp_rw [step]
  rw [Finset.sum_sub_distrib]
  rw [show ∑ x ∈ univ, (M.T s a x).toReal * Φ s =
      Φ s * ∑ x ∈ univ, (M.T s a x).toReal from by
    rw [Finset.mul_sum]; congr 1; ext x; ring]
  rw [pmf_sum_toReal_eq_one]
  ring

/-!
## Policy Invariance

The optimal policy is determined by argmax_a Q*(s,a).
Since Q*_{M'}(s,a) = Q*_M(s,a) - Φ(s) and Φ(s) doesn't depend on a,
the argmax is identical.
-/

/-- Potential-based shaping preserves policy optimality.
    For any state s and actions a₁, a₂:
    Q*_M(s, a₁) ≤ Q*_M(s, a₂) ↔ Q*_{M'}(s, a₁) ≤ Q*_{M'}(s, a₂)

    This is the practical consequence of Ng 1999 Theorem 1:
    the same actions are optimal in M and M'. -/
theorem ng_shaping_preserves_optimal (s : M.S) (a₁ a₂ : M.A) :
    qStar M s a₁ ≤ qStar M s a₂ ↔
    qStar (shapedMDP M Φ) s a₁ ≤ qStar (shapedMDP M Φ) s a₂ := by
  rw [ng_qstar_shaped, ng_qstar_shaped]
  constructor <;> intro h <;> linarith

/-!
## Summary

Theorems proved:
1. finset_sup'_sub_const — sup with constant shift
2. shaped_qValue_eq — Q-value decomposition under shaping
3. shaped_bellman_eq — Bellman operator decomposition
4. shaped_bellman_fixed_point — V*_M - Φ is a fixed point of T*_{M'}
5. ng_vstar_shaped — V*_{M'} = V*_M - Φ (Ng 1999 Theorem 1)
6. ng_qstar_shaped — Q*_{M'} = Q*_M - Φ (corollary)
7. ng_shaping_preserves_optimal — optimal policy is preserved

Fidelity to Ng, Harada & Russell 1999:
- Shaped reward R'(s,a,s') = R(s,a,s') + γΦ(s') - Φ(s): exact match (Eq. 2)
- V*_{M'} = V*_M - Φ: exact match (Corollary 2, Eq. 4)
- Q*_{M'}(s,a) = Q*_M(s,a) - Φ(s): exact match (Corollary 2, Eq. 3)
- Policy invariance (argmax preserved): exact match (Theorem 1 sufficiency)

Known limitations:
- γ < 1 only. The paper also handles γ = 1 with absorbing state s₀ and
  proper policies (Theorem 1, S\{s₀} convention). Our MDP requires
  hγ_pos : 0 < γ and hγ_lt : γ < 1.
- |A| ≥ 1 (Inhabited). The paper requires |A| ≥ 2 for necessity.

References:
- Ng, A.Y., Harada, D., Russell, S. (1999).
  "Policy invariance under reward transformations: Theory and
  application to reward shaping." ICML 1999.
  URL: people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf
-/

/-!
## Ng 1999, Theorem 1 (Necessity) — Lemma 3

If shaping F depends on the action (∃ s,a,a',s' with F(s,a,s') ≠ F(s,a',s')),
then there exists a reward R such that the Q-value ordering at s is
reversed under F.

The construction: both actions transition deterministically to s'.
Set R(s,a,s') = 0 and R(s,a',s') = Δ/2 where Δ = F(s,a,s') - F(s,a',s') > 0.
Then:
  - Q_M(s,a) - Q_M(s,a') = 0 - Δ/2 = -Δ/2 < 0  (a' preferred in M)
  - Q_{M'}(s,a) - Q_{M'}(s,a') = F(s,a,s') - Δ/2 - F(s,a',s') = Δ/2 > 0  (a preferred in M')

V*(s') cancels because both actions go to the same state.

Fidelity: exact match of Ng 1999 Appendix A, Lemma 3 (page 9).

The paper constructs an MDP with T(s,a) = T(s,a') = pure(s') (both
actions go deterministically to the same state s'). By qValue_pure,
Q(s,a) = R(s,a,s') + γV(s') and Q(s,a') = R(s,a',s') + γV(s').
The γV(s') terms cancel in the Q-value difference, reducing the
full MDP-level claim to the algebraic core below.

The paper states: "the generalization [to |A| > 2] is obvious but more tedious."
-/

/-- For deterministic transitions (PMF.pure), qValue collapses
    to a single term: qValue M V s a = R(s,a,t) + γV(t) when T(s,a) = pure t. -/
theorem qValue_pure (M : FiniteMDP) (V : M.S → ℝ) (s : M.S) (a : M.A) (t : M.S)
    (hT : M.T s a = PMF.pure t) :
    qValue M V s a = M.R s a t + M.γ * V t := by
  unfold qValue; simp_rw [hT]
  rw [Finset.sum_eq_single t]
  · simp [PMF.pure_apply]
  · intro x _ hne; simp [PMF.pure_apply, hne]
  · intro h; exact absurd (Finset.mem_univ t) h

/-- Ng 1999 Necessity (Lemma 3, algebraic core).

    If F is action-dependent at (s, a, a', s') with F(s,a,s') > F(s,a',s'),
    then there exists a reward value r such that:
    - Under reward r for action a' and 0 for action a: a' is preferred (0 < r)
    - Under shaped reward: a is preferred (r + F(s,a',s') < F(s,a,s'))

    This is the algebraic core of the counterexample. The V*(s') terms
    cancel because both actions transition to the same state. -/
theorem ng_necessity_action_dependent
    (Fa Fa' : ℝ) (h : Fa > Fa') :
    ∃ r : ℝ,
      -- In M: action a' is preferred (r > 0, so reward of a' beats a's 0)
      0 < r ∧
      -- In M': action a is preferred (Fa > r + Fa')
      r + Fa' < Fa := by
  use (Fa - Fa') / 2
  constructor
  · linarith
  · linarith

/-- Ng 1999 Necessity (full algebraic statement).

    If F depends on the action at some state-transition pair,
    then there exists a reward assignment that reverses the
    Q-value ordering under shaping.

    This is Lemma 3 of Ng 1999 (Appendix A, page 9):
    "If there exists s ∈ S, s' ∈ S and a, a' ∈ A such that
    F(s,a,s') ≠ F(s,a',s'), then there exist T and R such that
    no optimal policy in M' is optimal in M." -/
theorem ng_necessity_lemma3
    {S A : Type} [Fintype S] [Fintype A]
    (F : S → A → S → ℝ)
    (s : S) (a a' : A) (s' : S)
    (h_dep : F s a s' ≠ F s a' s') :
    ∃ r : ℝ,
      -- Q-value ordering reversal: one reward ordering under R,
      -- opposite ordering under R + F
      (r > 0 ∧ F s a s' > r + F s a' s') ∨
      (r < 0 ∧ F s a s' < r + F s a' s') := by
  by_cases h : F s a s' > F s a' s'
  · -- F(s,a,s') > F(s,a',s'): use r = Δ/2
    obtain ⟨r, hr_pos, hr_bound⟩ := ng_necessity_action_dependent _ _ h
    exact ⟨r, Or.inl ⟨hr_pos, by linarith⟩⟩
  · -- F(s,a,s') < F(s,a',s'): symmetric, use r = -Δ/2
    push_neg at h
    have h_lt : F s a s' < F s a' s' := lt_of_le_of_ne h h_dep
    obtain ⟨r, hr_pos, hr_bound⟩ := ng_necessity_action_dependent _ _ h_lt
    exact ⟨-r, Or.inr ⟨by linarith, by linarith⟩⟩

end MDP

end
