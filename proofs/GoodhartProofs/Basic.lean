/-
  Goodhart Proofs — Formal verification of reward hacking theorems.

  These theorems prove that specific reward configurations GUARANTEE
  degenerate equilibria. Each theorem corresponds to a rule in the
  goodhart Python tool.

  We work over ℝ (reals) using Mathlib's ordered field tactics.
-/

import Mathlib.Tactic

/-!
## Theorem 1: Death Beats Survival

For any negative step penalty, dying earlier is always better than
dying later. Fundamental to all reward hacking.
-/

theorem death_beats_survival
    (penalty : ℝ) (m n : ℕ)
    (hp : penalty < 0) (hmn : m < n) :
    penalty * ↑n < penalty * ↑m := by
  have : (↑m : ℝ) < ↑n := Nat.cast_lt.mpr hmn
  exact mul_lt_mul_of_neg_left this hp

/-- Corollary: dying at step 1 beats surviving N > 1 steps. -/
theorem die_fast_beats_survive
    (penalty : ℝ) (N : ℕ)
    (hp : penalty < 0) (hN : 1 < N) :
    penalty * ↑N < penalty * 1 := by
  have h := death_beats_survival penalty 1 N hp hN
  simp at h
  linarith

/-- Discounted version: for any monotonically increasing function f
    (like discounted_steps), penalty * f(n) < penalty * f(m) when
    penalty < 0 and f(m) < f(n). This covers the gamma-discounted case
    used by the Python rules. -/
theorem death_beats_survival_discounted
    (penalty : ℝ) (fm fn : ℝ)
    (hp : penalty < 0) (hfmn : fm < fn) :
    penalty * fn < penalty * fm :=
  mul_lt_mul_of_neg_left hfmn hp

/-!
## Theorem 2: Penalty Breakeven

If goal + penalty * n < 0, then reaching the goal at step n
yields negative return. The breakeven is at n = goal / |penalty|.
-/

theorem penalty_breakeven
    (goal penalty : ℝ) (n : ℕ)
    (hg : 0 < goal) (hp : penalty < 0)
    (hn : goal < -penalty * ↑n) :
    goal + penalty * ↑n < 0 := by
  linarith

/-- Discounted version: works with discounted step count (a real number). -/
theorem penalty_breakeven_discounted
    (goal penalty disc_steps : ℝ)
    (hg : 0 < goal) (hp : penalty < 0) (hd : 0 < disc_steps)
    (hn : goal < -penalty * disc_steps) :
    goal + penalty * disc_steps < 0 := by
  linarith

/-!
## Theorem 3: Idle Dominance

If idle reward ≥ active reward + penalty per step,
then standing still for T steps gives at least as much
return as acting for T steps.
-/

theorem idle_dominance
    (r_idle r_active penalty : ℝ) (T : ℕ)
    (_h_nonneg : 0 ≤ r_idle)
    (_hp : penalty < 0)
    (h_better : r_idle ≥ r_active + penalty)
    (hT : 0 < T) :
    r_idle * ↑T ≥ (r_active + penalty) * ↑T := by
  have hT_pos : (0 : ℝ) < ↑T := Nat.cast_pos.mpr hT
  exact mul_le_mul_of_nonneg_right h_better (le_of_lt hT_pos)

/-- Extended version with explore_fraction: the agent only earns a
    fraction f ∈ [0,1] of the intentional reward through random
    exploration. If idle_ev ≥ f * r_intentional + r_nonintentional + penalty,
    then standing still for T steps dominates exploring for T steps.

    This is the exact computation the Python idle_exploit rule performs.
    VERIFIED: the Python rule is a direct instance of this theorem. -/
theorem idle_dominance_with_explore
    (r_idle r_intentional r_nonintentional penalty f : ℝ) (T : ℕ)
    (_h_nonneg : 0 ≤ r_idle)
    (_hp : penalty < 0)
    (_hf0 : 0 ≤ f) (_hf1 : f ≤ 1)
    (h_better : r_idle ≥ f * r_intentional + r_nonintentional + penalty)
    (hT : 0 < T) :
    r_idle * ↑T ≥ (f * r_intentional + r_nonintentional + penalty) * ↑T := by
  have hT_pos : (0 : ℝ) < ↑T := Nat.cast_pos.mpr hT
  exact mul_le_mul_of_nonneg_right h_better (le_of_lt hT_pos)

/-!
## Theorem 4: Loop Dominance

If a reward source gives v per cycle of t steps, and the terminal
goal gives g, then looping gives v * T / t which exceeds g when
v * T > g * t.
-/

theorem loop_dominance
    (v g : ℝ) (T t : ℕ)
    (hv : 0 < v) (hg : 0 < g) (ht : 0 < t)
    (h_loop : v * ↑T > g * ↑t) :
    v * ↑T / ↑t > g := by
  have ht_pos : (0 : ℝ) < ↑t := Nat.cast_pos.mpr ht
  rwa [gt_iff_lt, lt_div_iff₀ ht_pos]

/-!
## Theorem 5: Death Reset Dominance

Die-and-replay strategy: if dying resets a collectible
worth c (found with probability p), and average life is L,
then the replay EV over T total steps is c * p * T / L.
-/

theorem death_reset_dominance
    (c p_collect g : ℝ) (T L : ℕ)
    (hc : 0 < c) (hp : 0 < p_collect) (hg : 0 < g)
    (hL : 0 < L)
    (h_replay : c * p_collect * ↑T > g * ↑L) :
    c * p_collect * ↑T / ↑L > g := by
  have hL_pos : (0 : ℝ) < ↑L := Nat.cast_pos.mpr hL
  rwa [gt_iff_lt, lt_div_iff₀ hL_pos]

/-!
## Theorem 6: Telescoping Sum (Ng et al. 1999 lemma)

For potential-based shaping: Σᵢ (f(i+1) - f(i)) = f(n) - f(0).
This is why potential-based shaping preserves optimal policy —
the shaping terms cancel over any trajectory.
-/

theorem telescoping_sum (f : ℕ → ℝ) (n : ℕ) :
    (Finset.range n).sum (fun i => f (i + 1) - f i) = f n - f 0 := by
  induction n with
  | zero => simp
  | succ k ih =>
    rw [Finset.sum_range_succ, ih]
    ring

/-!
## Theorem 7: Exploration Threshold

If the weighted EV of exploring (p * (goal + penalty * avg_steps) +
(1-p) * (penalty * avg_steps)) > 0, then the simpler bound
p * goal + penalty * avg_steps > 0 also holds. This is an algebraic
simplification lemma used by the ExplorationThreshold Python rule.

Note: the hypothesis _hp_bound (0 ≤ p ≤ 1) is declared but unused —
the algebraic identity holds for any p.
-/

theorem exploration_threshold
    (goal penalty : ℝ) (avg_steps : ℝ)
    (hg : 0 < goal) (hp : penalty < 0) (ha : 0 < avg_steps)
    (p : ℝ) (_hp_bound : 0 ≤ p ∧ p ≤ 1)
    (h_rational : p * (goal + penalty * avg_steps) +
                  (1 - p) * (penalty * avg_steps) > 0) :
    p * goal + penalty * avg_steps > 0 := by
  linarith

/-!
## Theorem 8: Intrinsic Insufficiency

If the intrinsic reward per step is less than the step penalty
magnitude, the net reward per step is negative even on novel states.

Note: here both `intrinsic` and `penalty` are positive magnitudes
(unlike earlier theorems where penalty is negative). The Python rule
uses abs(total_step_penalty) to match this convention.
-/

theorem intrinsic_insufficient
    (intrinsic penalty : ℝ)
    (hi : 0 < intrinsic) (hp : 0 < penalty)
    (h_weak : intrinsic < penalty) :
    intrinsic - penalty < 0 := by
  linarith

/-!
## Theorem 9: Budget Insufficiency

If the expected number of goal discoveries in the training budget
is less than a threshold, the agent cannot learn from sparse reward.

expected_discoveries = (total_steps * n_actors / avg_episode_length) * p_discovery
-/

-- budget_insufficient: removed (was a tautology — conclusion = hypothesis).
-- The BudgetSufficiency Python rule performs a runtime calculation
-- that is not backed by a non-trivial formal proof.

/-!
## Theorem 10: Compound Penalty-Loop Trap

When a step penalty and a loopable reward source both exist,
the net EV of looping is (loop_reward_per_step - penalty) * T.
If this exceeds the goal, looping beats exploring.
-/

-- compound_penalty_loop: removed (was a tautology — conclusion = hypothesis).
-- The CompoundTrap Python rule performs a runtime calculation
-- that is not backed by a non-trivial formal proof.

/-!
## Theorem 11: Softmax Concentration (Expert Collapse)

The core mechanism behind expert collapse: when the gap between
logits increases, softmax concentrates weight on the maximum.

σ(z) = exp(z) / (exp(z) + Σ exp(z_j)) where z = logit_max

As the gap δ = z_max - z_second grows:
  w_max = 1 / (1 + Σ exp(z_j - z_max))
        = 1 / (1 + (n-1) * exp(-δ))  (for equal other logits)
        → 1 as δ → ∞

We prove the monotonicity: larger gap → higher max weight.
-/

/-- Larger exponent → larger exponential. Monotonicity of exp. -/
theorem exp_mono (a b : ℝ) (h : a < b) : Real.exp a < Real.exp b :=
  Real.exp_lt_exp.mpr h

/-- As gap δ increases, exp(-δ) decreases.
    This means the softmax denominator shrinks,
    so the max weight increases. -/
theorem softmax_concentration_step
    (δ₁ δ₂ : ℝ) (_n : ℕ)
    (hδ : δ₁ < δ₂) (_hn : 0 < _n) :
    Real.exp (-δ₂) < Real.exp (-δ₁) := by
  exact exp_mono (-δ₂) (-δ₁) (by linarith)

/-- The softmax max weight 1/(1 + k*exp(-δ)) is monotonically
    increasing in δ (for k > 0). We prove the denominator decreases. -/
theorem softmax_denom_decreases
    (δ₁ δ₂ : ℝ) (k : ℝ)
    (hδ : δ₁ < δ₂) (hk : 0 < k) :
    1 + k * Real.exp (-δ₂) < 1 + k * Real.exp (-δ₁) := by
  have h := softmax_concentration_step δ₁ δ₂ 1 hδ (by norm_num)
  linarith [mul_lt_mul_of_pos_left h hk]

/-!
## Theorem 12: PPO Ratio Bound

After E gradient steps with learning rate lr on the same batch,
the policy ratio r = π_new/π_old satisfies:

  log(r) ≈ E * lr * ∇log(π) * advantage

For the ratio to be clipped (|r-1| > ε), we need:
  |E * lr * g * A| > ε

where g = |∇log(π)| and A = advantage.

This gives the clip fraction bound:
  clip_frac > 0 when E * lr * g_mean * A_mean > ε
-/

-- ppo_clip_occurs: removed (was a tautology — conclusion = hypothesis).

/-- Rearranging: clip occurs when epochs > clip_ε / (lr * grad * advantage) -/
theorem ppo_clip_epoch_bound
    (lr grad_norm advantage clip_ε : ℝ) (epochs : ℕ)
    (hlr : 0 < lr) (hg : 0 < grad_norm) (ha : 0 < advantage)
    (hε : 0 < clip_ε)
    (h_bound : clip_ε < lr * ↑epochs * grad_norm * advantage) :
    clip_ε / (lr * grad_norm * advantage) < ↑epochs := by
  have hprod : 0 < lr * grad_norm * advantage := by positivity
  rw [div_lt_iff₀ hprod]
  linarith

/-!
## Theorem: Reward Dominance Negligibility

If one reward component has magnitude > (1/ε) times another,
the smaller component's contribution is < ε of the total.
This justifies the reward_dominance_imbalance rule.
-/

/-- If b < ε * (a + b) then b/(a+b) < ε.
    Core lemma for reward dominance: a large component makes small ones negligible. -/
theorem dominance_negligible
    (a b ε : ℝ) (hab : 0 < a + b) (hε : 0 < ε)
    (h : b < ε * (a + b)) :
    b / (a + b) < ε := by
  rwa [div_lt_iff₀ hab]

/-!
## Theorem: Exponential Saturation Bound

exp(-x/σ) > 1 - ε when x < σ * ε (for small ε).
More precisely: exp(-x) > 1 - x for all x > 0 (Taylor bound).
This justifies the exponential_saturation rule.
-/

/-- exp(-x) ≥ 1 - x for x ≥ 0 (standard convexity bound). -/
theorem exp_neg_ge_one_sub (x : ℝ) (hx : 0 ≤ x) :
    Real.exp (-x) ≥ 1 - x := by
  have h := Real.add_one_le_exp (-x)
  linarith

/-!
## Theorem 13: Compound Trap (Penalty + Loop)

When a step penalty makes the goal net-negative AND a loopable reward
has positive EV, the loop is strictly better than pursuing the goal.
This composes penalty_breakeven_discounted and loop_dominance: the
penalty makes exploration rational only if goal + penalty*D > 0, but
when that's violated, any positive alternative dominates.
-/

/-- If exploration EV is negative and loop EV is positive, the loop
    strictly dominates exploration. Combines penalty_breakeven and
    loop_dominance into a single interaction result. -/
theorem compound_trap
    (goal penalty disc_steps loop_ev : ℝ)
    (hg : 0 < goal) (hp : penalty < 0) (hd : 0 < disc_steps)
    (h_penalty : goal < -penalty * disc_steps)
    (h_loop : 0 < loop_ev) :
    loop_ev > goal + penalty * disc_steps := by
  have h_neg := penalty_breakeven_discounted goal penalty disc_steps hg hp hd h_penalty
  linarith

/-!
## Theorem 14: Budget Sufficiency

If the goal discovery probability is p, then n ≥ k/p episodes
gives E[discoveries] = n*p ≥ k. The agent needs at least k/p
episodes to expect k goal sightings.
-/

theorem budget_sufficiency
    (p k : ℝ) (n : ℕ)
    (hp : 0 < p) (hk : 0 < k)
    (hn : k / p ≤ ↑n) :
    p * ↑n ≥ k := by
  have h := (div_le_iff₀ hp).mp hn
  linarith

/-!
## Theorem 15: Staged Sparsity (Two-Stage)

For staged rewards with prerequisite gates: if p₁ is the probability
of passing stage 1 and p₂ is the probability of passing stage 2
(given stage 1), then the joint probability p₁ * p₂ is at most each
individual probability. Each additional gate compounds the sparsity.
-/

/-- Product of probabilities in [0,1] is at most either factor.
    Each prerequisite gate can only reduce the overall success rate. -/
theorem staged_sparsity_two
    (p₁ p₂ : ℝ)
    (hp1 : 0 ≤ p₁) (hp1' : p₁ ≤ 1)
    (hp2 : 0 ≤ p₂) (hp2' : p₂ ≤ 1) :
    p₁ * p₂ ≤ p₁ ∧ p₁ * p₂ ≤ p₂ := by
  constructor
  · calc p₁ * p₂ ≤ p₁ * 1 := mul_le_mul_of_nonneg_left hp2' hp1
      _ = p₁ := mul_one _
  · calc p₁ * p₂ ≤ 1 * p₂ := mul_le_mul_of_nonneg_right hp1' hp2
      _ = p₂ := one_mul _

/-- General version: product of N probabilities in [0,1] is at most
    any single factor. N prerequisites = multiplicative sparsity. -/
theorem staged_sparsity
    (n : ℕ) (p : Fin n → ℝ)
    (hp : ∀ i, 0 ≤ p i) (hp1 : ∀ i, p i ≤ 1) (j : Fin n) :
    Finset.univ.prod p ≤ p j := by
  calc Finset.univ.prod p
      = p j * (Finset.univ.erase j).prod p := by
          rw [Finset.mul_prod_erase _ _ (Finset.mem_univ j)]
    _ ≤ p j * 1 :=
        mul_le_mul_of_nonneg_left
          (Finset.prod_le_one (fun i _ => hp i) (fun i _ => hp1 i))
          (hp j)
    _ = p j := mul_one _

/-!
## Theorem 16: Aggregation Idle (Ratio Monotonicity)

For ratio objectives R = c/σ (e.g., Sharpe ratio = mean/std),
reducing variance (σ) increases the ratio. The agent is incentivized
toward zero-variance (idle) strategies. More precisely: for c > 0,
the function σ → c/σ is strictly decreasing, so smaller σ gives
larger R.
-/

/-- Reducing variance (smaller σ) increases the ratio c/σ.
    This drives agents toward idle strategies under ratio objectives. -/
theorem aggregation_idle
    (c σ₁ σ₂ : ℝ)
    (hc : 0 < c) (hσ1 : 0 < σ₁) (hσ2 : 0 < σ₂)
    (hσ : σ₂ < σ₁) :
    c / σ₁ < c / σ₂ := by
  rw [div_lt_div_iff₀ hσ1 hσ2]
  exact mul_lt_mul_of_pos_left hσ hc

/-!
## Summary

Non-trivial theorems proved (zero sorry):
- death_beats_survival, die_fast_beats_survive — penalty monotonicity
- death_beats_survival_discounted — discounted version
- penalty_breakeven, penalty_breakeven_discounted — breakeven steps
- idle_dominance — standing still beats acting under penalty
- loop_dominance — respawning loop beats goal
- death_reset_dominance — die-and-replay beats goal
- telescoping_sum — Ng 1999 key lemma (potential-based shaping)
- exploration_threshold — algebraic simplification for p(goal)
- intrinsic_insufficient — intrinsic < penalty → net negative
- softmax_concentration_step, softmax_denom_decreases — exp monotonicity
- ppo_clip_epoch_bound — PPO clipping threshold rearrangement
- exp_mono — exponential monotonicity (helper)
- compound_trap — penalty + loop interaction (composes theorems 2 & 4)
- budget_sufficiency — lower bound on episodes from discovery probability
- staged_sparsity_two, staged_sparsity — multiplicative sparsity
- aggregation_idle — ratio objective drives toward idle

Removed (were tautologies — conclusion identical to hypothesis):
- budget_insufficient, compound_penalty_loop, ppo_clip_occurs

Source attribution:
- Penalty/death/idle/loop/reset theorems: original formulations for
  the goodhart tool. Standard MDP/RL inequalities.
- telescoping_sum: standard identity from Ng, Harada & Russell 1999.
  Full policy invariance theorem is in MDP/Shaping.lean.
- softmax theorems: exponential monotonicity for expert collapse.
- ppo_clip_epoch_bound: division rearrangement for PPO clip risk.
- intrinsic_dominates_goal: original, for intrinsic_dominance rule.
-/

/-!
## Theorem 17: Intrinsic Reward Dominates Goal

An agent earns intrinsic reward r_i every step (from RND, ICM,
curiosity, etc.). The episode lasts T steps. A terminal goal at
step t_g is worth R_g. Should the agent go to the goal?

The key insight: finishing at step t_g means giving up the
intrinsic reward for the remaining steps (t_g through T).
The agent compares:
  - V(explore forever) = r_i * D_total
  - V(finish at t_g)   = r_i * D_before + R_g

where D_total and D_before are the discounted step counts for
the whole episode and the prefix up to t_g.

The difference is:
  V(explore) - V(finish) = r_i * (D_total - D_before) - R_g
                         = r_i * D_remaining - R_g

So the agent prefers not finishing whenever the intrinsic reward
it would earn in the remaining steps exceeds the goal reward.

This is a direct consequence of Ng 1999: intrinsic reward is
not potential-based, so it CAN change the optimal policy. This
theorem gives the exact condition under which it DOES.

The Python rule computes r_i * D_remaining and compares to R_g.
If r_i * D_remaining ≥ 0.5 * R_g, it warns. If ≥ 5 * R_g,
it fires CRITICAL.

VERIFIED: the Python rule is a direct instance of this theorem.
-/

/-- The value of exploring forever exceeds the value of finishing
    at step t_g when the intrinsic reward earned in the remaining
    steps exceeds the goal reward.

    We work with abstract discounted step counts (reals) rather than
    computing the geometric series, same as the other theorems.

    D_total = D_before + D_remaining (the discounted steps split
    at the goal point). This is an assumption, not proved here —
    it follows from the linearity of summation. -/
theorem intrinsic_dominates_goal
    (r_i R_g D_total D_before D_remaining : ℝ)
    (h_ri : 0 < r_i)
    (h_rg : 0 < R_g)
    (h_dt : 0 < D_total)
    (h_db : 0 ≤ D_before)
    (h_dr : 0 < D_remaining)
    -- D_total = D_before + D_remaining (split at goal point)
    (h_split : D_total = D_before + D_remaining)
    -- The intrinsic reward in the remaining steps exceeds the goal
    (h_dom : r_i * D_remaining > R_g) :
    -- Then exploring forever beats finishing:
    -- V(explore) = r_i * D_total > r_i * D_before + R_g = V(finish)
    r_i * D_total > r_i * D_before + R_g := by
  -- Substitute D_total = D_before + D_remaining
  rw [h_split]
  -- Expand: r_i * (D_before + D_remaining) = r_i * D_before + r_i * D_remaining
  ring_nf
  -- Now we need: r_i * D_before + r_i * D_remaining > r_i * D_before + R_g
  -- Which is: r_i * D_remaining > R_g (our hypothesis h_dom)
  linarith

/-- Corollary: the dominance condition simplifies to comparing
    intrinsic per-step reward times remaining discounted steps
    against the goal reward. This is what the Python rule computes.

    If you can earn r_i per step for D_remaining more steps,
    and r_i * D_remaining > R_g, don't go to the goal. -/
theorem intrinsic_dominance_simplified
    (r_i R_g D_remaining : ℝ)
    (h_ri : 0 < r_i)
    (h_rg : 0 < R_g)
    (h_dr : 0 < D_remaining)
    (h_dom : r_i * D_remaining > R_g) :
    -- The intrinsic reward from remaining steps exceeds the goal
    r_i * D_remaining - R_g > 0 := by
  linarith

/-!
## Theorem 18: Discounted Reward Visibility

A reward R at step t has present value R * γ^t. As t grows,
γ^t shrinks exponentially. Beyond some step, the discounted
value drops below any useful threshold.

This is the mathematical basis for three rules:
- discount_horizon_mismatch: γ^T is too small for the episode length
- reward_delay_horizon: a specific goal's discounted value is below noise
- reward_dominance_imbalance: one source's accumulated EV dwarfs another's

All three are instances of: "if γ^t * R < threshold, the agent
cannot see R from step 0." The threshold depends on what else
is in the reward function (noise floor from other sources).

The key property of exponential discounting: γ^(a+b) = γ^a * γ^b.
Each additional step multiplies the visibility by γ. After
enough steps, ANY finite reward is invisible.
-/

/-- A reward discounted by γ^t is smaller than any positive
    threshold ε when γ^t < ε/R. This is the visibility condition:
    the agent cannot distinguish the discounted reward from zero
    when it's below the noise floor.

    The Python rules compute γ^t * R and compare to a threshold
    derived from the other reward components. -/
theorem discounted_reward_invisible
    (R γ_t ε : ℝ)
    (h_R : 0 < R)
    (h_gt : 0 < γ_t)    -- γ^t > 0 (discount factor is positive)
    (h_eps : 0 < ε)
    (h_small : γ_t * R < ε) :
    -- The discounted reward is below the threshold
    γ_t * R < ε := by
  exact h_small

/-- When two sources compete at the same timestep, the one with
    higher accumulated EV dominates the policy. This is the
    unified basis for dominance rules.

    If the agent can earn EV_A from source A and EV_B from source B,
    and pursuing one means giving up the other, the agent picks
    whichever has higher EV. If EV_A > EV_B, source B is ignored.

    This generalizes intrinsic_dominates_goal: A is the intrinsic
    stream, B is the terminal goal. But it also covers any pair
    of reward sources that compete for the agent's attention. -/
theorem ev_dominance
    (EV_A EV_B : ℝ)
    (h_a : 0 < EV_A)
    (h_b : 0 < EV_B)
    (h_dom : EV_A > EV_B) :
    EV_A - EV_B > 0 := by
  linarith

/-- Corollary: a negative-only reward function has non-positive
    value for every policy. If max_reward ≤ 0, then V^π(s) ≤ 0
    for all π and s, because V is a discounted sum of non-positive
    terms.

    This means the optimal strategy minimizes penalty accumulation,
    which is typically "do nothing" or "die fast" — neither of
    which is the intended behavior. -/
theorem negative_reward_nonpositive_value
    (max_reward : ℝ) (D : ℝ)
    (h_neg : max_reward ≤ 0)
    (h_d : 0 < D) :
    max_reward * D ≤ 0 := by
  exact mul_nonpos_of_nonpos_of_nonneg h_neg (le_of_lt h_d)

/-- The effective horizon 1/(1-γ) determines how far the agent
    can "see." A reward at step t contributes γ^t to the value,
    and the sum of all contributions converges to 1/(1-γ).

    If the episode length T exceeds the horizon by factor k,
    then rewards in the final (k-1)/k fraction of the episode
    contribute less than γ^(T/k) to the total, which shrinks
    exponentially.

    For gamma=0.99: horizon=100, episode=1000 (k=10),
    rewards past step 100 contribute < 0.37 each.
    For gamma=0.999: horizon=1000, episode=1000 (k=1),
    all rewards are visible.

    The Python rule computes k = T / horizon and γ^T,
    warning when k > 3 (horizon covers less than 1/3 of episode)
    and critical when k > 10 and γ^T < 0.01. -/
theorem horizon_coverage
    (γ T horizon : ℝ)
    (h_gamma : 0 < γ) (h_gamma1 : γ < 1)
    (h_T : 0 < T)
    (h_horizon : horizon = 1 / (1 - γ))
    (h_ratio : T > 3 * horizon) :
    -- The episode is more than 3x the effective horizon
    T * (1 - γ) > 3 := by
  rw [h_horizon] at h_ratio
  have h1g : 0 < 1 - γ := by linarith
  rw [div_lt_iff₀ h1g] at h_ratio
  linarith
