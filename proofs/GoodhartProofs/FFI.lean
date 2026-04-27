/-
  FFI-exported check functions for Python integration.

  These are runtime heuristic implementations of the reward trap
  checks, written in LEAN for compilation to C via ctypes. They
  use Float arithmetic and practical thresholds.

  These functions implement the reward trap checks in LEAN using
  Float arithmetic. They are type-checked by LEAN's type system
  (guaranteeing totality and type correctness) and compiled to
  native C via the LEAN compiler.

  What you get from calling these via ctypes:
  - LEAN type-checker guarantee: every function is total and type-correct
  - LEAN compiler correctness: the compiled C matches the LEAN source
  - Deterministic Float arithmetic (no Python interpreter variability)

  What you do NOT get:
  - A proof that Float arithmetic matches the ℝ theorems in Basic.lean
    (rounding, overflow, and precision are not formally verified)
  - The functions use practical thresholds (e.g., risk_score > 5.0)
    that are heuristic, not derived from the formal proofs

  The formal verification chain is:
    Basic.lean (ℝ theorems) → validates mathematical properties
    FFI.lean (Float, type-checked) → provides fast runtime checks
    Python rules (goodhart/rules/) → authoritative analysis logic
-/

/-- Check if dying at step 1 beats surviving N steps.
    Returns true if the death incentive exists. -/
@[export goodhart_check_death_beats_survival]
def checkDeathBeatsSurvival (penalty : Float) (surviveSteps : UInt32) : Bool :=
  -- penalty < 0 AND |penalty * 1| < |penalty * N|
  -- i.e., dying costs less than surviving
  penalty < 0 && surviveSteps.toFloat > 1.0

/-- Check if total discounted penalty exceeds goal reward.
    Returns true if penalty dominates. -/
@[export goodhart_check_penalty_dominance]
def checkPenaltyDominance (goal penalty : Float) (maxSteps : UInt32) (gamma : Float) : Bool :=
  if penalty >= 0 || goal <= 0 then false
  else
    let discountedSteps :=
      if gamma >= 1.0 then maxSteps.toFloat
      else (1.0 - gamma ^ maxSteps.toFloat) / (1.0 - gamma)
    Float.abs penalty * discountedSteps > goal

/-- Check if standing still beats exploring.
    Returns true if idle exploit exists. -/
@[export goodhart_check_idle_exploit]
def checkIdleExploit (idleRewardPerStep penaltyPerStep : Float)
    (goalReward discoveryProb : Float) (maxSteps : UInt32) : Bool :=
  let evIdle := idleRewardPerStep * maxSteps.toFloat
  let evExplore := penaltyPerStep * maxSteps.toFloat + goalReward * discoveryProb
  evIdle >= 0 && evIdle > evExplore

/-- Check if a looping strategy beats the terminal goal.
    Returns true if loop dominates. -/
@[export goodhart_check_loop_dominance]
def checkLoopDominance (loopValue : Float) (loopPeriod : UInt32)
    (goalReward : Float) (maxSteps : UInt32) : Bool :=
  if loopPeriod.toFloat <= 0 || goalReward < 0 then false
  else
    let evLoop := loopValue * maxSteps.toFloat / loopPeriod.toFloat
    evLoop > goalReward && evLoop > 0

/-- Check if die-and-replay strategy beats the goal.
    Returns true if death reset is exploitable. -/
@[export goodhart_check_death_reset]
def checkDeathReset (collectValue collectProb goalReward : Float)
    (maxSteps : UInt32) (deathProb : Float) : Bool :=
  if deathProb <= 0 || collectValue <= 0 then false
  else
    let avgLife := let x := 1.0 / deathProb; if x < maxSteps.toFloat then x else maxSteps.toFloat
    let nLives := maxSteps.toFloat / avgLife
    let evReplay := collectValue * collectProb * nLives
    evReplay > goalReward

/-- Check if intrinsic reward is insufficient to overcome penalty. -/
@[export goodhart_check_intrinsic_insufficient]
def checkIntrinsicInsufficient (intrinsicPerStep penaltyPerStep : Float) : Bool :=
  intrinsicPerStep > 0 && penaltyPerStep > 0 && intrinsicPerStep < penaltyPerStep

/-- Check exploration threshold: is p(goal) sufficient? -/
@[export goodhart_check_exploration_threshold]
def checkExplorationThreshold (goalReward penalty : Float)
    (maxSteps : UInt32) (pGoal : Float) : Bool :=
  if penalty >= 0 then false  -- no penalty, exploration is free
  else
    let avgSteps := maxSteps.toFloat / 2.0
    let evExplore := pGoal * (goalReward + penalty * avgSteps) +
                     (1.0 - pGoal) * (penalty * maxSteps.toFloat)
    -- penalty is always < 0 here (guarded above), so best degenerate is 0
    let bestDegenerate := 0.0
    evExplore < bestDegenerate  -- true means exploration is irrational

/-- Check softmax concentration (expert collapse risk).
    Returns true if the gap is large enough for collapse. -/
@[export goodhart_check_softmax_collapse]
def checkSoftmaxCollapse (numSpecialists : UInt32) (routingFloor : Float) : Bool :=
  numSpecialists.toFloat > 1.0 && routingFloor <= 0.0

/-- Check PPO clip fraction risk. -/
@[export goodhart_check_ppo_clip_risk]
def checkPpoClipRisk (lr : Float) (numEpochs : UInt32) (clipEpsilon : Float) : Bool :=
  let riskScore := (lr / 3e-4) * numEpochs.toFloat * (0.2 / clipEpsilon)
  riskScore > 5.0
