/-
  Core definitions for the Lean formalization of
  "q-Thermodynamics of High-Entropy Alloys"

  Adapted from CESProofs/Foundations/Defs.lean.
  Contains the CES partition function Z_q, power mean, curvature K,
  and symmetric point — the mathematical core shared with the
  economics formalization.
-/

import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Topology.Order.Basic
import Mathlib.Order.Monotone.Basic

open Real Finset BigOperators

noncomputable section

-- ============================================================
-- Section 1: CES Partition Function and Power Mean
-- ============================================================

/-- The CES partition function with general weights:
    Z_q(x) = ( Σ aⱼ · xⱼ^q )^(1/q).
    In the alloy context: aⱼ are intrinsic property weights,
    xⱼ are atomic fractions, and q is the complementarity parameter. -/
def cesFun (J : ℕ) (a : Fin J → ℝ) (q : ℝ) : (Fin J → ℝ) → ℝ :=
  fun x => (∑ j : Fin J, a j * (x j) ^ q) ^ (1 / q)

/-- Power mean of order q (CES with equal weights):
    M_q(x) = ( (1/J) · Σ xⱼ^q )^(1/q). -/
def powerMean (J : ℕ) (q : ℝ) (_hq : q ≠ 0) : (Fin J → ℝ) → ℝ :=
  fun x => ((1 / J : ℝ) * ∑ j : Fin J, (x j) ^ q) ^ (1 / q)

-- ============================================================
-- Section 2: Curvature Parameters
-- ============================================================

/-- Unified curvature parameter: K_H = (1 - q)(1 - H).
    The master curvature that controls all four HEA core effects.
    H is the Herfindahl index of composition weights
    (H = 1/J for equimolar). -/
def curvatureKH (q H : ℝ) : ℝ := (1 - q) * (1 - H)

/-- Equal-weight (equimolar) curvature: K = (1 - q)(J - 1) / J.
    Abbreviation for curvatureKH with H = 1/J. -/
def curvatureK (J : ℕ) (q : ℝ) : ℝ := (1 - q) * (↑J - 1) / ↑J

/-- curvatureK is curvatureKH with H = 1/J. -/
theorem curvatureK_eq_curvatureKH {J : ℕ} (hJ : 0 < J) {q : ℝ} :
    curvatureK J q = curvatureKH q (1 / ↑J) := by
  simp only [curvatureK, curvatureKH]
  have hJr : (0 : ℝ) < ↑J := Nat.cast_pos.mpr hJ
  have hJne : (J : ℝ) ≠ 0 := ne_of_gt hJr
  field_simp

/-- The symmetric point: all J inputs equal to c (equimolar composition). -/
def symmetricPoint (J : ℕ) (c : ℝ) : Fin J → ℝ := fun _ => c

-- ============================================================
-- Section 3: Basic Properties
-- ============================================================

/-- At the symmetric point, the power mean equals c:
    M_q(c, c, ..., c) = c.
    **Proof.** ((1/J) · J · c^q)^(1/q) = (c^q)^(1/q) = c. -/
theorem powerMean_symmetricPoint {J : ℕ} (hJ : 0 < J) {q : ℝ} (hq : q ≠ 0)
    {c : ℝ} (hc : 0 < c) :
    powerMean J q hq (symmetricPoint J c) = c := by
  simp only [powerMean, symmetricPoint]
  have hJr : (0 : ℝ) < ↑J := Nat.cast_pos.mpr hJ
  have hJne : (J : ℝ) ≠ 0 := ne_of_gt hJr
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin]
  rw [nsmul_eq_mul]
  rw [one_div, inv_mul_cancel_left₀ hJne]
  rw [← rpow_mul hc.le, mul_one_div_cancel hq, rpow_one]

/-- K > 0 when q < 1 and J ≥ 2.
    **Proof.** K = (1-q)(J-1)/J is a product of positive terms. -/
theorem curvatureK_pos {J : ℕ} (hJ : 2 ≤ J) {q : ℝ} (hq : q < 1) :
    0 < curvatureK J q := by
  simp only [curvatureK]
  apply div_pos
  · apply mul_pos
    · linarith
    · have hJ1 : (1 : ℝ) < ↑J := by exact_mod_cast (by omega : 1 < J)
      linarith
  · exact_mod_cast (by omega : 0 < J)

/-- K = 0 when q = 1 (rule-of-mixtures, linear). -/
theorem curvatureK_eq_zero_of_q_one {J : ℕ} :
    curvatureK J 1 = 0 := by
  simp [curvatureK]

end
