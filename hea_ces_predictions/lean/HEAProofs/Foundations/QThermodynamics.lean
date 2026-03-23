/-
  q-Logarithm, q-Exponential, and Tsallis Entropy.
  Adapted from CESProofs/Potential/Defs.lean.

  These are the core objects of Tsallis (non-extensive) statistical
  mechanics, which provides the thermodynamic framework for HEAs.
-/

import HEAProofs.Foundations.Defs

open Real Finset BigOperators

noncomputable section

variable {J : ℕ}

-- ============================================================
-- Section 1: q-Logarithm and q-Exponential
-- ============================================================

/-- The q-logarithm: ln_q(x) = (x^{1-q} - 1)/(1-q) for q ≠ 1, log(x) for q = 1.
    Generalizes the natural logarithm; recovers log as q → 1. -/
def qLog (q x : ℝ) : ℝ :=
  if q = 1 then Real.log x
  else (x ^ (1 - q) - 1) / (1 - q)

/-- The q-exponential: exp_q(x) = [1 + (1-q)x]_+^{1/(1-q)} for q ≠ 1, exp(x) for q = 1.
    The [·]_+ = max(0, ·) ensures compact support for q < 1. -/
def qExp (q x : ℝ) : ℝ :=
  if q = 1 then Real.exp x
  else (max 0 (1 + (1 - q) * x)) ^ (1 / (1 - q))

-- ============================================================
-- Section 2: Basic Properties
-- ============================================================

/-- q-log at x = 1 gives 0 for any q. -/
theorem qLog_one (q : ℝ) : qLog q 1 = 0 := by
  simp only [qLog]
  split_ifs with h
  · exact Real.log_one
  · simp [rpow_def_of_pos one_pos]

/-- q-exp at x = 0 gives 1 for any q. -/
theorem qExp_zero (q : ℝ) : qExp q 0 = 1 := by
  simp only [qExp]
  split_ifs with h
  · exact Real.exp_zero
  · simp [rpow_def_of_pos one_pos]

/-- For q ≠ 1, q-log has explicit form. -/
theorem qLog_eq_of_ne {q : ℝ} (hq : q ≠ 1) (x : ℝ) :
    qLog q x = (x ^ (1 - q) - 1) / (1 - q) := by
  simp [qLog, hq]

/-- For q ≠ 1, q-exp has explicit form. -/
theorem qExp_eq_of_ne {q : ℝ} (hq : q ≠ 1) (x : ℝ) :
    qExp q x = (max 0 (1 + (1 - q) * x)) ^ (1 / (1 - q)) := by
  simp [qExp, hq]

-- ============================================================
-- Section 3: Tsallis Entropy
-- ============================================================

/-- The Tsallis entropy of order q on the simplex:
    S_q(p) = (1 - Σ pⱼ^q) / (q - 1)  for q ≠ 1
    S_1(p) = -Σ pⱼ log(pⱼ)           for q = 1 (Shannon entropy).

    In the HEA context: S_q measures the effective configurational
    entropy of the alloy composition, with q encoding the degree
    of non-ideality (departure from random mixing). -/
def tsallisEntropy (J : ℕ) (q : ℝ) (p : Fin J → ℝ) : ℝ :=
  if q = 1 then -∑ j : Fin J, p j * Real.log (p j)
  else (1 - ∑ j : Fin J, (p j) ^ q) / (q - 1)

/-- Tsallis entropy at the uniform (equimolar) distribution.
    For q ≠ 1: (1 - J^{1-q}) / (q-1).
    For q = 1: log(J) (Shannon entropy of equimolar). -/
theorem tsallisEntropy_uniform (hJ : 0 < J) (q : ℝ) :
    tsallisEntropy J q (fun _ => (1 : ℝ) / ↑J) =
    if q = 1 then Real.log ↑J
    else (1 - (↑J : ℝ) ^ (1 - q)) / (q - 1) := by
  simp only [tsallisEntropy]
  have hJpos : (0 : ℝ) < ↑J := Nat.cast_pos.mpr hJ
  have hJne : (↑J : ℝ) ≠ 0 := ne_of_gt hJpos
  split_ifs with h
  · rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    rw [Real.log_div one_ne_zero hJne, Real.log_one, zero_sub]
    field_simp
  · congr 1; congr 1
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    rw [one_div, inv_rpow (le_of_lt hJpos)]
    rw [← rpow_neg (le_of_lt hJpos)]
    rw [show (1 : ℝ) - q = 1 + (-q) from by ring]
    rw [rpow_add hJpos, rpow_one]

end
