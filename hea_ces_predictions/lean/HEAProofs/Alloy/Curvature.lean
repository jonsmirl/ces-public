/-
  Curvature K and K_eff for alloys.
  K = (1-q)(1-H) is the master parameter controlling all four
  HEA core effects simultaneously.
-/

import HEAProofs.Alloy.PartitionFunction

open Real Finset BigOperators

noncomputable section

variable {J : ℕ}

-- ============================================================
-- Section 1: Curvature K Properties
-- ============================================================

/-- At equimolar composition: K = (1-q)(J-1)/J.
    This is the maximum curvature for given q and J. -/
theorem K_equimolar (hJ : 0 < J) {q : ℝ} :
    curvatureK J q = (1 - q) * (↑J - 1) / ↑J := rfl

/-- K is the Gini-Simpson diversity index scaled by (1-q).
    GS = 1 - H = 1 - Σ xⱼ², so K = (1-q) · GS.
    This connects curvature to biodiversity/concentration measures. -/
theorem K_eq_gini_simpson_scaled (q : ℝ) (x : Fin J → ℝ) :
    generalCurvatureK J q x = (1 - q) * (1 - herfindahlIndex J x) := by
  simp only [generalCurvatureK, herfindahlIndex]

-- ============================================================
-- Section 2: Effective Curvature K_eff
-- ============================================================

/-- The effective curvature under ordering energy (temperature):
    K_eff = K · max(0, 1 - T/T*)

    In the alloy context, T is the actual temperature and T* is the
    critical temperature above which the alloy disorders completely.
    At T < T*: complementarity benefits survive (K_eff > 0).
    At T ≥ T*: the alloy behaves as an ideal solution (K_eff = 0). -/
def effectiveCurvatureKeff (J : ℕ) (q T Tstar : ℝ) : ℝ :=
  curvatureK J q * max 0 (1 - T / Tstar)

/-- K_eff = K when T = 0 (ground state). -/
theorem effectiveCurvatureKeff_zero_temp (J : ℕ) (q Tstar : ℝ)
    (_hTs : 0 < Tstar) :
    effectiveCurvatureKeff J q 0 Tstar = curvatureK J q := by
  simp only [effectiveCurvatureKeff, zero_div, sub_zero]
  rw [max_eq_right (zero_le_one), mul_one]

/-- K_eff = 0 when T ≥ T* (disordered regime). -/
theorem effectiveCurvatureKeff_above_critical (J : ℕ) (q T Tstar : ℝ)
    (hTs : 0 < Tstar) (hT : Tstar ≤ T) :
    effectiveCurvatureKeff J q T Tstar = 0 := by
  simp only [effectiveCurvatureKeff]
  have h : 1 - T / Tstar ≤ 0 := by
    rw [sub_nonpos]
    rwa [le_div_iff₀ hTs, one_mul]
  rw [max_eq_left h, mul_zero]

/-- K_eff ≥ 0 always (non-negative by the max construction). -/
theorem effectiveCurvatureKeff_nonneg (hJ : 2 ≤ J) {q : ℝ} (hq : q < 1)
    (T Tstar : ℝ) :
    0 ≤ effectiveCurvatureKeff J q T Tstar := by
  simp only [effectiveCurvatureKeff]
  apply mul_nonneg
  · exact le_of_lt (curvatureK_pos hJ hq)
  · exact le_max_left 0 _

/-- K_eff ≤ K always (temperature only degrades curvature). -/
theorem effectiveCurvatureKeff_le_K (hJ : 2 ≤ J) {q : ℝ} (hq : q < 1)
    {T Tstar : ℝ} (hT : 0 ≤ T) (hTs : 0 < Tstar) :
    effectiveCurvatureKeff J q T Tstar ≤ curvatureK J q := by
  simp only [effectiveCurvatureKeff]
  have hK : 0 < curvatureK J q := curvatureK_pos hJ hq
  have h1 : max 0 (1 - T / Tstar) ≤ 1 := by
    apply max_le (by linarith)
    rw [sub_le_self_iff]
    exact div_nonneg hT (le_of_lt hTs)
  calc curvatureK J q * max 0 (1 - T / Tstar)
      ≤ curvatureK J q * 1 := by
        exact mul_le_mul_of_nonneg_left h1 (le_of_lt hK)
    _ = curvatureK J q := by ring

/-- K_eff is strictly positive when T < T* (sub-critical). -/
theorem effectiveCurvatureKeff_pos (hJ : 2 ≤ J) {q : ℝ} (hq : q < 1)
    {T Tstar : ℝ} (_hT : 0 ≤ T) (hTs : 0 < Tstar) (hTlt : T < Tstar) :
    0 < effectiveCurvatureKeff J q T Tstar := by
  simp only [effectiveCurvatureKeff]
  apply mul_pos (curvatureK_pos hJ hq)
  rw [lt_max_iff]
  right
  rw [sub_pos, div_lt_one hTs]
  exact hTlt

end
