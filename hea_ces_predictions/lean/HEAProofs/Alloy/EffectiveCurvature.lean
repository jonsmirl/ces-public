/-
  Effective Curvature properties: K_eff ordering energy Ω,
  and the relationship between K_eff and phase stability.
  Adapted from CESProofs/Potential/EffectiveCurvature.lean.
-/

import HEAProofs.Alloy.Curvature

open Real Finset BigOperators

noncomputable section

variable {J : ℕ}

-- ============================================================
-- Section 1: Ordering Energy and Critical Temperature
-- ============================================================

/-- Effective ordering energy: Ω = T_m · ΔS / |ΔH|.
    At T = 0, the full curvature K is available.
    At T = T*, the ordering energy exactly compensates curvature.

    In the CES potential framework, T* is the critical temperature
    where K_eff vanishes — the alloy transitions from ordered
    (complementary) to disordered (linear/ideal). -/
def effectiveOrdering (T_m ΔS ΔH : ℝ) : ℝ :=
  T_m * ΔS / |ΔH|

/-- The degradation function f(t) = (1 - t/T*)⁺ is monotone decreasing. -/
theorem degradation_monotone {T₁ T₂ Tstar : ℝ} (hTs : 0 < Tstar)
    (h12 : T₁ ≤ T₂) :
    max 0 (1 - T₂ / Tstar) ≤ max 0 (1 - T₁ / Tstar) := by
  apply max_le_max_left 0
  apply sub_le_sub_left
  exact div_le_div_of_nonneg_right h12 (le_of_lt hTs)

-- ============================================================
-- Section 2: K² vs K Sensitivity
-- ============================================================

/-- Correlation robustness (∝ K²) degrades faster than superadditivity (∝ K).
    For f = 1 - T/T* ∈ (0,1): f² < f.

    In the HEA context: the correlation between elemental properties
    (which scales as K²) is more sensitive to temperature than
    the basic mixing enhancement (which scales as K). -/
theorem K_squared_degrades_faster {x : ℝ} (hx_pos : 0 < x) (hx_lt : x < 1) :
    (1 - x) ^ 2 < (1 - x) := by
  have h1 : 0 < 1 - x := by linarith
  nlinarith [sq_nonneg x]

/-- Sensitivity ratio: K²/K = K < 1 when K ∈ (0,1). -/
theorem sensitivity_ratio {K : ℝ} (hK_pos : 0 < K) (hK_lt : K < 1) :
    K ^ 2 / K = K := by
  rw [sq, mul_div_cancel_of_imp]
  intro h; linarith

-- ============================================================
-- Section 3: General-Weight Effective Curvature
-- ============================================================

/-- General-weight effective curvature:
    K_eff(q, x, T) = K(q, x) · max(0, 1 - T/T*(x))
    For non-equimolar compositions. -/
def generalEffectiveCurvatureKeff
    (J : ℕ) (q : ℝ) (x : Fin J → ℝ) (T Tstar : ℝ) : ℝ :=
  generalCurvatureK J q x * max 0 (1 - T / Tstar)

/-- General K_eff vanishes above T*. -/
theorem generalKeff_above_critical
    {J : ℕ} {q : ℝ} {x : Fin J → ℝ} {T Tstar : ℝ}
    (hTs : 0 < Tstar) (hT : Tstar ≤ T) :
    generalEffectiveCurvatureKeff J q x T Tstar = 0 := by
  unfold generalEffectiveCurvatureKeff
  have h : 1 - T / Tstar ≤ 0 := by
    rw [sub_nonpos]
    rwa [le_div_iff₀ hTs, one_mul]
  rw [max_eq_left h, mul_zero]

/-- General K_eff at zero temperature equals K(q, x). -/
theorem generalKeff_zero_temp
    {J : ℕ} {q : ℝ} {x : Fin J → ℝ} {Tstar : ℝ} (_hTs : 0 < Tstar) :
    generalEffectiveCurvatureKeff J q x 0 Tstar
    = generalCurvatureK J q x := by
  unfold generalEffectiveCurvatureKeff
  simp [zero_div, sub_zero, mul_one]

end
