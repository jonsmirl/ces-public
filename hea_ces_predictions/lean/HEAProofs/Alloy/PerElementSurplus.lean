/-
  Per-Element Surplus: π_K(J) = K/J peaks at J=2.

  This is a key quantitative prediction of the CES framework:
  the marginal curvature benefit per element is maximized at
  the binary (J=2) composition. Adding a third, fourth, etc.
  element still increases total K, but at diminishing rate per element.
-/

import HEAProofs.Alloy.RadiationTolerance

open Real Finset BigOperators

noncomputable section

-- ============================================================
-- Section 1: Per-Element Surplus
-- ============================================================

/-- Per-element curvature surplus:
    π_K(J) = K(J) / J = (1-q)(J-1)/J²

    Measures the curvature contribution per element.
    This peaks at J = 2, explaining why binary intermetallics
    often show the strongest per-atom synergistic effects. -/
def perElementSurplus (J : ℕ) (q : ℝ) : ℝ :=
  curvatureK J q / ↑J

-- ============================================================
-- Section 2: Value at J = 2 and Peak
-- ============================================================

/-- **Theorem (Per-Element Surplus Value at J = 2)**:
    π_K(2) = (1-q)/4.

    The maximum per-element surplus occurs at the binary composition
    and equals exactly one quarter of the complementarity parameter. -/
theorem perElementSurplus_value_at_2 {q : ℝ} :
    perElementSurplus 2 q = (1 - q) / 4 := by
  simp only [perElementSurplus, curvatureK]
  push_cast
  ring

/-- **Theorem (Per-Element Surplus Peaks at J = 2)**:
    π_K(J) = (1-q)(J-1)/J² has its maximum at J = 2.

    For integer J: π_K(2) ≥ π_K(J) for all J ≥ 1.

    **Proof.** $(J-1)/J^2 \leq 1/4$ for all $J \geq 1$: this follows from
    $4(J-1) \leq J^2$, i.e., $(J-2)^2 \geq 0$, with equality at $J = 2$. -/
theorem perElementSurplus_peaks_at_2 {q : ℝ} (hq : q < 1) (J : ℕ) (hJ : 1 ≤ J) :
    perElementSurplus J q ≤ perElementSurplus 2 q := by
  rw [perElementSurplus_value_at_2]
  simp only [perElementSurplus, curvatureK]
  have h1q : 0 ≤ 1 - q := by linarith
  have hJpos : (0 : ℝ) < ↑J := by exact_mod_cast (by omega : 0 < J)
  have hkey : 4 * ((↑J : ℝ) - 1) ≤ (↑J : ℝ) * ↑J := by
    nlinarith [sq_nonneg ((↑J : ℝ) - 2)]
  rw [div_div]
  -- Goal: (1 - q) * (↑J - 1) / (↑J * ↑J) ≤ (1 - q) / 4
  -- Since 1-q ≥ 0, suffices: (↑J - 1) / (↑J * ↑J) ≤ 1 / 4
  -- Since ↑J * ↑J > 0 and 4 > 0, suffices: 4 * (↑J - 1) ≤ ↑J * ↑J
  have h4pos : (0 : ℝ) < 4 := by norm_num
  have hJJ : (0 : ℝ) < ↑J * ↑J := by positivity
  rw [div_le_div_iff₀ hJJ h4pos]
  nlinarith

end
