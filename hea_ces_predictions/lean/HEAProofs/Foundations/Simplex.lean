/-
  Simplex definitions and Herfindahl index.
  Adapted from CESProofs Potential/Defs.lean and Foundations/GeneralWeights.lean.
-/

import HEAProofs.Foundations.Defs
import Mathlib.Algebra.Order.Chebyshev

open Real Finset BigOperators

noncomputable section

variable {J : ℕ}

-- ============================================================
-- Section 1: Simplex
-- ============================================================

/-- The composition simplex: all components non-negative and sum to 1. -/
def OnSimplex (J : ℕ) (p : Fin J → ℝ) : Prop :=
  (∀ j, 0 ≤ p j) ∧ ∑ j : Fin J, p j = 1

/-- The open simplex: all components strictly positive and sum to 1. -/
def OnOpenSimplex (J : ℕ) (p : Fin J → ℝ) : Prop :=
  (∀ j, 0 < p j) ∧ ∑ j : Fin J, p j = 1

/-- The open simplex implies the simplex. -/
theorem OnOpenSimplex.toSimplex {p : Fin J → ℝ} (hp : OnOpenSimplex J p) :
    OnSimplex J p :=
  ⟨fun j => le_of_lt (hp.1 j), hp.2⟩

/-- The uniform distribution is on the open simplex. -/
theorem uniform_onOpenSimplex (hJ : 0 < J) :
    OnOpenSimplex J (fun _ : Fin J => (1 : ℝ) / ↑J) := by
  constructor
  · intro _
    exact div_pos one_pos (Nat.cast_pos.mpr hJ)
  · rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    field_simp

-- ============================================================
-- Section 2: Herfindahl Index
-- ============================================================

/-- The Herfindahl index: H = Σ xⱼ². Measures composition concentration.
    H = 1/J at equimolar; H = 1 for a pure element. -/
def herfindahlIndex (J : ℕ) (x : Fin J → ℝ) : ℝ :=
  ∑ j : Fin J, x j ^ 2

/-- General-weight curvature: K(q, x) = (1 - q)(1 - H).
    For composition x with Σ xⱼ = 1:
    At equimolar xⱼ = 1/J: reduces to curvatureK J q. -/
def generalCurvatureK (J : ℕ) (q : ℝ) (x : Fin J → ℝ) : ℝ :=
  (1 - q) * (1 - ∑ j : Fin J, x j ^ 2)

/-- At equimolar, general curvature reduces to standard curvature. -/
theorem K_reduction_equimolar (hJ : 0 < J) {q : ℝ} :
    generalCurvatureK J q (fun _ : Fin J => (1 / ↑J : ℝ)) = curvatureK J q := by
  simp only [generalCurvatureK, curvatureK]
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hJne : (↑J : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  field_simp

/-- The Herfindahl index at equimolar is 1/J. -/
theorem herfindahl_equimolar (hJ : 0 < J) :
    herfindahlIndex J (fun _ : Fin J => (1 / ↑J : ℝ)) = 1 / ↑J := by
  simp only [herfindahlIndex]
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hJne : (↑J : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  field_simp

/-- Higher Herfindahl → lower curvature. -/
theorem K_decreasing_in_herfindahl
    {J : ℕ} {q : ℝ} (hq : q < 1)
    {x₁ x₂ : Fin J → ℝ}
    (hH : herfindahlIndex J x₁ < herfindahlIndex J x₂) :
    generalCurvatureK J q x₂ < generalCurvatureK J q x₁ := by
  unfold generalCurvatureK herfindahlIndex at *
  have hq_pos : 0 < 1 - q := by linarith
  nlinarith

/-- Equal weights maximize K for given q and J.
    **Proof.** By Cauchy-Schwarz, H ≥ 1/J with equality at equimolar. -/
theorem equalWeights_maximize_K (_hJ : 2 ≤ J) {q : ℝ} (_hq : q < 1)
    {x : Fin J → ℝ} (_hx_pos : ∀ j, 0 < x j) (_hx_sum : ∑ j : Fin J, x j = 1) :
    generalCurvatureK J q x ≤ curvatureK J q := by
  simp only [generalCurvatureK, curvatureK]
  have hJne : (↑J : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  have hJpos : (0 : ℝ) < ↑J := by exact_mod_cast (by omega : 0 < J)
  have h1q : 0 < 1 - q := by linarith
  have cs := sq_sum_le_card_mul_sum_sq (s := Finset.univ) (f := x)
  rw [Finset.card_univ, Fintype.card_fin, _hx_sum] at cs
  simp only [one_pow] at cs
  have goal_rw : (1 - q) * (↑J - 1) / ↑J = (1 - q) * ((↑J - 1) / ↑J) := by ring
  rw [goal_rw]
  apply mul_le_mul_of_nonneg_left _ (le_of_lt h1q)
  rw [sub_div, div_self hJne]
  have h1J : 1 / (↑J : ℝ) ≤ ∑ j : Fin J, x j ^ 2 := by
    rw [div_le_iff₀ hJpos]
    linarith [mul_comm (↑J : ℝ) (∑ i : Fin J, x i ^ 2)]
  linarith

end
