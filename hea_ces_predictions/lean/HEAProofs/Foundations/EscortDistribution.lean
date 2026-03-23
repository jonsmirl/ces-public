/-
  The Universal Share Function (Escort Distribution).
  Adapted from CESProofs/Foundations/TenWayIdentity.lean.

  In the HEA context, the escort distribution P_j = a_j x_j^q / Z_q
  gives the effective property contribution of element j, which differs
  from the atomic fraction x_j when q ≠ 1.
-/

import HEAProofs.Foundations.Simplex
import Mathlib.Algebra.BigOperators.Field

open Real Finset BigOperators

noncomputable section

variable {J : ℕ}

-- ============================================================
-- Section 1: The Universal Share Function
-- ============================================================

/-- The universal share function: given a weight vector w,
    the share of component j is w_j / Σ_k w_k.

    This is simultaneously the CES factor share, the Tsallis escort
    distribution, the softmax/logit probability, the Gibbs-Boltzmann
    distribution, and the Luce choice probability — all are the same
    mathematical object (proved in CESProofs). -/
def shareFunction (w : Fin J → ℝ) (j : Fin J) : ℝ :=
  w j / ∑ k : Fin J, w k

-- ============================================================
-- Section 2: Universal Properties
-- ============================================================

/-- Shares sum to 1 when the total weight is nonzero. -/
theorem shareFunction_sum_one {w : Fin J → ℝ}
    (hw : (∑ k : Fin J, w k) ≠ 0) :
    ∑ j : Fin J, shareFunction w j = 1 := by
  simp only [shareFunction]
  rw [← Finset.sum_div]
  exact div_self hw

/-- Each share is non-negative when all weights are non-negative. -/
theorem shareFunction_nonneg {w : Fin J → ℝ}
    (hw : ∀ j, 0 ≤ w j) (j : Fin J) :
    0 ≤ shareFunction w j :=
  div_nonneg (hw j) (Finset.sum_nonneg fun k _ => hw k)

/-- IIA (Independence of Irrelevant Alternatives):
    The ratio of any two shares depends only on the ratio of their
    weights, not on any other component's weight. -/
theorem shareFunction_iia [NeZero J] {w : Fin J → ℝ}
    (hw : ∀ j, 0 < w j) (j k : Fin J) :
    shareFunction w j / shareFunction w k = w j / w k := by
  simp only [shareFunction]
  have hsum : (0 : ℝ) < ∑ i, w i :=
    Finset.sum_pos (fun i _ => hw i) Finset.univ_nonempty
  field_simp [ne_of_gt hsum, ne_of_gt (hw k)]

/-- Scale invariance: scaling all weights by the same constant
    does not change shares. -/
theorem shareFunction_scale_invariant {w : Fin J → ℝ}
    {c : ℝ} (hc : c ≠ 0) (j : Fin J) :
    shareFunction (fun k => c * w k) j = shareFunction w j := by
  simp only [shareFunction, ← Finset.mul_sum]
  exact mul_div_mul_left (w j) (∑ k, w k) hc

/-- At equal weights, every share equals 1/J. -/
theorem shareFunction_uniform_at_symmetry {J : ℕ} (hJ : 0 < J)
    {c : ℝ} (hc : c ≠ 0) (j : Fin J) :
    shareFunction (fun _ : Fin J => c) j = 1 / ↑J := by
  simp only [shareFunction, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul]
  have hJne : (↑J : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  field_simp

-- ============================================================
-- Section 3: Escort Distribution (HEA specialization)
-- ============================================================

/-- The escort distribution: P_j = x_j^q / Σ_k x_k^q.
    In the HEA context, this gives the effective property share
    of element j, deformed by the complementarity parameter q. -/
def escortDistribution (J : ℕ) (q : ℝ) (x : Fin J → ℝ) (j : Fin J) : ℝ :=
  shareFunction (fun k => (x k) ^ q) j

/-- The escort distribution is a specialization of shareFunction. -/
theorem escortDistribution_is_shareFunction (q : ℝ)
    (x : Fin J → ℝ) (j : Fin J) :
    escortDistribution J q x j = shareFunction (fun k => (x k) ^ q) j :=
  rfl

/-- Escort distribution sums to 1 (when well-defined). -/
theorem escortDistribution_sum_one {q : ℝ} {x : Fin J → ℝ}
    (h : (∑ k : Fin J, (x k) ^ q) ≠ 0) :
    ∑ j : Fin J, escortDistribution J q x j = 1 :=
  shareFunction_sum_one h

/-- Escort distribution is non-negative for non-negative inputs. -/
theorem escortDistribution_nonneg {q : ℝ} {x : Fin J → ℝ}
    (hx : ∀ j, 0 ≤ (x j) ^ q) (j : Fin J) :
    0 ≤ escortDistribution J q x j :=
  shareFunction_nonneg hx j

/-- IIA for escort distribution. -/
theorem escortDistribution_iia [NeZero J] {q : ℝ} {x : Fin J → ℝ}
    (hx : ∀ j, 0 < (x j) ^ q) (j k : Fin J) :
    escortDistribution J q x j / escortDistribution J q x k =
    (x j) ^ q / (x k) ^ q :=
  shareFunction_iia hx j k

end
