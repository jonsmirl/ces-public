/-
  Alloy Partition Function: Z_q = cesFun for alloys,
  aggregate property F, and escort weights P_j.
-/

import HEAProofs.Alloy.Defs
import HEAProofs.Foundations.QThermodynamics

open Real Finset BigOperators

noncomputable section

variable {J : ℕ}

-- ============================================================
-- Section 1: Alloy Aggregate Property
-- ============================================================

/-- Alloy aggregate property via CES partition function:
    F = Z_q^{1/q} = (Σ aⱼ xⱼ^q)^{1/q}
    where aⱼ are intrinsic property weights and xⱼ are atomic fractions.

    When q = 1: F = Σ aⱼ xⱼ (rule of mixtures).
    When q < 1: F > Σ aⱼ xⱼ (synergistic, cocktail effect). -/
def alloyAggregate (J : ℕ) (a : Fin J → ℝ) (q : ℝ) (x : Fin J → ℝ) : ℝ :=
  cesFun J a q x

/-- Escort weight of element j:
    P_j = a_j x_j^q / Z_q
    Gives the effective contribution of element j to the aggregate
    property, which differs from the atomic fraction when q ≠ 1. -/
def escortWeight (J : ℕ) (a : Fin J → ℝ) (q : ℝ) (x : Fin J → ℝ)
    (j : Fin J) : ℝ :=
  shareFunction (fun k => a k * (x k) ^ q) j

/-- Escort weights sum to 1 (when partition function is nonzero). -/
theorem escortWeight_sum_one {a : Fin J → ℝ} {q : ℝ} {x : Fin J → ℝ}
    (h : (∑ k : Fin J, a k * (x k) ^ q) ≠ 0) :
    ∑ j : Fin J, escortWeight J a q x j = 1 :=
  shareFunction_sum_one h

/-- Escort weights are non-negative for non-negative inputs. -/
theorem escortWeight_nonneg {a : Fin J → ℝ} {q : ℝ} {x : Fin J → ℝ}
    (ha : ∀ j, 0 ≤ a j) (hx : ∀ j, 0 ≤ (x j) ^ q) (j : Fin J) :
    0 ≤ escortWeight J a q x j :=
  shareFunction_nonneg (fun k => mul_nonneg (ha k) (hx k)) j

-- ============================================================
-- Section 2: q-Distortion (Lattice Distortion Measure)
-- ============================================================

/-- q-distortion: δ_q = √(Var_P[log r])
    The lattice distortion measured under the escort distribution,
    not the atomic fractions. This is the Fisher-information-derived
    measure of lattice strain.

    When q = 1: reduces to standard δ.
    When q < 1: weights extreme-radius elements more heavily. -/
def qDistortion (J : ℕ) (a : Fin J → ℝ) (q : ℝ) (x : Fin J → ℝ)
    (r : Fin J → ℝ) : ℝ :=
  let P := fun j => escortWeight J a q x j
  let logr := fun j => Real.log (r j)
  let mean_logr := ∑ j : Fin J, P j * logr j
  Real.sqrt (∑ j : Fin J, P j * (logr j - mean_logr) ^ 2)

-- ============================================================
-- Section 3: q-Entropy for Alloy
-- ============================================================

/-- q-entropy for the alloy composition (wraps tsallisEntropy).
    Measures the effective configurational entropy with
    q-deformation encoding non-ideal mixing. -/
def qEntropy (J : ℕ) (q : ℝ) (x : Fin J → ℝ) : ℝ :=
  tsallisEntropy J q x

end
