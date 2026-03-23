/-
  HEA-specific definitions: alloy elements, compositions,
  and derived quantities (atomic size mismatch δ, q parameter).
-/

import HEAProofs.Foundations.InformationGeometry

open Real Finset BigOperators

noncomputable section

variable {J : ℕ}

-- ============================================================
-- Section 1: Alloy Element Properties
-- ============================================================

/-- An alloy element with its physical properties.
    r: atomic radius (pm)
    mass: atomic mass (amu)
    T_m: melting temperature (K)
    VEC: valence electron concentration
    σ_a: atomic scattering cross-section
    χ: electronegativity (Pauling) -/
structure AlloyElement where
  r : ℝ      -- atomic radius
  mass : ℝ   -- atomic mass
  T_m : ℝ    -- melting temperature
  VEC : ℝ    -- valence electron concentration
  σ_a : ℝ    -- scattering cross-section
  χ : ℝ      -- electronegativity
  r_pos : 0 < r
  mass_pos : 0 < mass
  T_m_pos : 0 < T_m

-- ============================================================
-- Section 2: Alloy Composition
-- ============================================================

/-- An alloy composition: J elements with atomic fractions on the simplex. -/
structure AlloyComposition (J : ℕ) where
  elements : Fin J → AlloyElement
  fractions : Fin J → ℝ
  on_simplex : OnSimplex J fractions

/-- Mean atomic radius: r̄ = Σ cⱼ rⱼ. -/
def meanRadius (comp : AlloyComposition J) : ℝ :=
  ∑ j : Fin J, comp.fractions j * (comp.elements j).r

/-- Mean atomic mass: m̄ = Σ cⱼ mⱼ. -/
def meanMass (comp : AlloyComposition J) : ℝ :=
  ∑ j : Fin J, comp.fractions j * (comp.elements j).mass

/-- Mean melting temperature: T̄_m = Σ cⱼ T_m,j. -/
def meanMeltingTemp (comp : AlloyComposition J) : ℝ :=
  ∑ j : Fin J, comp.fractions j * (comp.elements j).T_m

/-- Mean electronegativity: χ̄ = Σ cⱼ χⱼ. -/
def meanElectronegativity (comp : AlloyComposition J) : ℝ :=
  ∑ j : Fin J, comp.fractions j * (comp.elements j).χ

/-- Mean VEC: VĒC = Σ cⱼ VECⱼ. -/
def meanVEC (comp : AlloyComposition J) : ℝ :=
  ∑ j : Fin J, comp.fractions j * (comp.elements j).VEC

-- ============================================================
-- Section 3: Atomic Size Mismatch δ
-- ============================================================

/-- Atomic size mismatch parameter:
    δ = √(Σ cⱼ (1 - rⱼ/r̄)²).
    This is the standard Yang-Zhang parameter measuring lattice
    distortion from atomic size differences. -/
def atomicSizeMismatch (comp : AlloyComposition J) (hmr : meanRadius comp ≠ 0) : ℝ :=
  Real.sqrt (∑ j : Fin J, comp.fractions j *
    (1 - (comp.elements j).r / meanRadius comp) ^ 2)

/-- Electronegativity difference: Δχ = √(Σ cⱼ (χⱼ - χ̄)²). -/
def electronegativityDiff (comp : AlloyComposition J) : ℝ :=
  Real.sqrt (∑ j : Fin J, comp.fractions j *
    ((comp.elements j).χ - meanElectronegativity comp) ^ 2)

/-- VEC difference: ΔVEC = √(Σ cⱼ (VECⱼ - VĒC)²). -/
def vecDiff (comp : AlloyComposition J) : ℝ :=
  Real.sqrt (∑ j : Fin J, comp.fractions j *
    ((comp.elements j).VEC - meanVEC comp) ^ 2)

-- ============================================================
-- Section 4: q Parameter from Physical Properties
-- ============================================================

/-- The complementarity parameter q derived from physical mismatch:
    q ≈ 1 - α·δ² - β·(Δχ)² - γ·(ΔVEC)²

    The parameter q encodes how far the alloy departs from ideal
    (rule-of-mixtures) behavior. q = 1 is ideal; q < 1 means
    synergistic interactions (complementarity).

    The coefficients α, β, γ are material-class-dependent and
    determined from experimental calibration. -/
def qFromMismatch (δ Δχ ΔVEC α β γ_coeff : ℝ) : ℝ :=
  1 - α * δ ^ 2 - β * Δχ ^ 2 - γ_coeff * ΔVEC ^ 2

/-- q < 1 when mismatch parameters are positive and coefficients non-negative. -/
theorem qFromMismatch_lt_one {δ Δχ ΔVEC α β γ_coeff : ℝ}
    (hα : 0 < α) (hδ : 0 < δ) (hβ : 0 ≤ β) (hγ : 0 ≤ γ_coeff) :
    qFromMismatch δ Δχ ΔVEC α β γ_coeff < 1 := by
  simp only [qFromMismatch]
  have h1 : 0 < α * δ ^ 2 := mul_pos hα (sq_pos_of_pos hδ)
  have h2 : 0 ≤ β * Δχ ^ 2 := mul_nonneg hβ (sq_nonneg _)
  have h3 : 0 ≤ γ_coeff * ΔVEC ^ 2 := mul_nonneg hγ (sq_nonneg _)
  linarith

-- ============================================================
-- Section 5: Yang-Zhang Stability Criterion
-- ============================================================

/-- Yang-Zhang stability parameter:
    Ω = T_m · ΔS_mix / |ΔH_mix|
    Measures the balance between entropy-driven stabilization
    and enthalpy-driven phase separation. -/
def yangZhangOmega (T_m ΔS_mix ΔH_mix : ℝ) (hΔH : ΔH_mix ≠ 0) : ℝ :=
  T_m * ΔS_mix / |ΔH_mix|

/-- Yang-Zhang stability criterion: Ω ≥ 1.1 and δ ≤ 6.6%.
    Empirical rule for predicting solid-solution formation in HEAs. -/
def yangZhangStable (Ω δ : ℝ) : Prop :=
  1.1 ≤ Ω ∧ δ ≤ 0.066

end
