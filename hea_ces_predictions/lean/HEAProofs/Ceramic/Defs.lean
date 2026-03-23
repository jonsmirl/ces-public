/-
  High-Entropy Ceramic Definitions:
  Ceramic compositions, A-site/B-site sublattices, ionic radii.

  Models pyrochlore/fluorite A₂B₂O₇ ceramics with rare-earth
  A-site and Zr/Hf B-site elements for thermal protection systems.
-/

import HEAProofs.Alloy.Predictions

open Real Finset BigOperators

noncomputable section

-- ============================================================
-- Section 1: Ceramic Element Properties
-- ============================================================

/-- A ceramic ion with its physical properties for phonon scattering.
    r_ion: Shannon ionic radius (pm)
    mass: atomic mass (amu)
    label: which sublattice (A or B) -/
structure CeramicIon where
  r_ion : ℝ   -- Shannon ionic radius
  mass : ℝ    -- atomic mass
  r_pos : 0 < r_ion
  mass_pos : 0 < mass

-- ============================================================
-- Section 2: Dual-Sublattice Composition
-- ============================================================

/-- A dual-sublattice ceramic composition (A₂B₂O₇ pyrochlore).
    J_A: number of A-site species (e.g., La, Gd, Y)
    J_B: number of B-site species (e.g., Zr, Hf)
    f_A, f_B: sublattice weight fractions for phonon scattering -/
structure CeramicComposition (J_A J_B : ℕ) where
  A_ions : Fin J_A → CeramicIon
  B_ions : Fin J_B → CeramicIon
  A_fractions : Fin J_A → ℝ
  B_fractions : Fin J_B → ℝ
  A_on_simplex : OnSimplex J_A A_fractions
  B_on_simplex : OnSimplex J_B B_fractions
  f_A : ℝ  -- A-sublattice weight fraction for scattering
  f_B : ℝ  -- B-sublattice weight fraction for scattering
  f_sum : f_A + f_B = 1
  f_A_pos : 0 < f_A
  f_B_pos : 0 < f_B

/-- Mean A-site mass: m̄_A = Σ cⱼ mⱼ. -/
def meanAMass {J_A J_B : ℕ} (comp : CeramicComposition J_A J_B) : ℝ :=
  ∑ j : Fin J_A, comp.A_fractions j * (comp.A_ions j).mass

/-- Mean B-site mass: m̄_B = Σ cⱼ mⱼ. -/
def meanBMass {J_A J_B : ℕ} (comp : CeramicComposition J_A J_B) : ℝ :=
  ∑ j : Fin J_B, comp.B_fractions j * (comp.B_ions j).mass

end
