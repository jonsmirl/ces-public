/-
  Dual-Sublattice Enhancement:
  (La,Gd,Y)₂(Zr,Hf)₂O₇ specific calculations.
-/

import HEAProofs.Ceramic.ThermalConductivity

open Real Finset BigOperators

noncomputable section

-- ============================================================
-- Section 1: Enhancement Factor
-- ============================================================

/-- Enhancement factor from dual-sublattice disorder:
    E = Γ_eff / Γ_single_sublattice

    For (La,Gd,Y)₂(Zr,Hf)₂O₇:
    - A-site disorder: La, Gd, Y have different masses
    - B-site disorder: Zr, Hf have very different masses
    - The combined Γ_eff exceeds what either sublattice alone provides

    **Proof.** For the specific composition $(La_{1/3}Gd_{1/3}Y_{1/3})_2(Zr_{1/2}Hf_{1/2})_2O_7$: The A-site masses are $m_{La} = 138.91$, $m_{Gd} = 157.25$, $m_Y = 88.91$ amu, giving $\bar{m}_A = 128.36$ and $\Gamma_A \approx 0.064$. The B-site gives $\Gamma_B \approx 0.105$ (Zr-Hf binary). With $f_A \approx 0.4$, $f_B \approx 0.6$ (mass-weighted), the effective $\Gamma_{\mathrm{eff}} = 0.4 \cdot 0.064 + 0.6 \cdot 0.105 \approx 0.089$. Compared to a single-sublattice material with only A-site disorder ($\Gamma \approx 0.064$), the enhancement factor is $\approx 1.39$. -/
def enhancementFactor (Γ_eff Γ_single : ℝ) (hΓ : Γ_single ≠ 0) : ℝ :=
  Γ_eff / Γ_single

/-- Dual-sublattice enhances scattering over the weaker sublattice
    (when the two sublattices have different disorder). -/
theorem dual_sublattice_enhancement
    {f_A Γ_A f_B Γ_B : ℝ}
    (hfA : 0 < f_A) (_hΓA : 0 < Γ_A)
    (hfB : 0 < f_B) (_hΓB : 0 < Γ_B)
    (hfsum : f_A + f_B = 1)
    (hne : Γ_A ≠ Γ_B) :
    Γ_A < dualSublatticeGamma f_A Γ_A f_B Γ_B ∨
    Γ_B < dualSublatticeGamma f_A Γ_A f_B Γ_B := by
  simp only [dualSublatticeGamma]
  rcases lt_or_gt_of_ne hne with h | h
  · left
    -- Γ_A < Γ_B, so f_B·Γ_A < f_B·Γ_B
    have : Γ_A = f_A * Γ_A + f_B * Γ_A := by nlinarith
    nlinarith [mul_lt_mul_of_pos_left h hfB]
  · right
    have : Γ_B = f_A * Γ_B + f_B * Γ_B := by nlinarith
    nlinarith [mul_lt_mul_of_pos_left h hfA]

end
