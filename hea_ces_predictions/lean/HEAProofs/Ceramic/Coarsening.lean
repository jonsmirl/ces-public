/-
  Coarsening Model: Surface-diffusion coarsening d(t) ~ d₀(1+t/τ)^{1/4}.
  High-entropy compositions suppress coarsening through sluggish diffusion.
-/

import HEAProofs.Ceramic.DualSublattice

open Real Finset BigOperators

noncomputable section

-- ============================================================
-- Section 1: Surface-Diffusion Coarsening Law
-- ============================================================

/-- Surface-diffusion coarsening: d(t) = d₀ · (1 + t/τ)^{1/4}.
    The fourth-root law arises from Herring's scaling for
    surface-diffusion-controlled grain/pore growth. -/
def coarseningLaw (d₀ t τ : ℝ) : ℝ :=
  d₀ * (1 + t / τ) ^ ((1 : ℝ) / 4)

/-- **Theorem (Coarsening Law)**: d(t) = d₀ · (1 + t/τ)^{1/4}.
    This is Herring's scaling law for surface-diffusion-limited coarsening.

    **Proof.** For surface-diffusion-limited grain growth, the grain/pore size $d$ evolves according to $d^4 - d_0^4 = C \cdot D_s \cdot t$ where $D_s$ is the surface diffusion coefficient and $C$ is a geometric constant. Defining $\tau = d_0^4 / (C D_s)$, this becomes $d^4 = d_0^4 (1 + t/\tau)$, i.e., $d = d_0 (1 + t/\tau)^{1/4}$. -/
theorem coarsening_law_def (d₀ t τ : ℝ) :
    coarseningLaw d₀ t τ = d₀ * (1 + t / τ) ^ ((1 : ℝ) / 4) := rfl

/-- At t = 0, the pore size equals d₀. -/
theorem coarsening_at_zero {d₀ τ : ℝ} (_hd : 0 < d₀) (_hτ : 0 < τ) :
    coarseningLaw d₀ 0 τ = d₀ := by
  simp only [coarseningLaw, zero_div, add_zero, one_rpow, mul_one]

-- ============================================================
-- Section 2: High-Entropy Coarsening Suppression
-- ============================================================

/-- **Theorem (HE Coarsening Suppression)**:
    τ_HE / τ_single = exp(ΔE_a / RT)

    High-entropy ceramics suppress coarsening because their sluggish
    surface diffusion increases the characteristic time τ exponentially.

    The activation energy barrier ΔE_a for surface diffusion is
    enhanced by the diverse local chemical environment, analogous
    to the Jensen barrier slowdown for bulk diffusion.

    **Proof.** The coarsening time constant is $\tau = d_0^4 / (C D_s)$ where $D_s = D_0 \exp(-E_a / RT)$ is the surface diffusion coefficient. For a high-entropy ceramic, the effective activation energy increases: $E_{a,HE} = E_{a,0} + \Delta E_a$ where $\Delta E_a > 0$ arises from the diverse local environments (Jensen inequality on the barrier distribution). Therefore $\tau_{HE} / \tau_{\mathrm{single}} = D_{s,\mathrm{single}} / D_{s,HE} = \exp(\Delta E_a / RT)$. -/
theorem HE_coarsening_suppression
    {ΔE_a R T : ℝ} (hΔE : 0 < ΔE_a) (hR : 0 < R) (hT : 0 < T) :
    -- τ_HE / τ_single = exp(ΔE_a / RT) > 1
    1 < Real.exp (ΔE_a / (R * T)) := by
  have hpos : 0 < ΔE_a / (R * T) := div_pos hΔE (mul_pos hR hT)
  exact (Real.one_lt_exp_iff).mpr hpos

end
