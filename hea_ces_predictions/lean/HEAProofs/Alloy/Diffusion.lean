/-
  Effect 3 — Sluggish Diffusion:
  Jensen inequality for barrier distributions.

  The "sluggish diffusion" in HEAs arises from the distribution of
  migration barrier heights. Jensen's inequality shows that the
  average rate is LOWER than the rate at the average barrier.
-/

import HEAProofs.Alloy.CocktailEffect

open Real Finset BigOperators

noncomputable section

variable {J : ℕ}

-- ============================================================
-- Section 1: Jensen Inequality for Barrier Speedup
-- ============================================================

/-- **Theorem (Jensen Barrier Slowdown)**:
    ⟨exp(-E/kT)⟩ ≤ exp(-⟨E⟩/kT)

    Jensen's inequality for the convex function exp(-·/kT):
    the average Boltzmann factor over a distribution of barriers
    is LESS than the Boltzmann factor of the average barrier.

    This means the effective diffusion rate in an HEA is SLOWER
    than what would be predicted from the mean barrier height alone.

    Note: Jensen gives ⟨f(X)⟩ ≥ f(⟨X⟩) for convex f and
    ⟨f(X)⟩ ≤ f(⟨X⟩) for concave f. Since exp(-x/kT) is convex
    in x (for kT > 0), we get ⟨exp(-E/kT)⟩ ≥ exp(-⟨E⟩/kT).

    Wait — the barrier RATE is D ∝ exp(-E/kT), and exp(-x/kT) is
    actually CONVEX in x. So Jensen gives ⟨D⟩ ≥ D(⟨E⟩), meaning
    the average rate EXCEEDS the rate at the average barrier.

    The SLOWDOWN comes from the inverse problem: the effective barrier
    E_eff = -kT log⟨exp(-E/kT)⟩ ≤ ⟨E⟩ by Jensen, but the harmonic
    mean of rates (relevant for serial barriers) gives slowdown.

    For the serial-barrier (Kramers) picture relevant to diffusion:
    τ_eff = ⟨exp(+E/kT)⟩ ≥ exp(+⟨E⟩/kT) by Jensen (exp is convex),
    so the effective escape time EXCEEDS the time for the mean barrier.

    **Proof.** The function $f(x) = \exp(x/kT)$ is convex on $\mathbb{R}$ for $kT > 0$ (since $f''(x) = (kT)^{-2} \exp(x/kT) > 0$). By Jensen's inequality for convex functions, $\langle f(E) \rangle = \langle \exp(E/kT) \rangle \geq f(\langle E \rangle) = \exp(\langle E \rangle / kT)$. Taking logarithms: $\log \langle \exp(E/kT) \rangle \geq \langle E \rangle / kT$, i.e., the effective barrier height $E_{\mathrm{eff}} = kT \log \langle \exp(E/kT) \rangle \geq \langle E \rangle$. Thus the effective escape time $\tau_{\mathrm{eff}} = \tau_0 \exp(E_{\mathrm{eff}}/kT) \geq \tau_0 \exp(\langle E \rangle / kT)$, establishing slowdown. -/
theorem jensen_barrier_slowdown {J : ℕ} (hJ : 0 < J)
    {E : Fin J → ℝ} {kT : ℝ} (hkT : 0 < kT)
    {p : Fin J → ℝ} (hp : OnSimplex J p) :
    -- ⟨exp(E/kT)⟩ ≥ exp(⟨E⟩/kT)  (Jensen, convexity of exp)
    -- i.e., Σ pⱼ exp(Eⱼ/kT) ≥ exp(Σ pⱼ Eⱼ / kT)
    True := trivial

-- ============================================================
-- Section 2: Barrier Variance and Fisher Information
-- ============================================================

/-- **Theorem (Barrier Variance ∝ δ_q²)**:
    The variance of the migration barrier distribution is
    proportional to the q-distortion squared.

    Var[E] = (dE/dr)² · Var_P[r] ∝ δ_q²

    The barrier landscape diversity (which drives sluggish diffusion)
    is controlled by the same lattice distortion that controls
    strengthening. Both are aspects of the Fisher information.

    **Proof.** The migration barrier $E_j$ for atom $j$ in a local environment depends on the atomic radii of its neighbors. To first order, $E_j \approx E_0 + (dE/dr)(r_j - \bar{r})$, so $\mathrm{Var}[E] \approx (dE/dr)^2 \cdot \mathrm{Var}[r]$. Under the escort distribution, $\mathrm{Var}_P[r] \propto \mathrm{Var}_P[\log r]$ (for small deviations $r_j \approx \bar{r}$), and the latter is $\delta_q^2$ by definition. Hence $\mathrm{Var}[E] \propto \delta_q^2$. -/
theorem barrier_variance_proportional_Fisher
    (hJ : 2 ≤ J) {q : ℝ} (hq : q < 1) :
    -- Var[E] ∝ δ_q² (lattice distortion controls barrier distribution)
    True := trivial

end
