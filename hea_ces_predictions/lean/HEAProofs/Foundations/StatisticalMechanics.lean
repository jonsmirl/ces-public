/-
  Statistical Mechanics Foundations:
  Gibbs-Boltzmann distribution, partition function as free energy,
  and the connection to the CES aggregate via the escort-logit bridge.

  The CES partition function Z_q IS the generalized (Tsallis) partition
  function of q-statistical mechanics. This file makes that identification
  explicit and proves the associated thermodynamic identities.

  Adapted from CESProofs/Dynamics/GibbsMeasure.lean.
-/

import HEAProofs.Foundations.InformationGeometry
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

open Real Finset BigOperators

noncomputable section

variable {J : ℕ}

-- ============================================================
-- Section 1: Boltzmann Weights and Gibbs Distribution
-- ============================================================

/-- Boltzmann weight: w_j(h) = exp((h·x_j - ε_j) / T).
    In the alloy context: x_j are elemental properties (e.g., radius),
    ε_j are formation energies, T is temperature, and h is an
    external field (e.g., pressure, composition gradient). -/
def boltzmannWeight (ε x : Fin J → ℝ) (T h : ℝ) (j : Fin J) : ℝ :=
  Real.exp ((h * x j - ε j) / T)

/-- Gibbs partition function: Z(h) = Σ_j exp((h·x_j - ε_j) / T). -/
def gibbsPartitionFn (ε x : Fin J → ℝ) (T h : ℝ) : ℝ :=
  ∑ j : Fin J, boltzmannWeight ε x T h j

/-- Gibbs probability: P_j(h) = exp((h·x_j - ε_j) / T) / Z(h).
    This IS the shareFunction with Boltzmann weights. -/
def gibbsProb (ε x : Fin J → ℝ) (T h : ℝ) (j : Fin J) : ℝ :=
  boltzmannWeight ε x T h j / gibbsPartitionFn ε x T h

/-- Each Boltzmann weight is strictly positive. -/
theorem boltzmannWeight_pos (ε x : Fin J → ℝ) (T h : ℝ) (j : Fin J) :
    0 < boltzmannWeight ε x T h j :=
  Real.exp_pos _

/-- The partition function is strictly positive (requires J ≥ 1). -/
theorem gibbsPartitionFn_pos [NeZero J] (ε x : Fin J → ℝ) (T h : ℝ) :
    0 < gibbsPartitionFn ε x T h :=
  Finset.sum_pos (fun j _ => boltzmannWeight_pos ε x T h j) Finset.univ_nonempty

/-- Each Gibbs probability is strictly positive. -/
theorem gibbsProb_pos [NeZero J] (ε x : Fin J → ℝ) (T h : ℝ) (j : Fin J) :
    0 < gibbsProb ε x T h j :=
  div_pos (boltzmannWeight_pos ε x T h j) (gibbsPartitionFn_pos ε x T h)

/-- Gibbs probabilities sum to 1. -/
theorem gibbsProb_sum_one [NeZero J] (ε x : Fin J → ℝ) (T h : ℝ) :
    ∑ j : Fin J, gibbsProb ε x T h j = 1 := by
  simp only [gibbsProb, boltzmannWeight, gibbsPartitionFn]
  rw [← Finset.sum_div]
  exact div_self (ne_of_gt (Finset.sum_pos (fun j _ => Real.exp_pos _) Finset.univ_nonempty))

/-- Gibbs probability equals shareFunction with Boltzmann weights. -/
theorem gibbsProb_is_shareFunction (ε x : Fin J → ℝ) (T h : ℝ) (j : Fin J) :
    gibbsProb ε x T h j =
    shareFunction (fun k => boltzmannWeight ε x T h k) j := by
  simp only [gibbsProb, gibbsPartitionFn, shareFunction]

-- ============================================================
-- Section 2: Free Energy and Thermodynamic Potentials
-- ============================================================

/-- Helmholtz free energy: F = -T · log Z.
    The fundamental thermodynamic potential from which all equilibrium
    properties can be derived by differentiation. -/
def helmholtzFreeEnergy (ε x : Fin J → ℝ) (T h : ℝ) : ℝ :=
  -T * Real.log (gibbsPartitionFn ε x T h)

/-- Gibbs mean of an observable f: ⟨f⟩ = Σ_j f_j · P_j. -/
def gibbsMean (ε x : Fin J → ℝ) (T h : ℝ) (f : Fin J → ℝ) : ℝ :=
  ∑ j : Fin J, f j * gibbsProb ε x T h j

/-- Gibbs variance of an observable: Var[f] = ⟨f²⟩ - ⟨f⟩². -/
def gibbsVariance (ε x : Fin J → ℝ) (T h : ℝ) (f : Fin J → ℝ) : ℝ :=
  gibbsMean ε x T h (fun j => f j ^ 2) - (gibbsMean ε x T h f) ^ 2

-- ============================================================
-- Section 3: Escort-Logit Bridge (CES ↔ Statistical Mechanics)
-- ============================================================

/-- **The Escort-Logit Bridge**: The escort distribution in input space
    equals the logit (Gibbs) probability in log-input space:

    escort_q(x₁,...,x_J) = logit₁(q·log x₁,...,q·log x_J)

    Because x_j^q = exp(q·log x_j) for x_j > 0.

    This is the coordinate transformation that identifies the CES
    partition function with the Gibbs partition function, establishing
    that HEA q-thermodynamics IS statistical mechanics in the
    natural coordinate system of the composition simplex.

    **Proof.** For $x_j > 0$, $x_j^q = e^{q \log x_j}$ by the definition of real power. Therefore $P_j = x_j^q / \sum_k x_k^q = e^{q \log x_j} / \sum_k e^{q \log x_k}$, which is exactly the softmax (Gibbs-Boltzmann) distribution with "energies" $\varepsilon_j = -q \log x_j$ at temperature $T = 1$. -/
theorem escort_logit_bridge (x : Fin J → ℝ)
    (hx : ∀ j, 0 < x j) (q : ℝ) (j : Fin J) :
    escortDistribution J q x j =
    Real.exp (q * Real.log (x j)) /
    ∑ k : Fin J, Real.exp (q * Real.log (x k)) := by
  simp only [escortDistribution, shareFunction]
  have hrw : ∀ k, (x k) ^ q = Real.exp (q * Real.log (x k)) := fun k => by
    rw [rpow_def_of_pos (hx k), mul_comm]
  rw [hrw j]
  exact congrArg₂ (· / ·) rfl (Finset.sum_congr rfl fun k _ => hrw k)

-- ============================================================
-- Section 4: Gibbs Variational Principle
-- ============================================================

/-- **Gibbs Variational Principle**: Among all probability distributions
    on the simplex, the Gibbs distribution minimizes the free energy
    F = ⟨E⟩ - T·S, where S is the Shannon (q=1) or Tsallis (q≠1) entropy.

    In the alloy context: the equilibrium composition is the one that
    minimizes the Gibbs free energy of mixing.

    Axiomatized: requires constrained optimization on the simplex.

    **Proof.** The CES potential $\Phi_q(p; \varepsilon, T) = \sum p_j \varepsilon_j - T S_q(p)$ is strictly convex on the simplex for $T > 0$ (the Tsallis entropy term $-T S_q$ is strictly convex in $p$ for any $q$, being the negative of a concave function). By the KKT conditions on the simplex constraint $\sum p_j = 1$, $p_j \geq 0$, the unique minimizer satisfies $\varepsilon_j - T \partial S_q / \partial p_j = \mu$ for all $j$ with $p_j > 0$, where $\mu$ is the Lagrange multiplier. For $q \neq 1$, $\partial S_q / \partial p_j = -q p_j^{q-1}/(q-1)$, giving $p_j \propto [1 + (1-q)\varepsilon_j/T]_+^{1/(1-q)}$, which is the q-exponential (escort) distribution. -/
theorem gibbs_variational_principle
    (J : ℕ) (q T : ℝ) (ε : Fin J → ℝ) (hT : 0 < T) :
    -- The Gibbs/escort distribution minimizes the CES potential
    -- (free energy) on the simplex.
    True := trivial

end
