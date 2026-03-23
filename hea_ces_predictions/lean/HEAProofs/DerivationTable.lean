/-
  Derivation Table: Catalog of all theorems in the HEA proof tree.

  This file imports every module and provides #check statements
  for all major theorems, organized by tier and topic.
-/

import HEAProofs.Foundations.Defs
import HEAProofs.Foundations.Simplex
import HEAProofs.Foundations.QThermodynamics
import HEAProofs.Foundations.EscortDistribution
import HEAProofs.Foundations.InformationGeometry
import HEAProofs.Foundations.StatisticalMechanics
import HEAProofs.Foundations.FluctuationResponse
import HEAProofs.Foundations.EntropyInequalities
import HEAProofs.Alloy.Defs
import HEAProofs.Alloy.PartitionFunction
import HEAProofs.Alloy.Curvature
import HEAProofs.Alloy.EffectiveCurvature
import HEAProofs.Alloy.PhaseStability
import HEAProofs.Alloy.LatticeDistortion
import HEAProofs.Alloy.CocktailEffect
import HEAProofs.Alloy.Diffusion
import HEAProofs.Alloy.RadiationTolerance
import HEAProofs.Alloy.PerElementSurplus
import HEAProofs.Alloy.Predictions
import HEAProofs.Alloy.PhaseTransition
import HEAProofs.Alloy.QEquilibrium
import HEAProofs.Alloy.KramersEscape
import HEAProofs.Alloy.SharperPredictions
import HEAProofs.Ceramic.Defs
import HEAProofs.Ceramic.PhononScattering
import HEAProofs.Ceramic.ThermalConductivity
import HEAProofs.Ceramic.DualSublattice
import HEAProofs.Ceramic.Coarsening

/-! # Derivation Table

  Complete catalog of theorems in the HEA q-thermodynamics proof tree,
  organized by dependency tier.

  ## Tier 0: Pure Algebra (Foundations, adapted from CESProofs)
-/

-- Definitions
#check cesFun
#check powerMean
#check curvatureK
#check curvatureKH
#check symmetricPoint

-- Basic CES properties
#check powerMean_symmetricPoint
#check curvatureK_pos
#check curvatureK_eq_zero_of_q_one
#check curvatureK_eq_curvatureKH

-- Simplex and Herfindahl
#check OnSimplex
#check OnOpenSimplex
#check herfindahlIndex
#check generalCurvatureK
#check K_reduction_equimolar
#check herfindahl_equimolar
#check K_decreasing_in_herfindahl
#check equalWeights_maximize_K

-- q-Thermodynamics
#check qLog
#check qExp
#check qLog_one
#check qExp_zero
#check tsallisEntropy
#check tsallisEntropy_uniform

-- Escort Distribution (Universal Share Function)
#check shareFunction
#check shareFunction_sum_one
#check shareFunction_nonneg
#check shareFunction_iia
#check shareFunction_scale_invariant
#check escortDistribution
#check escortDistribution_sum_one

-- Information Geometry
#check cesHessianQF
#check cesEigenvaluePerp
#check cesHessianQF_on_perp
#check cesHessianQF_neg_semidef
#check bridge_theorem
#check curvatureK_eq_bridge_times_fisher
#check VRI_identity

/-! ## Tier 1: HEA Definitions -/

#check AlloyElement
#check AlloyComposition
#check atomicSizeMismatch
#check qFromMismatch
#check alloyAggregate
#check escortWeight
#check qDistortion
#check qEntropy
#check effectiveOrdering
#check yangZhangStable

/-! ## Tier 2: Core Effect Theorems -/

-- Effect 1: Phase Stabilization
#check qEntropy_gt_shannon
#check freeEnergy_q_lt_standard
#check yangZhang_from_Keff
#check omega_condition_from_enthalpy

-- Effect 2: Lattice Distortion
#check VRI_alloy
#check qDistortion_zero_pure
#check strengthening_proportional_K_Fisher
#check VLGC_equimolar_agreement

-- Effect 3: Diffusion
#check jensen_barrier_slowdown
#check barrier_variance_proportional_Fisher

-- Effect 4: Cocktail Effect
#check cocktail_resistance_positive
#check cocktail_transport_negative

-- Curvature and Effective Curvature
#check effectiveCurvatureKeff
#check effectiveCurvatureKeff_zero_temp
#check effectiveCurvatureKeff_above_critical
#check effectiveCurvatureKeff_nonneg
#check effectiveCurvatureKeff_le_K
#check effectiveCurvatureKeff_pos
#check K_squared_degrades_faster

-- Per-Element Surplus
#check perElementSurplus
#check perElementSurplus_peaks_at_2
#check perElementSurplus_value_at_2

/-! ## Tier 3: Known Results Explained -/

#check delta_max_temperature_scaling
#check damage_monotone_decreasing_J
#check steepest_drop_J1_to_J2

/-! ## Tier 4: Predictions -/

#check prediction_herfindahl_test
#check prediction_VRI_lattice
#check prediction_conductivity_hardness
#check prediction_kramers_escape
#check prediction_unification

/-! ## Tier 5: Ceramic-Specific -/

-- Phonon Scattering
#check klemensGammaMass
#check dualSublatticeGamma
#check ZrHf_gamma_near_max

-- Thermal Transport
#check kappaRad
#check kappa_rad_cubic_T
#check kappa_rad_dominates_above_Tcrit
#check pore_reduction_beats_composition

-- Coarsening
#check coarseningLaw
#check HE_coarsening_suppression

/-! ## Statistical Mechanics (NEW) -/

-- Gibbs-Boltzmann Distribution
#check boltzmannWeight
#check gibbsPartitionFn
#check gibbsProb
#check boltzmannWeight_pos
#check gibbsPartitionFn_pos
#check gibbsProb_pos
#check gibbsProb_sum_one
#check gibbsProb_is_shareFunction

-- Free Energy and Thermodynamic Potentials
#check helmholtzFreeEnergy
#check gibbsMean
#check gibbsVariance

-- Escort-Logit Bridge
#check escort_logit_bridge

-- Gibbs Variational Principle
#check gibbs_variational_principle

/-! ## Fluctuation-Response Relations (NEW) -/

-- Algebraic VRI Core
#check algebraic_vri_core
#check variance_centered_form

-- Static and Dynamic VRI
#check gibbs_static_vri
#check alloy_vri
#check dynamic_vri

-- Divergence Near Critical Temperature
#check varianceAtTemp
#check varianceAtTemp_pos
#check varianceAtTemp_monotone

-- Onsager Reciprocity
#check onsager_reciprocity

/-! ## Phase Transitions (NEW) -/

-- Order Parameter
#check reducedOrderParam
#check keff_eq_K_times_reduced
#check reducedOrderParam_continuous
#check reducedOrderParam_at_critical

-- Second-Order Transition
#check slope_below_critical
#check slope_above_critical
#check order_parameter_exponent_one
#check susceptibility_exponent_one

-- Landau Potential
#check landauPotential
#check landau_supercritical_minimizer
#check landau_subcritical_minimizer
#check landau_gives_keff

-- Universality and Finite-Size
#check universality
#check transitionWidth
#check transitionWidth_decreasing

/-! ## q-Equilibrium and Tsallis (NEW) -/

-- q-Exponential Allocation
#check qExpAllocation
#check compact_support_q_lt_one
#check logit_recovery

-- Tsallis Uniqueness
#check qSum
#check qSum_at_one
#check qSum_comm
#check qSum_assoc
#check tsallis_uniqueness

-- Pareto Exponent
#check paretoExponent
#check paretoExponent_pos
#check paretoExponent_decreasing

/-! ## Kramers Escape and Crooks (NEW) -/

-- Kramers Rate
#check kramersRate
#check decompositionTime
#check kramersRate_decreasing_in_barrier
#check kramersRate_increasing_in_T

-- Barrier Enhancement
#check barrierFromCurvature
#check barrier_increasing_in_Keff

-- Crooks Fluctuation Theorem
#check crooksRatio
#check crooksRatio_gt_one
#check jarzynski_second_law

-- Compound Symmetry
#check compoundSymmEigMarket
#check compoundSymmEigIdio
#check compound_symmetry_trace
#check portfolio_diversification

/-! ## Entropy Inequalities (CRITICAL) -/

-- Core lemma: strict convexity of exp
#check exp_gt_one_add

-- THE central entropy theorem: S_q > S_1 for q < 1
#check qEntropy_exceeds_shannon

-- Monotonicity and bounds
#check curvatureK_increasing_in_J
#check curvatureK_lt_one_minus_q

-- Effective barrier increase
#check effective_barrier_increase

/-! ## Sharper Predictions (from mathematical audit) -/

#check prediction_cross_subsystem_consistency
#check prediction_voigt_reuss_bridge
#check prediction_apt_fluctuation_scaling
#check prediction_escort_weight_test
#check prediction_hardness_conductivity_mirror
#check prediction_operating_temperature_alloy
