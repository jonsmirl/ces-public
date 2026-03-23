#!/usr/bin/env python3
"""
Reassess all 7 proposed materials in light of computational validation.

For each proposal, evaluate:
1. Which q-theory predictions rely on VALIDATED mechanisms?
2. Which rely on INVALIDATED mechanisms?
3. Which rely on mechanisms INDEPENDENT of q-theory (standard physics)?
4. Overall confidence assessment.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hea_validate_core import (
    ELEMENTS, ALPHA_DEFAULT, compute_delta, compute_q_from_delta,
    compute_CES, compute_H, compute_K, get_alloy_properties,
)

# =============================================================================
# Classification of mechanisms by validation status
# =============================================================================

VALIDATED = [
    "K orders property deviations (r>0.85)",
    "Radiation damage D=1/(1+cK), R²=0.96",
    "Yang-Zhang classification K_eff>0, 94%",
    "S_q > S_1 (mathematical theorem)",
    "Per-element surplus peaks at J=2",
    "CES bounds bracket E for BCC alloys (crystal-consistent inputs)",
    "Steepest effect at J=1→2",
]

INVALIDATED = [
    "Single q fits all properties (fails: q varies by channel)",
    "CES power mean for thermal conductivity (κ below Reuss bound)",
    "K predicts magnitude of transport deficits (off by 50-200×)",
    "Mirror prediction Δσ/σ ≈ -Δκ/κ (slope≈3.6, not 1)",
    "Voigt-Reuss bounds with stable-phase moduli for FCC alloys",
]

INDEPENDENT_OF_Q = [
    "Preferential oxidation (thermodynamic ΔG, not q-theory)",
    "Self-healing oxide (reactive element effect, standard metallurgy)",
    "Mass disorder phonon scattering (Klemens model)",
    "Entropy-stabilized sluggish diffusion (established for HE oxides)",
    "Neutron absorption cross-sections (nuclear physics)",
    "Density reduction from lighter elements (arithmetic)",
    "Wiedemann-Franz electronic thermal conductivity",
    "VEC→BCC crystal structure prediction (Guo criterion)",
    "Preferential sputtering by mass (momentum transfer kinematics)",
    "Bifunctional oxide (solid scaffold + liquid sealant, phase diagram)",
]


def assess_alloy(name, symbols, predictions):
    """Assess a proposed alloy/ceramic."""
    print(f"\n{'='*75}")
    print(f"  {name}")
    print(f"{'='*75}")

    if all(s in ELEMENTS for s in symbols):
        props = get_alloy_properties(symbols)
        print(f"\n  Composition: {'-'.join(symbols)} (equimolar)")
        print(f"  δ = {props['delta_pct']:.2f}%, q = {props['q']:.3f}, "
              f"K = {props['K']:.3f}")
        print(f"  ρ = {props['rho']:.2f} g/cm³, T̄_m = {props['T_m']:.0f} K, "
              f"VEC = {props['VEC']:.1f}")

        # Is this a good test case for q-theory?
        if props['K'] < 0.05:
            print(f"  ⚠ K = {props['K']:.3f} is very small — CES ≈ ROM, "
                  f"q-theory has minimal predictive power")
        elif props['K'] < 0.15:
            print(f"  ◐ K = {props['K']:.3f} is moderate — some CES curvature")
        else:
            print(f"  ✓ K = {props['K']:.3f} is substantial — meaningful CES curvature")

        # Crystal structure consistency
        bcc_elems = {'W', 'Mo', 'Ta', 'Nb', 'Cr', 'V'}
        hcp_elems = {'Co', 'Zr', 'Hf', 'Ti'}
        fcc_elems = {'Ni', 'Cu', 'Al'}
        complex_elems = {'Mn'}

        structs = set()
        for s in symbols:
            if s in bcc_elems: structs.add('BCC')
            elif s in hcp_elems: structs.add('HCP')
            elif s in fcc_elems: structs.add('FCC')
            elif s in complex_elems: structs.add('complex')

        if len(structs) == 1:
            print(f"  ✓ All elements share crystal structure ({structs.pop()}) "
                  f"— no input data mismatch")
        else:
            print(f"  ⚠ Mixed crystal structures: {structs} — "
                  f"Voigt-Reuss bounds may not bracket E")

    # Assess each prediction
    print(f"\n  Predictions and their validation status:")
    n_valid = 0
    n_invalid = 0
    n_independent = 0
    n_untested = 0

    for pred_name, mechanism, status in predictions:
        if status == 'validated':
            icon = '✓'
            n_valid += 1
        elif status == 'invalidated':
            icon = '✗'
            n_invalid += 1
        elif status == 'independent':
            icon = '◆'
            n_independent += 1
        else:
            icon = '?'
            n_untested += 1

        print(f"    {icon} {pred_name}")
        print(f"      Mechanism: {mechanism}")
        print(f"      Status: {status}")

    total = n_valid + n_invalid + n_independent + n_untested
    print(f"\n  SCORECARD: {n_valid} validated, {n_invalid} invalidated, "
          f"{n_independent} independent of q-theory, {n_untested} untested")

    # Confidence assessment
    if n_invalid == 0 and n_valid + n_independent > 0:
        conf = "HIGH"
    elif n_invalid > 0 and n_independent > n_invalid:
        conf = "MODERATE (core claims not dependent on q-theory)"
    elif n_invalid > 0 and n_valid > n_invalid:
        conf = "MODERATE"
    else:
        conf = "LOW"
    print(f"  CONFIDENCE: {conf}")

    return {'valid': n_valid, 'invalid': n_invalid, 'independent': n_independent,
            'untested': n_untested, 'confidence': conf}


if __name__ == '__main__':
    print("=" * 75)
    print("  REASSESSMENT OF PROPOSED MATERIALS")
    print("  In light of computational validation findings")
    print("=" * 75)

    print(f"\n  Validation summary:")
    print(f"  • K correctly ORDERS property deviations (correlations >0.85)")
    print(f"  • K does NOT predict magnitudes for transport properties")
    print(f"  • CES bounds work for BCC alloys, FAIL for FCC (crystal mismatch)")
    print(f"  • Thermal κ is electronic (Wiedemann-Franz), not CES aggregate")
    print(f"  • Radiation damage model D=1/(1+cK) works well (R²=0.96)")
    print(f"  • Many 'q-theory predictions' are actually standard physics")

    results = {}

    # HEA-1: WMoTaNbZr
    results['HEA-1'] = assess_alloy(
        "HEA-1: WMoTaNbZr — NTP Cladding",
        ['W', 'Mo', 'Ta', 'Nb', 'Zr'],
        [
            ("Radiation tolerance ~4× better than W-Re",
             "D=1/(1+cK) with K=0.22 vs K≈0.02 for W-Re",
             "validated"),
            ("Phase stability (Ω=13.6≫1.1, δ=5.25%<6.6%)",
             "Yang-Zhang / K_eff>0 classification",
             "validated"),
            ("38% lighter than W-Re (ρ=12.25 vs 19.7)",
             "Arithmetic average of element densities",
             "independent"),
            ("53% fewer neutrons absorbed",
             "σ_a averaging (nuclear physics, not q-theory)",
             "independent"),
            ("BCC phase (VEC=5.2)",
             "Guo VEC criterion",
             "independent"),
            ("CES elastic modulus prediction",
             "CES bounds with BCC-consistent inputs",
             "validated"),
        ]
    )

    # HEA-2: CrMoNbTaV
    results['HEA-2'] = assess_alloy(
        "HEA-2: CrMoNbTaV — Sputter-Resistant Ion Thruster Grid",
        ['Cr', 'Mo', 'Nb', 'Ta', 'V'],
        [
            ("Self-hardening surface via escort distribution",
             "Preferential sputtering enriches heavy atoms (Ta)",
             "independent"),
            ("~30% lower equilibrium sputter yield vs Mo",
             "Escort distribution / CES surface evolution",
             "untested (escort mechanism not validated)"),
            ("Phase stability (BCC, moderate δ)",
             "Yang-Zhang / K_eff>0",
             "validated"),
            ("Mass diversity drives surface adaptation",
             "Momentum transfer kinematics (F=ma, not q-theory)",
             "independent"),
        ]
    )

    # HEA-3: WMoTaCrHf
    results['HEA-3'] = assess_alloy(
        "HEA-3: WMoTaCrHf — Operating-Temperature NTP Alloy",
        ['W', 'Mo', 'Ta', 'Cr', 'Hf'],
        [
            ("Stable ONLY above 2130K (δ=7.17% > 6.6%)",
             "Temperature-dependent δ_max = 6.6%×√(T/T_ref)",
             "untested (novel prediction, no data)"),
            ("q-entropy stabilization S_q > S_1",
             "Tsallis entropy exceeds Shannon (proven theorem)",
             "validated (math), untested (physical effect)"),
            ("K=0.41 → 1.9× better radiation tolerance than HEA-1",
             "D=1/(1+cK) ratio",
             "validated (model form), quantitative ratio untested"),
            ("87% higher curvature than HEA-1",
             "K computation from δ",
             "validated (arithmetic)"),
        ]
    )

    # HEA-4: WTaNbHfZr
    results['HEA-4'] = assess_alloy(
        "HEA-4: WTaNbHfZr — Self-Protecting Rocket Nozzle",
        ['W', 'Ta', 'Nb', 'Hf', 'Zr'],
        [
            ("Self-forming HfO₂-ZrO₂ oxide layer",
             "Thermodynamic ΔG preference (600 kJ/mol), reactive element effect",
             "independent"),
            ("Self-healing after thermal cycling",
             "Hf/Zr diffusion to cracks (standard oxidation metallurgy)",
             "independent"),
            ("39% lighter than Re/Ir (ρ=12.86 vs 21.0)",
             "Arithmetic density average",
             "independent"),
            ("No Mo (avoids volatile MoO₃)",
             "Thermochemical design (vapor pressure data)",
             "independent"),
            ("Escort-distribution surface evolution",
             "CES escort model for surface composition",
             "untested (escort mechanism not validated)"),
            ("Specific creep advantage from K=0.24",
             "K × δ² strengthening",
             "validated (K ordering), magnitude uncertain"),
            ("Phase stability (δ=5.45%, Ω≫1.1)",
             "Yang-Zhang / K_eff>0",
             "validated"),
        ]
    )

    # HEC-5: High-entropy diboride
    results['HEC-5'] = assess_alloy(
        "HEC-5: (Hf,Zr,Ta,Nb,Ti)B₂ — UHTC Heat Shield",
        ['Hf', 'Zr', 'Ta', 'Nb', 'Ti'],
        [
            ("Bifunctional oxide scale (solid scaffold + liquid sealant)",
             "Phase diagram: HfO₂/ZrO₂ solid, Ta₂O₅/Nb₂O₅/TiO₂ liquid at 2000°C",
             "independent"),
            ("Self-healing oxide on crack exposure",
             "Standard oxidation/reactive element metallurgy",
             "independent"),
            ("5× lower oxidation rate from dense scale",
             "Liquid-phase sintering of oxide eliminates porosity",
             "independent"),
            ("Low thermal conductivity (~15 W/mK)",
             "Mass disorder phonon scattering (Klemens model)",
             "independent"),
            ("Thermal shock resistance",
             "Liquid oxide phase accommodates strain",
             "independent"),
            ("Phase stability of AlB₂ structure",
             "Confirmed experimentally (Gild 2016)",
             "independent"),
            ("Escort distribution drives surface evolution",
             "CES escort model",
             "untested"),
        ]
    )

    # HEO-6: Rare-earth zirconate foam
    results['HEO-6'] = assess_alloy(
        "HEO-6: (La,Nd,Sm,Gd,Y)₂Zr₂O₇ — Foam Tile TPS",
        [],  # Ceramic, not in element DB
        [
            ("Entropy-suppressed sintering (150-250°C higher onset)",
             "Sluggish diffusion in HE oxides (measured experimentally)",
             "independent"),
            ("Ultra-low bulk κ ≈ 1.0 W/mK",
             "Mass disorder phonon scattering (Klemens), confirmed by Zhao 2020",
             "independent"),
            ("High intrinsic emissivity (ε ≥ 0.85)",
             "RE-O phonon absorption + Nd/Sm f-f transitions (spectroscopy)",
             "independent"),
            ("Non-hygroscopic (no waterproofing needed)",
             "Ceramic chemistry (zirconate vs silica)",
             "independent"),
            ("Thermal cycling durability (>1000 cycles)",
             "No phase transformation + entropy-stabilized lattice",
             "independent"),
            ("K predicts quantitative sintering rate",
             "K × δ² coupling for diffusion",
             "invalidated (K magnitude too small for quantitative prediction)"),
        ]
    )

    # HEO-6b: Nanofiber tile
    results['HEO-6b'] = assess_alloy(
        "HEO-6b: (La,Gd,Y)₂(Zr,Hf)₂O₇ — Nanofiber Tile",
        [],  # Ceramic
        [
            ("Dual-sublattice sintering suppression",
             "Sluggish diffusion on both A and B sublattices",
             "independent"),
            ("Fine pore size (2-5 μm) survives at operating T",
             "HE sintering suppression preserves nanofiber architecture",
             "independent"),
            ("κ_eff ≤ 0.03 W/mK from fine pores",
             "Radiative transport suppression (κ_rad ∝ T³·d_pore)",
             "independent"),
            ("Coarsening resistance (inter-fiber gaps stable)",
             "Dual-sublattice disorder raises activation energy",
             "independent"),
        ]
    )

    # Overall summary
    print("\n" + "=" * 75)
    print("  OVERALL REASSESSMENT SUMMARY")
    print("=" * 75)

    print(f"\n  {'Material':<40s} {'V':>3s} {'I':>3s} {'Ind':>4s} {'U':>3s} {'Confidence':<15s}")
    print("  " + "-" * 70)
    for name in ['HEA-1', 'HEA-2', 'HEA-3', 'HEA-4', 'HEC-5', 'HEO-6', 'HEO-6b']:
        r = results[name]
        print(f"  {name:<40s} {r['valid']:>3d} {r['invalid']:>3d} "
              f"{r['independent']:>4d} {r['untested']:>3d} {r['confidence']:<15s}")
    print(f"\n  V=validated, I=invalidated, Ind=independent of q-theory, U=untested")

    print(f"""
  KEY FINDING: The strongest proposals (HEA-1, HEA-4, HEC-5, HEO-6,
  HEO-6b) derive their advantages primarily from STANDARD PHYSICS
  (thermodynamics, kinetics, nuclear data) rather than from q-theory.
  The q-theory provided the DESIGN FRAMEWORK for selecting these
  compositions, but the predicted performance relies on mechanisms
  that are independent of whether q < 1 or K = (1-q)(1-H) is correct.

  This is actually GOOD NEWS for the proposals: their viability does
  NOT depend on the validity of q-thermodynamics. The compositions
  were selected using q-theory as a heuristic, but their performance
  derives from established physics.

  The q-theory's validated contributions:
  • Radiation tolerance ordering via K (HEA-1, HEA-3)
  • Phase stability screening via K_eff (all metallic HEAs)
  • Design heuristic: maximize compositional diversity (all)

  The q-theory's invalidated claims that DON'T affect the proposals:
  • Single q across properties (doesn't affect any specific prediction)
  • CES for thermal conductivity (proposals use Klemens, not CES)
  • Mirror prediction (not used in any proposal's design)

  BOTTOM LINE: 6 of 7 proposals are sound. HEA-3 (operating-temperature
  alloy) is the only high-risk proposal, as it tests the untested
  δ_max(T) ∝ √T prediction. But that's by design — it's explicitly
  labeled as the "bold test" of the theory.
""")
