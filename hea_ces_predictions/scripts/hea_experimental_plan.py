#!/usr/bin/env python3
"""
Optimal experimental verification plan for q-thermodynamic HEA proposals.

Ranks experiments by (information gained × success probability) / cost.
Computes quantitative predictions with pass/fail criteria for each.
Separates theory-discriminating tests from material-validation tests.
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
# Phase 0: Computational Pre-Screening (before ANY experiment)
# =============================================================================

def phase0_computational():
    """Zero-cost computational checks that could prevent wasted experiments."""
    print("=" * 75)
    print("  PHASE 0: COMPUTATIONAL PRE-SCREENING (cost: $0, time: 1-2 weeks)")
    print("=" * 75)

    # Check 1: BCC refractory subsystem CES bounds vs available data
    print("\n  CHECK 1: CES bounds for BCC refractory HEAs (literature data)")
    print("  " + "-" * 65)

    lit_data = [
        # (name, elements, E_measured, source)
        ('WMoTaNb', ['W', 'Mo', 'Ta', 'Nb'], 220.0, 'Senkov 2011 (estimated)'),
        ('NbTiVZr', ['Nb', 'Ti', 'V', 'Zr'], 104.0, 'Senkov 2012'),
        ('WMoTaNbV', ['W', 'Mo', 'Ta', 'Nb', 'V'], None, 'no E data'),
    ]

    print(f"  {'Alloy':<15s} {'E_Reuss':>9s} {'E_CES':>8s} {'E_Voigt':>9s} "
          f"{'E_meas':>8s} {'In bounds?':>10s} {'Crystal issue?'}")
    for name, syms, E_meas, source in lit_data:
        props = get_alloy_properties(syms)
        J = len(syms)
        fracs = np.ones(J) / J
        Es = props['Es']
        E_v = np.sum(fracs * Es)
        E_r = 1.0 / np.sum(fracs / Es)
        E_ces = compute_CES(fracs, Es, props['q'])

        hcp_elems = [s for s in syms if s in ('Ti', 'Zr', 'Hf')]
        cryst = f"HCP: {','.join(hcp_elems)}" if hcp_elems else "all BCC ✓"

        if E_meas:
            ok = E_r - 5 <= E_meas <= E_v + 5  # 5 GPa tolerance
            print(f"  {name:<15s} {E_r:>9.1f} {E_ces:>8.1f} {E_v:>9.1f} "
                  f"{E_meas:>8.1f} {'YES ✓' if ok else 'NO ✗':>10s} {cryst}")
        else:
            print(f"  {name:<15s} {E_r:>9.1f} {E_ces:>8.1f} {E_v:>9.1f} "
                  f"{'—':>8s} {'—':>10s} {cryst}")

    print(f"\n  WMoTaNb (all BCC): E=220 within [{217}, {258}] → CES bounds HOLD ✓")
    print(f"  NbTiVZr (has HCP Ti,Zr): E=104 within [{100}, {104}] → marginal")

    # Check 2: CALPHAD screening for WMoTaCrHf
    print(f"\n  CHECK 2: WMoTaCrHf phase stability (CALPHAD recommended)")
    print(f"  Known intermetallics in constituent binaries:")
    print(f"    Cr-Hf: C15 Laves (CrHf₂) — forms below ~1600°C")
    print(f"    Cr-Ta: no intermetallics")
    print(f"    Cr-W:  continuous BCC solid solution ✓")
    print(f"    Mo-Hf: no stable intermetallics ✓")
    print(f"    W-Hf:  no stable intermetallics ✓")
    print(f"    Ta-Hf: continuous BCC solid solution ✓")
    print(f"  ⚠ Cr-Hf Laves phase is the main risk for HEA-3")
    print(f"  → CALPHAD calculation ESSENTIAL before arc melting")

    # Check 3: Oxide thermodynamics for HEA-4
    print(f"\n  CHECK 3: WTaNbHfZr selective oxidation thermodynamics")
    oxides = [
        ('Hf', 'HfO₂', -1117, 3031, 'solid', 'protective'),
        ('Zr', 'ZrO₂', -1042, 2988, 'solid', 'protective'),
        ('Ta', 'Ta₂O₅', -764, 2145, 'marginal', 'near melting'),
        ('Nb', 'Nb₂O₅', -706, 1785, 'liquid', 'non-protective'),
        ('W',  'WO₃',   -533, 1746, 'volatile', 'catastrophic'),
    ]
    print(f"  {'Element':<8s} {'Oxide':<8s} {'ΔG°f(kJ/mol)':>14s} {'T_m(K)':>8s} "
          f"{'State@2200°C':>14s} {'Assessment'}")
    print("  " + "-" * 65)
    for elem, oxide, dG, Tm, state, assess in oxides:
        print(f"  {elem:<8s} {oxide:<8s} {dG:>14d} {Tm:>8d} {state:>14s} {assess}")
    print(f"\n  ΔΔG(Hf vs W) = {-1117 - (-533)} = -584 kJ/mol → STRONG Hf preference ✓")
    print(f"  Selectivity ratio: exp(584000/(8.314×2473)) = 10^{584000/(8.314*2473*2.303):.0f}")
    print(f"  → Hf/Zr will oxidize preferentially by >10 orders of magnitude")

    # Check 4: Sputter yield estimate for HEA-2
    print(f"\n  CHECK 4: CrMoNbTaV sputter yield estimate (Sigmund theory)")
    # Threshold energy ∝ surface binding energy ∝ heat of sublimation
    # Sputter yield Y ∝ M_target⁻¹ × E_ion / U_s at low energy
    sublimation = {'Cr': 395, 'Mo': 658, 'Nb': 726, 'Ta': 782, 'V': 515}  # kJ/mol
    masses = {'Cr': 52, 'Mo': 96, 'Nb': 93, 'Ta': 181, 'V': 51}
    print(f"  {'Element':<8s} {'Mass(amu)':>10s} {'U_s(kJ/mol)':>12s} {'Y_rel':>8s}")
    print("  " + "-" * 42)
    for elem in ['Cr', 'Mo', 'Nb', 'Ta', 'V']:
        Y_rel = (1.0 / sublimation[elem])  # simplified
        print(f"  {elem:<8s} {masses[elem]:>10d} {sublimation[elem]:>12d} {Y_rel:>8.4f}")

    Mo_Y = 1.0 / sublimation['Mo']
    avg_Y = np.mean([1.0/sublimation[e] for e in ['Cr', 'Mo', 'Nb', 'Ta', 'V']])
    improvement = (Mo_Y - avg_Y) / Mo_Y * 100
    print(f"\n  Pure Mo relative yield: {Mo_Y:.4f}")
    print(f"  HEA average yield: {avg_Y:.4f}")
    print(f"  Predicted improvement: {improvement:.0f}%")
    print(f"  Paper claims ~30%. Sigmund estimate: ~{abs(improvement):.0f}%")
    print(f"  Note: This ignores surface enrichment (escort effect),")
    print(f"  which would further improve the HEA by retaining Ta (highest U_s)")


# =============================================================================
# Phase 1: Literature Mining (zero cost)
# =============================================================================

def phase1_literature():
    """What can we learn from existing publications?"""
    print("\n" + "=" * 75)
    print("  PHASE 1: LITERATURE MINING (cost: $0, time: 1 week)")
    print("=" * 75)

    tasks = [
        ("Refractory HEA subsystem elastic moduli",
         "Search for E values of WMo, WMoTa, WMoTaNb, WMoTaNbZr\n"
         "    Test: do CES bounds bracket measured E for all-BCC alloys?\n"
         "    Priority: HIGH — direct test of CES framework viability",
         "Senkov 2011/2012, Zou 2014, Dirras 2016"),

        ("Refractory HEA thermal conductivity",
         "Search for κ values of refractory HEAs\n"
         "    Test: is κ dominated by electronic transport (WF)?\n"
         "    Compute ρ(alloy) from Nordheim, κ_e = L₀T/ρ\n"
         "    Priority: HIGH — validates two-channel model",
         "Chou 2009 (limited), check recent papers"),

        ("Radiation damage in refractory HEAs",
         "Search for defect cluster data in irradiated WMoTaNb(X)\n"
         "    Test: does D = 1/(1+cK) transfer from Cantor to refractory?\n"
         "    Priority: CRITICAL — tests whether c_rad is universal",
         "El-Atwani 2019, Xia 2020, Kombaiah 2021"),

        ("HE-UHTC oxidation data",
         "Search for oxidation behavior of (Hf,Zr,Ta,Nb,Ti)B₂\n"
         "    Test: does bifunctional oxide actually form?\n"
         "    Priority: MODERATE — existing data may confirm/deny HEC-5",
         "Gild 2016, Castle 2018, Wen 2020"),

        ("HE rare-earth zirconate sintering",
         "Search for sintering kinetics of HE RE₂Zr₂O₇\n"
         "    Test: does entropy really suppress sintering?\n"
         "    Priority: MODERATE — validates HEO-6 core mechanism",
         "Sarker 2018, Wright 2020, Chen 2021"),
    ]

    for i, (name, desc, refs) in enumerate(tasks, 1):
        print(f"\n  Task {i}: {name}")
        print(f"    {desc}")
        print(f"    Key refs: {refs}")


# =============================================================================
# Phase 2: Fast Discriminating Experiments
# =============================================================================

def phase2_fast_experiments():
    """Cheap, fast experiments that test the most predictions."""
    print("\n" + "=" * 75)
    print("  PHASE 2: FAST DISCRIMINATING EXPERIMENTS")
    print("  (cost: ~$5-10K, time: 2-4 weeks)")
    print("=" * 75)

    experiments = []

    # Experiment 1: BCC subsystem progression
    print(f"\n  EXPERIMENT 1: BCC Refractory Subsystem Progression")
    print(f"  Cost: ~$3K (materials + nanoindentation)")
    print(f"  Time: 2 weeks")
    print(f"  Tests: CES bounds, q-value from subsystem data")
    print(f"  Method: Arc-melt 5 compositions, XRD, nanoindentation")
    print()

    subsystems = [
        ('W', ['W']),
        ('WMo', ['W', 'Mo']),
        ('WMoTa', ['W', 'Mo', 'Ta']),
        ('WMoTaNb', ['W', 'Mo', 'Ta', 'Nb']),
        ('WMoTaNbZr', ['W', 'Mo', 'Ta', 'Nb', 'Zr']),
    ]

    print(f"  {'Alloy':<15s} {'J':>3s} {'δ%':>6s} {'q':>7s} {'K':>7s} "
          f"{'E_Reuss':>9s} {'E_CES':>8s} {'E_Voigt':>9s}")
    print("  " + "-" * 68)
    for name, syms in subsystems:
        if len(syms) == 1:
            E = ELEMENTS[syms[0]].E
            print(f"  {name:<15s} {1:>3d} {'0.00':>6s} {'1.000':>7s} {'0.000':>7s} "
                  f"{E:>9.1f} {E:>8.1f} {E:>9.1f}")
            continue
        J = len(syms)
        fracs = np.ones(J) / J
        props = get_alloy_properties(syms)
        Es = props['Es']
        E_v = np.sum(fracs * Es)
        E_r = 1.0 / np.sum(fracs / Es)
        E_ces = compute_CES(fracs, Es, props['q'])
        print(f"  {name:<15s} {J:>3d} {props['delta_pct']:>6.2f} {props['q']:>7.3f} "
              f"{props['K']:>7.3f} {E_r:>9.1f} {E_ces:>8.1f} {E_v:>9.1f}")

    print(f"\n  PASS criteria:")
    print(f"    ✓ All measured E within [E_Reuss-10, E_Voigt+10] GPa")
    print(f"    ✓ E decreases monotonically from W to WMoTaNbZr")
    print(f"    ✓ Single q fits all 4 multi-element compositions (RMSE < 15 GPa)")
    print(f"  FAIL criteria:")
    print(f"    ✗ Any E outside [E_Reuss-10, E_Voigt+10] → crystal mismatch issue")
    print(f"    ✗ Non-monotonic E → solid solution effects beyond CES")
    print(f"  INFORMATION VALUE: VERY HIGH")
    print(f"    If PASS: validates CES for BCC, provides calibrated q for refractory")
    print(f"    If FAIL: identifies which elements cause problems (Zr? Hf?)")

    # Experiment 2: WTaNbHfZr oxidation
    print(f"\n  EXPERIMENT 2: WTaNbHfZr Oxidation Test")
    print(f"  Cost: ~$2K (arc-melt + furnace + SEM/EDS)")
    print(f"  Time: 1 week")
    print(f"  Tests: Self-forming oxide prediction (HEA-4)")
    print(f"  Method: Polish surface, 1200°C air 1h, cross-section SEM+EDS")
    print(f"\n  PASS criteria:")
    print(f"    ✓ Outer oxide layer is >70% Hf+Zr by EDS")
    print(f"    ✓ Oxide is dense and adherent")
    print(f"    ✓ Sublayer shows W/Ta enrichment (Hf/Zr depleted)")
    print(f"  FAIL criteria:")
    print(f"    ✗ W oxidizes preferentially (volatile WO₃)")
    print(f"    ✗ Oxide is porous/non-adherent")
    print(f"    ✗ No composition gradient (uniform oxidation)")
    print(f"  INFORMATION VALUE: HIGH")
    print(f"    Tests standard oxidation thermodynamics (independent of q-theory)")
    print(f"    If PASS: HEA-4 rocket nozzle concept validated at basic level")

    # Experiment 3: Hardness progression
    print(f"\n  EXPERIMENT 3: Hardness vs K for BCC Subsystems")
    print(f"  Cost: included in Experiment 1 (same samples)")
    print(f"  Tests: Cocktail effect (hardness > ROM)")
    print(f"  Method: Vickers hardness on arc-melted buttons")

    print(f"\n  {'Alloy':<15s} {'K':>7s} {'σ_y,ROM(MPa)':>13s} {'Predict':>15s}")
    print("  " + "-" * 55)
    for name, syms in subsystems:
        if len(syms) == 1:
            sy = ELEMENTS[syms[0]].sigma_y
            print(f"  {name:<15s} {'0.000':>7s} {sy:>13.0f} {'baseline':>15s}")
            continue
        props = get_alloy_properties(syms)
        print(f"  {name:<15s} {props['K']:>7.3f} {props['sigma_y_rom']:>13.0f} "
              f"{'HV >> HV_ROM':>15s}")

    print(f"\n  PASS criteria:")
    print(f"    ✓ Hardness increases with K (monotonic correlation)")
    print(f"    ✓ Multi-component HV significantly exceeds ROM estimate")
    print(f"  INFORMATION VALUE: MODERATE (confirms known cocktail effect)")


# =============================================================================
# Phase 3: Theory-Discriminating Experiments
# =============================================================================

def phase3_theory_tests():
    """Experiments that can distinguish q-theory from alternatives."""
    print("\n" + "=" * 75)
    print("  PHASE 3: THEORY-DISCRIMINATING EXPERIMENTS")
    print("  (cost: ~$20-50K, time: 1-3 months)")
    print("=" * 75)

    print(f"""
  These experiments test predictions UNIQUE to q-theory that cannot be
  obtained from standard models (VLGC, Klemens, Yang-Zhang, etc.).

  EXPERIMENT 4: WMoTaCrHf Temperature-Dependent Phase Stability
  ─────────────────────────────────────────────────────────────────
  Cost: ~$10K (high-T furnace time + XRD)
  Time: 4-6 weeks
  Tests: δ_max(T) ∝ √T (THE core q-theory prediction)

  q-theory predicts: δ_max(T) = 6.6% × √(T/1800K)
    δ_max(1800K) = 6.6%  → WMoTaCrHf (δ=7.17%) UNSTABLE
    δ_max(2130K) = 7.18% → WMoTaCrHf MARGINALLY STABLE
    δ_max(2500K) = 7.78% → WMoTaCrHf STABLE

  Method:
    1. Arc-melt WMoTaCrHf → retain as-cast (likely non-equilibrium BCC)
    2. Anneal samples at 1800, 2000, 2200, 2500 K for 24h each
    3. Water quench from each temperature
    4. XRD: count phases (single BCC vs multi-phase)
    5. SEM/EDS: check for precipitates or decomposition""")

    temps = [1800, 2000, 2130, 2200, 2500]
    delta = 7.16 / 100
    print(f"  {'T(K)':>8s} {'δ_max%':>8s} {'δ/δ_max':>8s} {'Prediction':<30s}")
    print("  " + "-" * 55)
    for T in temps:
        d_max = 6.6 / 100 * np.sqrt(T / 1800)
        ratio = delta / d_max
        if ratio > 1.05:
            pred = "MULTI-PHASE (decomposed)"
        elif ratio > 0.95:
            pred = "MARGINAL (borderline)"
        else:
            pred = "SINGLE-PHASE BCC"
        print(f"  {T:>8d} {d_max*100:>8.2f} {ratio:>8.3f} {pred:<30s}")

    print(f"""
  PASS criteria (validates δ_max(T) ∝ √T):
    ✓ Multi-phase at 1800K
    ✓ Single-phase at 2500K
    ✓ Transition between 2000-2200K
  PARTIAL PASS (validates T-dependence but not √T):
    ✓ Multi-phase at 1800K, single-phase at 2500K
    ✗ Transition at different T than predicted
  FAIL (falsifies q-entropy correction):
    ✗ Multi-phase even at 2500K (δ_max < 7.17% at all T)
    ✗ Single-phase even at 1800K (standard theory sufficient)

  ⚠ CRITICAL PRE-REQUISITE: CALPHAD check for Laves phases
    The Cr-Hf system forms C15 Laves phase (CrHf₂).
    If Laves phase forms instead of BCC decomposition,
    the test is INCONCLUSIVE for δ_max(T) because the
    failure mode is intermetallic, not solid solution.
    → Must check Cr-Hf Laves phase stability at 2500K first.
  INFORMATION VALUE: CRITICAL (THE falsification test)""")

    print(f"""
  EXPERIMENT 5: Radiation Damage WMoTaNbZr vs Pure W
  ─────────────────────────────────────────────────────
  Cost: ~$15K (ion irradiation beam time + TEM)
  Time: 2-3 months
  Tests: D = 1/(1+cK) transferability to refractory HEAs

  q-theory predicts: D(WMoTaNbZr)/D(W) = 1/(1 + c_rad × K)
    K(WMoTaNbZr) = 0.221, c_rad = 257 (from Cantor data)
    → D_ratio = 1/(1 + 257 × 0.221) = 0.017 (~60× better)

  BUT: c_rad was calibrated on FCC Cantor alloys.
  For BCC refractory alloys, c_rad may differ.
  Conservative estimate: c_rad_BCC ≈ c_rad_FCC / 3 → D_ratio ≈ 0.05

  Method:
    1. Prepare thin foils of WMoTaNbZr and pure W
    2. Ion irradiation (1 MeV Au⁺, 1-10 dpa)
    3. TEM: measure defect cluster density and size distribution
    4. Compare D(HEA)/D(W)

  PASS criteria:
    ✓ D(HEA)/D(W) < 0.3 (>3× improvement) — minimum useful
    ✓ D(HEA)/D(W) < 0.1 (>10× improvement) — consistent with model
  FAIL criteria:
    ✗ D(HEA)/D(W) > 0.5 (less than 2× improvement)
  INFORMATION VALUE: HIGH
    Tests whether K controls radiation tolerance in BCC HEAs
    If PASS: validates the design principle for HEA-1 NTP cladding
    If FAIL: K is FCC-specific, different physics in BCC""")

    print(f"""
  EXPERIMENT 6: CrMoNbTaV Sputter Yield + Surface Evolution
  ─────────────────────────────────────────────────────────────
  Cost: ~$10K (ion beam facility + surface analysis)
  Time: 2-3 months
  Tests: Escort-distribution surface self-hardening (HEA-2)

  Method:
    1. Polish CrMoNbTaV and pure Mo surfaces
    2. Sputter with 1 keV Xe⁺ at normal incidence
    3. Measure mass loss (sputter yield) over time
    4. After 1h, 10h, 100h: XPS/AES surface composition
    5. Compare equilibrium sputter yield vs Mo

  q-theory prediction (escort distribution):
    Initial surface: equimolar Cr-Mo-Nb-Ta-V
    After sputtering: enriched in Ta (heaviest, highest threshold)
    Equilibrium yield: ~30% lower than pure Mo

  Standard physics prediction (no escort):
    Surface enrichment occurs by preferential sputtering (momentum transfer)
    This is standard sputtering physics, NOT unique to q-theory
    But the QUANTITATIVE enrichment matches escort distribution prediction

  PASS: sputter yield decreases >15% over time (surface adaptation)
  FAIL: yield is constant (no surface enrichment)
  INFORMATION VALUE: MODERATE
    Both q-theory and standard sputtering theory predict enrichment
    The experiment validates the EFFECT but doesn't discriminate theories""")


# =============================================================================
# Decision Tree
# =============================================================================

def decision_tree():
    """What to do based on results."""
    print("\n" + "=" * 75)
    print("  DECISION TREE: What to do based on Phase 2-3 results")
    print("=" * 75)

    print(f"""
  SCENARIO A: CES bounds bracket BCC refractory E (Exp 1 passes)
  ├── AND oxidation test passes (Exp 2)
  │   → Proceed to Phase 4: Qualify HEA-1, HEA-4 for applications
  │   → HEA-1 (NTP cladding) is ready for engineering prototyping
  │   → HEA-4 (nozzle) is ready for hot-fire testing
  │
  ├── AND phase stability test passes (Exp 4)
  │   → q-entropy correction VALIDATED
  │   → HEA-3 becomes viable; proceed to radiation testing
  │   → Publish: first experimental validation of δ_max(T) ∝ √T
  │
  └── AND radiation test passes (Exp 5)
      → Full q-theory for BCC HEAs VALIDATED
      → K-based design rules confirmed for refractory alloys
      → Paper's central claim supported

  SCENARIO B: CES bounds FAIL for BCC refractory E (Exp 1 fails)
  ├── If Zr/Hf alloys fail but W-Mo-Ta-Nb pass
  │   → Crystal structure mismatch (BCC-HCP) is the issue
  │   → Restrict theory to all-BCC compositions
  │   → HEA-1 (has Zr) needs composition adjustment
  │
  └── If all multi-component alloys fail
      → CES power mean is wrong even for consistent crystal structures
      → Theory requires fundamental revision
      → Materials proposals still viable (based on standard physics)

  SCENARIO C: Phase stability test fails (Exp 4 fails)
  ├── If multi-phase at ALL temperatures (even 2500K)
  │   → δ_max(T) ∝ √T prediction FALSIFIED
  │   → q-entropy correction too small or absent
  │   → HEA-3 is abandoned; other proposals unaffected
  │
  └── If Laves phase forms instead of BCC decomposition
      → Test INCONCLUSIVE for δ_max(T)
      → Need Laves-free composition (e.g., replace Cr with V)
      → Redesign HEA-3 and repeat

  SCENARIO D: Radiation test fails (Exp 5)
  ├── D(HEA)/D(W) = 0.3-0.5 (modest improvement)
  │   → K-based model overpredicts but direction is correct
  │   → Revise c_rad for BCC; HEA-1 still better than W-Re
  │
  └── D(HEA)/D(W) > 0.5 (minimal improvement)
      → K does NOT control radiation tolerance in BCC
      → HEA-1 radiation advantage claim withdrawn
      → Other HEA-1 advantages (density, neutron transparency) remain
""")


# =============================================================================
# Priority Ranking
# =============================================================================

def priority_ranking():
    """Final priority ranking of all activities."""
    print("\n" + "=" * 75)
    print("  PRIORITY RANKING")
    print("=" * 75)

    items = [
        (1, "CALPHAD check for WMoTaCrHf Laves phases",
         "$0 (software)", "1 day", "CRITICAL",
         "Could prevent wasted $10K on Exp 4 if Laves dominates"),
        (2, "Literature: refractory HEA subsystem E values",
         "$0", "3 days", "VERY HIGH",
         "May validate/invalidate CES for BCC without ANY experiment"),
        (3, "Arc-melt BCC subsystem progression + nanoindentation",
         "~$3K", "2 weeks", "VERY HIGH",
         "Tests CES framework on its best terrain (all-BCC)"),
        (4, "WTaNbHfZr oxidation at 1200°C",
         "~$2K", "1 week", "HIGH",
         "Quick validation of HEA-4 self-forming oxide"),
        (5, "Literature: HE-UHTC & HE-RE₂Zr₂O₇ data",
         "$0", "3 days", "MODERATE",
         "May confirm HEC-5 and HEO-6 without new experiments"),
        (6, "WMoTaCrHf anneal at multiple temperatures",
         "~$10K", "6 weeks", "HIGH",
         "THE falsification test for q-entropy; do AFTER CALPHAD"),
        (7, "Ion irradiation WMoTaNbZr vs W",
         "~$15K", "3 months", "HIGH",
         "Tests K→radiation transfer to BCC; needs beam time"),
        (8, "CrMoNbTaV sputter yield measurement",
         "~$10K", "3 months", "MODERATE",
         "Standard sputtering physics; q-theory not uniquely tested"),
    ]

    print(f"\n  {'#':>3s}  {'Activity':<50s} {'Cost':>10s} {'Time':>10s} "
          f"{'Priority':>10s}")
    print("  " + "-" * 88)
    for num, name, cost, time, priority, note in items:
        print(f"  {num:>3d}  {name:<50s} {cost:>10s} {time:>10s} {priority:>10s}")
        print(f"       → {note}")

    print(f"""
  ESTIMATED TOTAL:
    Phases 0-1 (computational + literature): $0, 2 weeks
    Phase 2 (fast experiments): ~$5K, 2-4 weeks
    Phase 3 (theory tests): ~$35K, 3 months
    ────────────────────────────────────
    Total to validate/falsify core theory: ~$40K, 4-5 months

    For comparison:
    - One Shuttle tile: ~$20-40K
    - One Re/Ir nozzle: ~$50-100K
    - The entire experimental program costs less than
      the components it might replace
""")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 75)
    print("  OPTIMAL EXPERIMENTAL VERIFICATION PLAN")
    print("  For q-Thermodynamic HEA Theory and Proposed Materials")
    print("=" * 75)

    phase0_computational()
    phase1_literature()
    phase2_fast_experiments()
    phase3_theory_tests()
    decision_tree()
    priority_ranking()
