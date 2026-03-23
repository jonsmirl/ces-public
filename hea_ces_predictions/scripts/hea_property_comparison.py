#!/usr/bin/env python3
"""
Systematic property predictions vs published experimental data.

Compares CES q-thermodynamic predictions against measured properties
for Cantor-alloy subsystems (J=2..5). Generates multi-panel matplotlib
figures and prints numerical results.

Literature sources:
  - Elastic modulus: Laplanche 2015, Wu 2014
  - Thermal conductivity: Jin 2016
  - Hardness: Chou 2009 type data
  - Radiation damage: Zhang et al. 2015
"""

import sys
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# Import core database and functions from hea_validate_core
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hea_validate_core import (
    ELEMENTS,
    Element,
    compute_delta,
    compute_H,
    compute_K,
    compute_q_from_delta,
    compute_Sq,
    compute_S1,
    compute_CES,
    compute_Zq,
    compute_escort,
    get_alloy_properties,
)

# =============================================================================
# Constants & System Definitions
# =============================================================================

# Cantor subsystems in order of increasing J
CANTOR_SUBSYSTEMS = [
    (['Ni', 'Fe'],                 'NiFe',        2),
    (['Ni', 'Co', 'Cr'],          'NiCoCr',      3),
    (['Ni', 'Co', 'Fe', 'Cr'],   'NiCoFeCr',    4),
    (['Co', 'Cr', 'Fe', 'Mn', 'Ni'], 'CoCrFeMnNi', 5),
]

# Published experimental data
EXPT_E_GPA = {
    'NiFe': 200.0,       # Laplanche 2015
    'NiCoCr': 235.0,     # Laplanche 2015
    'NiCoFeCr': 213.0,   # Wu 2014
    'CoCrFeMnNi': 200.0, # Wu 2014
}

EXPT_KAPPA_WMK = {
    'NiFe': 29.0,        # Jin 2016
    'NiCoCr': 12.1,      # Jin 2016
    'NiCoFeCr': 12.0,    # Jin 2016
    'CoCrFeMnNi': 11.8,  # Jin 2016
}

EXPT_HV_GPA = {
    'NiFe': 1.5,         # Chou 2009 type
    'NiCoCr': 1.6,       # approximate
    'NiCoFeCr': 1.3,     # approximate
    'CoCrFeMnNi': 1.1,   # approximate
}

# Pure-element thermal conductivities (for reference line at J=1)
PURE_KAPPAS = {s: ELEMENTS[s].kappa for s in ['Ni', 'Co', 'Cr', 'Fe', 'Mn']}

# Radiation damage: normalized defect cluster density D(J)/D(J=1)
# Zhang et al. 2015
RAD_DAMAGE_DATA = {
    1: 1.00,   # pure Ni
    2: 0.55,   # NiFe
    3: 0.35,   # NiCoCr
    4: 0.25,   # NiCoFeCr
    5: 0.20,   # CoCrFeMnNi
}


# =============================================================================
# Helper Utilities
# =============================================================================

def alloy_props(symbols):
    """Shortcut: equimolar properties dict."""
    return get_alloy_properties(symbols)


def e_ces_band(symbols, q_range=None):
    """
    Compute E_Voigt, E_Reuss, E_CES(q) for an alloy.

    Returns dict with E_voigt, E_reuss, E_ces, q.
    If q_range given, also returns E values across that range.
    """
    p = alloy_props(symbols)
    fracs = p['fracs']
    Es = p['Es']
    q = p['q']

    E_voigt = compute_CES(fracs, Es, 1.0)
    E_reuss = compute_CES(fracs, Es, -1.0)
    E_ces = compute_CES(fracs, Es, q)

    result = {
        'E_voigt': E_voigt,
        'E_reuss': E_reuss,
        'E_ces': E_ces,
        'q': q,
        'J': len(symbols),
        'delta': p['delta'],
        'K': p['K'],
        'H': p['H'],
    }

    if q_range is not None:
        result['q_scan'] = q_range
        result['E_scan'] = np.array([compute_CES(fracs, Es, qq) for qq in q_range])

    return result


def kappa_rom(symbols):
    """Rule-of-mixtures thermal conductivity."""
    p = alloy_props(symbols)
    return p['kappa_rom']


def compute_atomic_volumes(symbols):
    """Atomic volumes proportional to r^3 (in pm^3)."""
    return np.array([ELEMENTS[s].r**3 for s in symbols])


def hardness_rom(symbols):
    """ROM hardness (Vickers, GPa) from pure-element yield strengths.

    Approximate: HV ~ sigma_y / 3  (Tabor relation), then convert to GPa.
    We use sigma_y directly as a proxy (MPa -> GPa /1000 * scaling).
    For consistency with literature HV ~ 1-2 GPa range, use
    HV_ROM = mean(sigma_y) * 0.005 as rough calibration.
    """
    p = alloy_props(symbols)
    sigma_avg = p['sigma_y_rom']  # MPa
    return sigma_avg * 0.005  # crude GPa calibration


# =============================================================================
# Comparison 1: CES Elastic Modulus for Cantor Subsystems
# =============================================================================

def comparison_1_elastic_modulus():
    """
    CES elastic modulus bands vs measured values for Cantor subsystems.

    Returns data dict for plotting and prints results.
    """
    print("\n" + "=" * 70)
    print("  COMPARISON 1: CES Elastic Modulus for Cantor Subsystems")
    print("=" * 70)

    q_range = np.linspace(-1.0, 1.0, 201)
    results = []

    print(f"\n  {'Alloy':<14s} {'J':>2s} {'E_Reuss':>8s} {'E_CES(q)':>9s} "
          f"{'E_Voigt':>8s} {'E_expt':>7s} {'q':>7s}")
    print(f"  {'':14s} {'':>2s} {'(GPa)':>8s} {'(GPa)':>9s} "
          f"{'(GPa)':>8s} {'(GPa)':>7s}")
    print(f"  {'-'*60}")

    for symbols, label, J in CANTOR_SUBSYSTEMS:
        band = e_ces_band(symbols, q_range=q_range)
        E_expt = EXPT_E_GPA[label]
        results.append({
            'label': label,
            'J': J,
            'symbols': symbols,
            'band': band,
            'E_expt': E_expt,
            'q_range': q_range,
        })

        print(f"  {label:<14s} {J:2d} {band['E_reuss']:8.1f} "
              f"{band['E_ces']:9.1f} {band['E_voigt']:8.1f} "
              f"{E_expt:7.1f} {band['q']:7.3f}")

    # Check if measured values fall within Reuss-Voigt bounds
    print(f"\n  Bounds check:")
    for r in results:
        b = r['band']
        inside = b['E_reuss'] <= r['E_expt'] <= b['E_voigt']
        status = "PASS" if inside else "FAIL"
        print(f"    {r['label']:<14s}: Reuss={b['E_reuss']:.1f} <= "
              f"Expt={r['E_expt']:.1f} <= Voigt={b['E_voigt']:.1f}  [{status}]")

    return results


# =============================================================================
# Comparison 2: Thermal Conductivity vs J
# =============================================================================

def comparison_2_thermal_conductivity():
    """
    ROM baseline vs measured thermal conductivity; deficit correlates with K.

    Returns data dict for plotting and prints results.
    """
    print("\n" + "=" * 70)
    print("  COMPARISON 2: Thermal Conductivity vs J")
    print("=" * 70)

    results = []

    # Pure elements (J=1)
    print(f"\n  Pure elements (J=1):")
    for sym in ['Ni', 'Co', 'Cr', 'Fe', 'Mn']:
        k = ELEMENTS[sym].kappa
        print(f"    {sym}: kappa = {k:.1f} W/mK")

    print(f"\n  {'Alloy':<14s} {'J':>2s} {'kappa_ROM':>10s} {'kappa_expt':>11s} "
          f"{'deficit%':>9s} {'K':>7s}")
    print(f"  {'-'*60}")

    for symbols, label, J in CANTOR_SUBSYSTEMS:
        p = alloy_props(symbols)
        k_rom = p['kappa_rom']
        k_expt = EXPT_KAPPA_WMK[label]
        deficit_frac = (k_rom - k_expt) / k_rom
        K = p['K']

        results.append({
            'label': label,
            'J': J,
            'kappa_rom': k_rom,
            'kappa_expt': k_expt,
            'deficit_frac': deficit_frac,
            'K': K,
            'delta': p['delta'],
            'q': p['q'],
        })

        print(f"  {label:<14s} {J:2d} {k_rom:10.1f} {k_expt:11.1f} "
              f"{deficit_frac*100:9.1f} {K:7.4f}")

    # Correlation: deficit fraction vs K
    deficits = np.array([r['deficit_frac'] for r in results])
    Ks = np.array([r['K'] for r in results])
    if len(Ks) > 1:
        corr = np.corrcoef(deficits, Ks)[0, 1]
        print(f"\n  Correlation(deficit_frac, K) = {corr:.4f}")
    else:
        corr = np.nan

    return results, corr


# =============================================================================
# Comparison 3: Hardness-Conductivity Mirror Test
# =============================================================================

def comparison_3_mirror_test():
    """
    Fractional hardness excess vs fractional kappa deficit.

    If the CES 'mirror' holds, these should be anti-correlated with
    |slope| approximately 1.
    """
    print("\n" + "=" * 70)
    print("  COMPARISON 3: Hardness-Conductivity Mirror Test")
    print("=" * 70)

    results = []

    # Use ROM values as baseline for fractional excess/deficit
    print(f"\n  {'Alloy':<14s} {'J':>2s} {'HV_expt':>8s} {'HV_ROM':>7s} "
          f"{'HV_exc%':>8s} {'k_expt':>7s} {'k_ROM':>6s} {'k_def%':>7s}")
    print(f"  {'-'*65}")

    for symbols, label, J in CANTOR_SUBSYSTEMS:
        p = alloy_props(symbols)

        HV_expt = EXPT_HV_GPA[label]
        HV_rom = hardness_rom(symbols)
        HV_excess = (HV_expt - HV_rom) / HV_rom if HV_rom > 0 else np.nan

        k_expt = EXPT_KAPPA_WMK[label]
        k_rom = p['kappa_rom']
        k_deficit = (k_rom - k_expt) / k_rom

        results.append({
            'label': label,
            'J': J,
            'HV_expt': HV_expt,
            'HV_rom': HV_rom,
            'HV_excess': HV_excess,
            'k_expt': k_expt,
            'k_rom': k_rom,
            'k_deficit': k_deficit,
            'K': p['K'],
        })

        print(f"  {label:<14s} {J:2d} {HV_expt:8.2f} {HV_rom:7.2f} "
              f"{HV_excess*100:8.1f} {k_expt:7.1f} {k_rom:6.1f} "
              f"{k_deficit*100:7.1f}")

    # Fit slope of HV_excess vs k_deficit
    hv_exc = np.array([r['HV_excess'] for r in results])
    k_def = np.array([r['k_deficit'] for r in results])

    if len(hv_exc) > 1:
        # Linear regression: HV_excess = a * k_deficit + b
        A = np.vstack([k_def, np.ones(len(k_def))]).T
        slope, intercept = np.linalg.lstsq(A, hv_exc, rcond=None)[0]
        corr = np.corrcoef(hv_exc, k_def)[0, 1]

        print(f"\n  Linear fit: HV_excess = {slope:.3f} * k_deficit + {intercept:.3f}")
        print(f"  Correlation(HV_excess, k_deficit) = {corr:.4f}")
        print(f"  Mirror prediction: slope ~ -1, correlation ~ -1")
        print(f"  Observed |slope| = {abs(slope):.3f}")

        return results, slope, corr
    else:
        return results, np.nan, np.nan


# =============================================================================
# Comparison 4: Radiation Damage vs J
# =============================================================================

def comparison_4_radiation_damage():
    """
    Fit D(J) ~ 1/(1 + c*K(J)) to Zhang et al. 2015 data.
    Check steepest drop at J=1->2.
    """
    print("\n" + "=" * 70)
    print("  COMPARISON 4: Radiation Damage vs J")
    print("=" * 70)

    # Compute curvatures for each J
    # J=1: pure Ni -> K=0 (single component, H=1, 1-H=0)
    curvature_data = {}

    # J=1: pure Ni
    fracs_1 = np.array([1.0])
    radii_1 = np.array([ELEMENTS['Ni'].r])
    delta_1 = compute_delta(radii_1, fracs_1)
    q_1 = compute_q_from_delta(delta_1)
    H_1 = compute_H(fracs_1)
    K_1 = compute_K(q_1, H_1)
    curvature_data[1] = {'K': K_1, 'q': q_1, 'H': H_1, 'delta': delta_1,
                         'label': 'Ni', 'symbols': ['Ni']}

    # J=2..5 from Cantor subsystems
    for symbols, label, J in CANTOR_SUBSYSTEMS:
        p = alloy_props(symbols)
        curvature_data[J] = {
            'K': p['K'], 'q': p['q'], 'H': p['H'],
            'delta': p['delta'], 'label': label, 'symbols': symbols,
        }

    print(f"\n  {'J':>2s} {'Alloy':<14s} {'D(J)/D(1)':>10s} {'K':>8s} "
          f"{'q':>7s} {'delta%':>7s}")
    print(f"  {'-'*55}")

    Js = sorted(RAD_DAMAGE_DATA.keys())
    D_vals = np.array([RAD_DAMAGE_DATA[j] for j in Js])
    K_vals = np.array([curvature_data[j]['K'] for j in Js])

    for j in Js:
        d = RAD_DAMAGE_DATA[j]
        cd = curvature_data[j]
        print(f"  {j:2d} {cd['label']:<14s} {d:10.2f} {cd['K']:8.4f} "
              f"{cd['q']:7.3f} {cd['delta']*100:7.2f}")

    # Fit: D(J) = 1 / (1 + c*K(J))
    # => 1/D - 1 = c*K  =>  c = (1/D - 1) / K  for K > 0
    # Use least-squares: minimize sum( (D_i - 1/(1+c*K_i))^2 )
    # Scan c values
    c_scan = np.linspace(0.1, 50.0, 10000)
    best_c = None
    best_sse = np.inf

    for c in c_scan:
        D_pred = 1.0 / (1.0 + c * K_vals)
        sse = np.sum((D_vals - D_pred)**2)
        if sse < best_sse:
            best_sse = sse
            best_c = c

    # Compute R^2
    D_pred = 1.0 / (1.0 + best_c * K_vals)
    ss_res = np.sum((D_vals - D_pred)**2)
    ss_tot = np.sum((D_vals - np.mean(D_vals))**2)
    R2 = 1.0 - ss_res / ss_tot

    print(f"\n  Fit: D(J) = 1 / (1 + c * K(J))")
    print(f"  Optimal c = {best_c:.2f}")
    print(f"  R^2 = {R2:.4f}")

    print(f"\n  Predicted vs observed:")
    for j in Js:
        d_obs = RAD_DAMAGE_DATA[j]
        K = curvature_data[j]['K']
        d_pred = 1.0 / (1.0 + best_c * K)
        print(f"    J={j}: D_obs={d_obs:.2f}, D_pred={d_pred:.2f}, "
              f"residual={d_obs - d_pred:+.3f}")

    # Steepest drop check: J=1->2 vs J=2->3 etc.
    print(f"\n  Per-step drop in D(J):")
    drops = []
    for i in range(len(Js) - 1):
        drop = D_vals[i] - D_vals[i + 1]
        drops.append(drop)
        print(f"    J={Js[i]}->{Js[i+1]}: Delta_D = {drop:.2f}")

    steepest_idx = np.argmax(drops)
    steepest_step = f"J={Js[steepest_idx]}->{Js[steepest_idx+1]}"
    print(f"  Steepest drop at {steepest_step} (Delta_D = {drops[steepest_idx]:.2f})")
    steepest_at_1_to_2 = (steepest_idx == 0)
    print(f"  Per-element surplus prediction (steepest at J=1->2): "
          f"{'CONFIRMED' if steepest_at_1_to_2 else 'NOT confirmed'}")

    return {
        'Js': Js,
        'D_vals': D_vals,
        'K_vals': K_vals,
        'curvature_data': curvature_data,
        'best_c': best_c,
        'R2': R2,
        'D_pred': D_pred,
        'drops': drops,
        'steepest_at_1_to_2': steepest_at_1_to_2,
    }


# =============================================================================
# Comparison 5: VLGC Strengthening
# =============================================================================

def comparison_5_vlgc():
    """
    Compute VLGC volume-mismatch strengthening terms for Cantor subsystems.

    Two equivalent forms at equimolar:
      (A) Classical: sum_i c_i * (DeltaV_i)^2
      (B) CES:      K * Var_P[DeltaV]

    At equimolar, P_j = c_j, so they should agree numerically.
    """
    print("\n" + "=" * 70)
    print("  COMPARISON 5: VLGC Strengthening (Volume Mismatch)")
    print("=" * 70)

    results = []

    print(f"\n  {'Alloy':<14s} {'J':>2s} {'Sum c_i*DV^2':>13s} {'K*Var_P[DV]':>12s} "
          f"{'ratio':>7s} {'K':>7s}")
    print(f"  {'-'*60}")

    for symbols, label, J in CANTOR_SUBSYSTEMS:
        p = alloy_props(symbols)
        fracs = p['fracs']
        radii = p['radii']

        # Atomic volumes proportional to r^3
        V = radii**3  # pm^3 (proportional)

        # Mean volume
        V_bar = np.sum(fracs * V)

        # Volume deviations
        DeltaV = V - V_bar

        # (A) Classical form: sum c_i * (DeltaV_i)^2
        classical = np.sum(fracs * DeltaV**2)

        # (B) CES form: K * Var_P[DeltaV]
        # At equimolar, escort P_j = c_j (since all weights equal)
        q = p['q']
        K = p['K']

        # Compute escort distribution
        # For volume-based property, use V as the "property"
        P_escort = compute_escort(fracs, V, q)

        # Escort-weighted variance of DeltaV
        DeltaV_escort_mean = np.sum(P_escort * DeltaV)
        Var_P = np.sum(P_escort * (DeltaV - DeltaV_escort_mean)**2)
        ces_form = K * Var_P

        # At equimolar with equal properties, P_j ~ c_j
        # But V values differ, so P_j != c_j exactly
        # The ratio checks numerical relationship

        ratio = ces_form / classical if abs(classical) > 1e-20 else np.nan

        results.append({
            'label': label,
            'J': J,
            'classical': classical,
            'ces_form': ces_form,
            'ratio': ratio,
            'K': K,
            'P_escort': P_escort,
            'fracs': fracs,
            'DeltaV': DeltaV,
            'V': V,
        })

        print(f"  {label:<14s} {J:2d} {classical:13.1f} {ces_form:12.1f} "
              f"{ratio:7.4f} {K:7.4f}")

    # Detailed element-level check for Cantor alloy
    print(f"\n  Detailed element breakdown for CoCrFeMnNi:")
    cantor = results[-1]
    symbols_cantor = CANTOR_SUBSYSTEMS[-1][0]
    for i, sym in enumerate(symbols_cantor):
        r = ELEMENTS[sym].r
        V = cantor['V'][i]
        DV = cantor['DeltaV'][i]
        c = cantor['fracs'][i]
        P = cantor['P_escort'][i]
        print(f"    {sym}: r={r:.0f} pm, V~{V:.0f}, DeltaV={DV:+.0f}, "
              f"c={c:.3f}, P_escort={P:.4f}")

    # Check: at equimolar with identical properties, P_j = c_j exactly
    print(f"\n  Verification: escort = composition when properties equal?")
    fracs_eq = np.ones(5) / 5.0
    V_equal = np.ones(5) * 100.0
    P_test = compute_escort(fracs_eq, V_equal, 0.8)
    match = np.allclose(P_test, fracs_eq, atol=1e-10)
    print(f"    Equal V, q=0.8: P_escort = {P_test}, match c_j: {match}")

    # When V values differ, P_j != c_j, so the ratio != 1
    # But the CES form captures the escort-weighted physics
    print(f"\n  Note: ratio != 1 when volumes differ because escort P_j != c_j.")
    print(f"  The classical form uses composition weights; the CES form uses")
    print(f"  escort weights, which upweight elements with extreme volumes.")

    return results


# =============================================================================
# Plotting
# =============================================================================

def make_figure(res1, res2, corr2, res3, slope3, corr3, res4, res5):
    """Create 5-panel comparison figure."""

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('CES q-Thermodynamic Predictions vs Experiment\n'
                 '(Cantor Subsystems: NiFe, NiCoCr, NiCoFeCr, CoCrFeMnNi)',
                 fontsize=13, fontweight='bold', y=0.98)

    # Color scheme
    c_voigt = '#2196F3'   # blue
    c_reuss = '#FF9800'   # orange
    c_ces = '#4CAF50'     # green
    c_expt = '#E91E63'    # pink/red
    c_rom = '#9E9E9E'     # gray
    c_fit = '#673AB7'     # purple

    # -------------------------------------------------------------------------
    # Panel 1: Elastic Modulus Bands
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]

    labels = [r['label'] for r in res1]
    Js = [r['J'] for r in res1]
    x_pos = np.arange(len(labels))

    E_voigt = [r['band']['E_voigt'] for r in res1]
    E_reuss = [r['band']['E_reuss'] for r in res1]
    E_ces = [r['band']['E_ces'] for r in res1]
    E_expt = [r['E_expt'] for r in res1]

    bar_width = 0.18
    ax1.bar(x_pos - 1.5 * bar_width, E_reuss, bar_width, label='Reuss (q=-1)',
            color=c_reuss, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.bar(x_pos - 0.5 * bar_width, E_ces, bar_width, label='CES (q=q*)',
            color=c_ces, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.bar(x_pos + 0.5 * bar_width, E_voigt, bar_width, label='Voigt (q=1)',
            color=c_voigt, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.bar(x_pos + 1.5 * bar_width, E_expt, bar_width, label='Experiment',
            color=c_expt, alpha=0.8, edgecolor='black', linewidth=0.5)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Young's Modulus (GPa)")
    ax1.set_title('(a) Elastic Modulus: CES Bands vs Expt', fontsize=10)
    ax1.legend(fontsize=7, loc='upper left')
    ax1.set_ylim(150, 280)
    ax1.grid(axis='y', alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel 2: Thermal Conductivity vs J
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]

    Js_tc = [r['J'] for r in res2]
    k_rom_vals = [r['kappa_rom'] for r in res2]
    k_expt_vals = [r['kappa_expt'] for r in res2]

    # Add average pure-element kappa at J=1
    k_pure_avg = np.mean(list(PURE_KAPPAS.values()))

    Js_plot = [1] + Js_tc
    k_rom_plot = [k_pure_avg] + k_rom_vals
    k_expt_plot = [k_pure_avg] + k_expt_vals  # J=1 trivially equal

    ax2.plot(Js_plot, k_rom_plot, 's--', color=c_rom, markersize=8,
             label='ROM (rule of mixtures)', linewidth=1.5)
    ax2.plot(Js_plot, k_expt_plot, 'o-', color=c_expt, markersize=8,
             label='Experiment (Jin 2016)', linewidth=2)

    # Shade deficit region
    ax2.fill_between(Js_plot, k_expt_plot, k_rom_plot, alpha=0.15,
                     color=c_ces, label='Conductivity deficit')

    ax2.set_xlabel('Number of components J')
    ax2.set_ylabel('Thermal Conductivity (W/mK)')
    ax2.set_title('(b) Thermal Conductivity vs J', fontsize=10)
    ax2.set_xticks([1, 2, 3, 4, 5])
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)

    # Annotate correlation
    ax2.text(0.97, 0.05, f'Corr(deficit, K) = {corr2:.3f}',
             transform=ax2.transAxes, fontsize=8, ha='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    # -------------------------------------------------------------------------
    # Panel 3: Deficit fraction vs Curvature K
    # -------------------------------------------------------------------------
    ax3 = axes[0, 2]

    deficit_fracs = [r['deficit_frac'] for r in res2]
    K_vals_tc = [r['K'] for r in res2]

    ax3.scatter(K_vals_tc, deficit_fracs, s=100, c=c_ces, edgecolors='black',
                zorder=5, linewidths=1.0)

    # Label each point
    for r in res2:
        ax3.annotate(r['label'], (r['K'], r['deficit_frac']),
                     textcoords="offset points", xytext=(5, 8),
                     fontsize=7)

    # Fit line
    K_arr = np.array(K_vals_tc)
    d_arr = np.array(deficit_fracs)
    if len(K_arr) > 1:
        coeffs = np.polyfit(K_arr, d_arr, 1)
        K_fit = np.linspace(0, max(K_arr) * 1.1, 100)
        ax3.plot(K_fit, np.polyval(coeffs, K_fit), '--', color=c_fit,
                 linewidth=1.5, label=f'Linear fit (R={corr2:.3f})')

    ax3.set_xlabel('Curvature K = (1-q)(1-H)')
    ax3.set_ylabel('Conductivity deficit fraction')
    ax3.set_title('(c) Deficit Correlates with Curvature', fontsize=10)
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3)
    ax3.set_xlim(left=0)
    ax3.set_ylim(bottom=0)

    # -------------------------------------------------------------------------
    # Panel 4: Hardness-Conductivity Mirror
    # -------------------------------------------------------------------------
    ax4 = axes[1, 0]

    hv_exc = [r['HV_excess'] for r in res3]
    k_def = [r['k_deficit'] for r in res3]

    ax4.scatter(k_def, hv_exc, s=100, c=c_expt, edgecolors='black',
                zorder=5, linewidths=1.0)

    for r in res3:
        ax4.annotate(r['label'], (r['k_deficit'], r['HV_excess']),
                     textcoords="offset points", xytext=(5, 8), fontsize=7)

    # Plot mirror line: HV_excess = -k_deficit (shifted)
    k_def_arr = np.array(k_def)
    hv_exc_arr = np.array(hv_exc)
    if not np.isnan(slope3):
        k_fit_range = np.linspace(min(k_def_arr) * 0.9, max(k_def_arr) * 1.1, 100)
        coeffs_m = np.polyfit(k_def_arr, hv_exc_arr, 1)
        ax4.plot(k_fit_range, np.polyval(coeffs_m, k_fit_range), '--',
                 color=c_fit, linewidth=1.5,
                 label=f'Fit slope={slope3:.2f}, r={corr3:.2f}')

    ax4.set_xlabel('Fractional conductivity deficit')
    ax4.set_ylabel('Fractional hardness excess')
    ax4.set_title('(d) Hardness-Conductivity Mirror', fontsize=10)
    ax4.legend(fontsize=7)
    ax4.grid(alpha=0.3)
    ax4.axhline(0, color='gray', linewidth=0.5)

    # -------------------------------------------------------------------------
    # Panel 5: Radiation Damage vs J
    # -------------------------------------------------------------------------
    ax5 = axes[1, 1]

    Js_rad = res4['Js']
    D_obs = res4['D_vals']
    D_pred = res4['D_pred']
    K_rad = res4['K_vals']
    c_opt = res4['best_c']

    ax5.plot(Js_rad, D_obs, 'o-', color=c_expt, markersize=9,
             linewidth=2, label='Zhang et al. 2015', zorder=5)

    # Smooth prediction curve
    J_smooth = np.linspace(1, 5, 100)
    # Interpolate K values for smooth curve
    K_smooth = np.interp(J_smooth, Js_rad, K_rad)
    D_smooth = 1.0 / (1.0 + c_opt * K_smooth)
    ax5.plot(J_smooth, D_smooth, '--', color=c_ces, linewidth=2,
             label=f'Fit: 1/(1+{c_opt:.1f}K)')

    ax5.scatter(Js_rad, D_pred, marker='s', s=80, color=c_ces,
                edgecolors='black', zorder=4, label='Predicted')

    ax5.set_xlabel('Number of components J')
    ax5.set_ylabel('Normalized defect density D(J)/D(1)')
    ax5.set_title('(e) Radiation Damage vs J', fontsize=10)
    ax5.set_xticks([1, 2, 3, 4, 5])
    ax5.legend(fontsize=7)
    ax5.grid(alpha=0.3)

    ax5.text(0.97, 0.55, f'R$^2$ = {res4["R2"]:.4f}',
             transform=ax5.transAxes, fontsize=9, ha='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    # Mark steepest drop
    ax5.annotate('Steepest drop\n(per-element surplus)', xy=(1.5, 0.775),
                 fontsize=7, ha='center', color='#333333',
                 arrowprops=dict(arrowstyle='->', color='#333333'),
                 xytext=(2.5, 0.9))

    # -------------------------------------------------------------------------
    # Panel 6: VLGC Strengthening
    # -------------------------------------------------------------------------
    ax6 = axes[1, 2]

    labels_v = [r['label'] for r in res5]
    classical = [r['classical'] for r in res5]
    ces_vals = [r['ces_form'] for r in res5]

    x_pos_v = np.arange(len(labels_v))
    bar_w = 0.3

    ax6.bar(x_pos_v - bar_w / 2, classical, bar_w,
            label=r'Classical $\Sigma c_i \Delta V_i^2$',
            color=c_voigt, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax6.bar(x_pos_v + bar_w / 2, ces_vals, bar_w,
            label=r'CES: $K \cdot \mathrm{Var}_P[\Delta V]$',
            color=c_ces, alpha=0.8, edgecolor='black', linewidth=0.5)

    ax6.set_xticks(x_pos_v)
    ax6.set_xticklabels(labels_v, fontsize=8)
    ax6.set_ylabel(r'Volume mismatch parameter (pm$^6$)')
    ax6.set_title('(f) VLGC Strengthening: Classical vs CES', fontsize=10)
    ax6.legend(fontsize=7)
    ax6.grid(axis='y', alpha=0.3)

    # Annotate ratios
    for i, r in enumerate(res5):
        ax6.text(x_pos_v[i], max(r['classical'], r['ces_form']) * 1.02,
                 f'ratio={r["ratio"]:.3f}', ha='center', fontsize=7,
                 color='#333333')

    # -------------------------------------------------------------------------
    # Final adjustments
    # -------------------------------------------------------------------------
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    return fig


# =============================================================================
# Summary Table
# =============================================================================

def print_summary(res1, res2, corr2, res3, slope3, corr3, res4, res5):
    """Print consolidated summary table."""

    print("\n" + "=" * 70)
    print("  CONSOLIDATED SUMMARY")
    print("=" * 70)

    print(f"\n  Comparison 1 (Elastic Modulus):")
    for r in res1:
        b = r['band']
        E_ex = r['E_expt']
        in_band = b['E_reuss'] <= E_ex <= b['E_voigt']
        print(f"    {r['label']:<14s}: [{b['E_reuss']:.0f}, {b['E_voigt']:.0f}] GPa "
              f"contains E_expt={E_ex:.0f} GPa: {in_band}")

    print(f"\n  Comparison 2 (Thermal Conductivity):")
    print(f"    Correlation(deficit_frac, K) = {corr2:.4f}")
    for r in res2:
        print(f"    {r['label']:<14s}: ROM={r['kappa_rom']:.1f}, "
              f"expt={r['kappa_expt']:.1f}, deficit={r['deficit_frac']*100:.1f}%")

    print(f"\n  Comparison 3 (Hardness-Conductivity Mirror):")
    print(f"    Slope = {slope3:.3f} (mirror predicts ~ -1)")
    print(f"    Correlation = {corr3:.4f}")

    print(f"\n  Comparison 4 (Radiation Damage):")
    print(f"    Fit: D(J) = 1/(1 + {res4['best_c']:.1f} * K)")
    print(f"    R^2 = {res4['R2']:.4f}")
    print(f"    Steepest drop at J=1->2: {res4['steepest_at_1_to_2']}")

    print(f"\n  Comparison 5 (VLGC Strengthening):")
    for r in res5:
        print(f"    {r['label']:<14s}: classical={r['classical']:.1f}, "
              f"CES={r['ces_form']:.1f}, ratio={r['ratio']:.4f}")

    print()


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  HEA PROPERTY COMPARISON: CES PREDICTIONS vs EXPERIMENT")
    print("=" * 70)
    print("  Cantor subsystems: NiFe, NiCoCr, NiCoFeCr, CoCrFeMnNi")
    print("  Framework: q-thermodynamic CES partition function")
    print()

    # Run all comparisons
    res1 = comparison_1_elastic_modulus()
    res2, corr2 = comparison_2_thermal_conductivity()
    res3, slope3, corr3 = comparison_3_mirror_test()
    res4 = comparison_4_radiation_damage()
    res5 = comparison_5_vlgc()

    # Print consolidated summary
    print_summary(res1, res2, corr2, res3, slope3, corr3, res4, res5)

    # Generate figure
    print("  Generating figure...")
    fig = make_figure(res1, res2, corr2, res3, slope3, corr3, res4, res5)

    # Save figure
    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, 'property_comparison.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Figure saved to: {fig_path}")
    plt.close(fig)

    print("\n  Done.")
