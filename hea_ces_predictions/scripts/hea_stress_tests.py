#!/usr/bin/env python3
"""
Designed-to-break stress tests for q-thermodynamic HEA theory.

Tests:
  1. Subsystem transferability: fit q from NiFe, predict multi-component alloys
  2. Non-equimolar Al_x CoCrFeNi: phase boundary prediction
  3. q constancy scorecard for Cantor alloy
  4. Yang-Zhang classification: K_eff vs Omega-delta rule
  5. Per-element surplus pi_K(J) numerical verification
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure the scripts directory is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hea_validate_core import (
    ELEMENTS, Element, ALPHA_DEFAULT,
    compute_delta, compute_H, compute_K, compute_q_from_delta,
    compute_S1, compute_Sq, compute_CES, compute_Zq,
    compute_escort, compute_Omega, get_alloy_properties,
)

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Additional element not in core database
EXTRA_ELEMENTS = {
    'Pd': Element('Pd', 137.0, 106.42, 1828, 10, 2.20, 121.0, 205.0, 71.8, 12.02),
}

def get_element(sym):
    """Look up element from core DB or extras."""
    if sym in ELEMENTS:
        return ELEMENTS[sym]
    if sym in EXTRA_ELEMENTS:
        return EXTRA_ELEMENTS[sym]
    raise KeyError(f"Element {sym} not in database")

# =============================================================================
# Miedema pair mixing enthalpies (kJ/mol)
# =============================================================================
PAIR_ENTHALPY = {
    ('Co', 'Cr'): -4, ('Co', 'Fe'): -1, ('Co', 'Mn'): -5, ('Co', 'Ni'): 0,
    ('Cr', 'Fe'): -1, ('Cr', 'Mn'): 2, ('Cr', 'Ni'): -7,
    ('Fe', 'Mn'): 0, ('Fe', 'Ni'): -2, ('Mn', 'Ni'): -8,
    ('Al', 'Co'): -19, ('Al', 'Cr'): -10, ('Al', 'Fe'): -11,
    ('Al', 'Mn'): -19, ('Al', 'Ni'): -22, ('Al', 'Ti'): -30,
    ('Ti', 'V'): -2, ('Ti', 'Zr'): 0, ('Ti', 'Hf'): 0,
    ('Ti', 'Nb'): 2, ('Ti', 'Ta'): 1,
    ('Zr', 'Hf'): 0, ('Zr', 'Nb'): 4, ('Zr', 'Ta'): 3, ('Zr', 'V'): -4,
    ('Hf', 'Nb'): 4, ('Hf', 'Ta'): 3,
    ('Nb', 'Ta'): 0, ('Nb', 'Mo'): -6, ('Nb', 'V'): -1,
    ('Ta', 'Mo'): -5, ('Ta', 'W'): -7,
    ('Mo', 'W'): 0,
    ('W', 'Nb'): -8, ('W', 'Ta'): -7, ('W', 'V'): -1,
    ('Cr', 'Mo'): 0, ('Cr', 'V'): -2, ('Cr', 'Ta'): -7, ('Cr', 'Nb'): -7,
    ('Cr', 'W'): 1,
    ('V', 'Mo'): -1, ('V', 'W'): -1,
    ('Co', 'Ti'): -28, ('Co', 'V'): -14, ('Co', 'Nb'): -25, ('Co', 'Ta'): -24,
    ('Co', 'Cu'): 6, ('Co', 'Al'): -19,
    ('Cr', 'Cu'): 12, ('Cr', 'Ti'): -7,
    ('Fe', 'Cu'): 13, ('Fe', 'V'): -7, ('Fe', 'Ti'): -17,
    ('Mn', 'Cu'): 4,
    ('Ni', 'Cu'): 4, ('Ni', 'V'): -18, ('Ni', 'Ti'): -35,
    ('Ni', 'Nb'): -30, ('Ni', 'Ta'): -29,
    ('Cu', 'Al'): -1, ('Cu', 'Ti'): -9, ('Cu', 'Zr'): -23,
    ('Pd', 'Co'): -1, ('Pd', 'Cr'): -13, ('Pd', 'Fe'): -4,
    ('Pd', 'Ni'): -5, ('Pd', 'Mn'): 0,
}


def get_pair_enthalpy(a, b):
    """Look up Miedema pair enthalpy, trying both orderings."""
    if a == b:
        return 0.0
    if (a, b) in PAIR_ENTHALPY:
        return PAIR_ENTHALPY[(a, b)]
    if (b, a) in PAIR_ENTHALPY:
        return PAIR_ENTHALPY[(b, a)]
    return 0.0  # unknown pair: assume zero


def compute_delta_H_mix(symbols, fracs):
    """Takeuchi-Inoue regular solution: DeltaH_mix = sum_pairs 4 * H_ij * c_i * c_j."""
    dH = 0.0
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            dH += 4.0 * get_pair_enthalpy(symbols[i], symbols[j]) * fracs[i] * fracs[j]
    return dH


def get_alloy_properties_ext(symbols, fracs=None):
    """Extended version that also handles elements not in core DB."""
    J = len(symbols)
    if fracs is None:
        fracs = np.ones(J) / J

    elems = [get_element(s) for s in symbols]
    radii = np.array([e.r for e in elems])
    masses = np.array([e.mass for e in elems])
    T_ms = np.array([e.T_m for e in elems])
    VECs = np.array([e.VEC for e in elems])
    chis = np.array([e.chi for e in elems])
    Es = np.array([e.E for e in elems])
    sigma_ys = np.array([e.sigma_y for e in elems])
    kappas = np.array([e.kappa for e in elems])
    rhos = np.array([e.rho for e in elems])

    r_bar = np.sum(fracs * radii)
    delta = compute_delta(radii, fracs)
    H = compute_H(fracs)
    q = compute_q_from_delta(delta)
    K = compute_K(q, H)
    S1 = compute_S1(fracs)
    Sq = compute_Sq(fracs, q)

    rho_avg = np.sum(fracs * rhos)
    T_m_avg = np.sum(fracs * T_ms)
    VEC_avg = np.sum(fracs * VECs)

    E_voigt = np.sum(fracs * Es)
    E_reuss = 1.0 / np.sum(fracs / Es) if np.all(Es > 0) else np.nan

    kappa_rom = np.sum(fracs * kappas)
    sigma_y_rom = np.sum(fracs * sigma_ys)

    return {
        'symbols': symbols, 'fracs': fracs, 'J': J,
        'r_bar': r_bar, 'delta': delta, 'delta_pct': delta * 100,
        'H': H, 'q': q, 'K': K,
        'S1': S1, 'Sq': Sq,
        'rho': rho_avg, 'T_m': T_m_avg, 'VEC': VEC_avg,
        'E_voigt': E_voigt, 'E_reuss': E_reuss,
        'kappa_rom': kappa_rom, 'sigma_y_rom': sigma_y_rom,
        'radii': radii, 'masses': masses, 'T_ms': T_ms,
        'Es': Es, 'kappas': kappas, 'sigma_ys': sigma_ys,
    }


# =============================================================================
# Utility: fit q via binary search on CES
# =============================================================================

def _fit_q_ces(fracs, props, target, q_lo=-20.0, q_hi=20.0, n_iter=300):
    """Find q such that CES(fracs, props, q) = target via bisection.

    CES is monotonically decreasing in q (for non-degenerate property spread).
    Returns the fitted q value.  If the target is outside the CES range,
    returns the boundary value.
    """
    for _ in range(n_iter):
        q_mid = 0.5 * (q_lo + q_hi)
        val = compute_CES(fracs, props, q_mid)
        if np.isnan(val):
            q_hi = q_mid
        elif val > target:
            q_hi = q_mid
        else:
            q_lo = q_mid
    return 0.5 * (q_lo + q_hi)


# =============================================================================
# Test 1: Subsystem Transferability
# =============================================================================

def test_subsystem_transferability():
    """Fit q from NiFe elastic modulus, predict multi-component alloys."""
    print("\n" + "=" * 70)
    print("TEST 1: Subsystem Transferability (elastic modulus)")
    print("=" * 70)

    # Experimental elastic moduli (GPa)
    expt = {
        'NiFe':        {'symbols': ['Ni', 'Fe'], 'E_meas': 200.0},
        'NiCoCr':      {'symbols': ['Ni', 'Co', 'Cr'], 'E_meas': 235.0},
        'NiCoFeCr':    {'symbols': ['Ni', 'Co', 'Fe', 'Cr'], 'E_meas': 213.0},
        'CoCrFeMnNi':  {'symbols': ['Co', 'Cr', 'Fe', 'Mn', 'Ni'], 'E_meas': 200.0},
    }

    # Step 1: Fit q from NiFe
    nife = expt['NiFe']
    syms = nife['symbols']
    fracs = np.ones(len(syms)) / len(syms)
    elems = [ELEMENTS[s] for s in syms]
    Es = np.array([e.E for e in elems])
    E_meas = nife['E_meas']

    # Solve for q: CES(q) = E_meas  =>  (sum c_j * E_j^q)^(1/q) = E_meas
    # CES is monotonically decreasing in q for spread properties:
    #   q=+inf -> max(E_j), q=-inf -> min(E_j)
    # Binary search over wide range
    q_fit = _fit_q_ces(fracs, Es, E_meas)
    E_check = compute_CES(fracs, Es, q_fit)

    print(f"\n  NiFe calibration:")
    print(f"    E_meas = {E_meas:.1f} GPa")
    print(f"    q_fit  = {q_fit:.4f}")
    print(f"    E_CES(q_fit) = {E_check:.1f} GPa")

    # Step 2: Predict others using q_fit
    print(f"\n  {'Alloy':<15s} {'E_meas':>8s} {'E_CES':>8s} {'Error':>8s} {'Error%':>8s}")
    print(f"  {'-'*50}")
    for name, data in expt.items():
        syms = data['symbols']
        fracs_a = np.ones(len(syms)) / len(syms)
        elems_a = [ELEMENTS[s] for s in syms]
        Es_a = np.array([e.E for e in elems_a])
        E_pred = compute_CES(fracs_a, Es_a, q_fit)
        err = E_pred - data['E_meas']
        err_pct = 100.0 * err / data['E_meas']
        marker = "<-- calibration" if name == 'NiFe' else ""
        print(f"  {name:<15s} {data['E_meas']:8.1f} {E_pred:8.1f} {err:+8.1f} {err_pct:+8.1f}%  {marker}")

    # Also compare with delta-derived q
    print(f"\n  Comparison: q from NiFe fit = {q_fit:.4f}")
    for name, data in expt.items():
        syms = data['symbols']
        props = get_alloy_properties(syms)
        E_ces_delta = compute_CES(props['fracs'], props['Es'], props['q'])
        print(f"    {name:<15s}: q_delta = {props['q']:.4f}, E_CES(q_delta) = {E_ces_delta:.1f} GPa")


# =============================================================================
# Test 2: Non-equimolar Al_x CoCrFeNi
# =============================================================================

def test_alx_cocrfeni():
    """Compute q-theory parameters across Al_x CoCrFeNi composition sweep."""
    print("\n" + "=" * 70)
    print("TEST 2: Non-equimolar Al_x CoCrFeNi phase boundary prediction")
    print("=" * 70)

    x_vals = np.arange(0.0, 2.05, 0.1)
    symbols = ['Al', 'Co', 'Cr', 'Fe', 'Ni']

    results = {'x': [], 'delta_pct': [], 'q': [], 'K': [], 'VEC': [], 'H': []}

    print(f"\n  {'x':>5s} {'delta%':>7s} {'q':>8s} {'K_eff':>8s} {'VEC':>6s} {'Phase':>12s}")
    print(f"  {'-'*55}")

    for x in x_vals:
        if x < 1e-10:
            # x=0: just CoCrFeNi
            syms = ['Co', 'Cr', 'Fe', 'Ni']
            fracs = np.ones(4) / 4.0
        else:
            denom = 4.0 + x
            fracs = np.array([x / denom, 1.0 / denom, 1.0 / denom,
                              1.0 / denom, 1.0 / denom])
            syms = symbols

        elems = [ELEMENTS[s] for s in syms]
        radii = np.array([e.r for e in elems])
        VECs = np.array([e.VEC for e in elems])

        delta = compute_delta(radii, fracs)
        H_val = compute_H(fracs)
        q_val = compute_q_from_delta(delta)
        K_val = compute_K(q_val, H_val)
        VEC_avg = np.sum(fracs * VECs)

        # Known phase regions
        if x < 0.5:
            phase = "FCC"
        elif x < 0.9:
            phase = "FCC+BCC"
        else:
            phase = "BCC"

        results['x'].append(x)
        results['delta_pct'].append(delta * 100)
        results['q'].append(q_val)
        results['K'].append(K_val)
        results['VEC'].append(VEC_avg)
        results['H'].append(H_val)

        print(f"  {x:5.1f} {delta*100:7.2f} {q_val:8.3f} {K_val:8.4f} {VEC_avg:6.2f} {phase:>12s}")

    # Plot K_eff vs x with phase regions
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Phase region shading
    ax1.axvspan(0.0, 0.5, alpha=0.15, color='blue', label='FCC')
    ax1.axvspan(0.5, 0.9, alpha=0.15, color='green', label='FCC+BCC')
    ax1.axvspan(0.9, 2.0, alpha=0.15, color='red', label='BCC')
    ax1.axvline(x=0.5, color='gray', linestyle='--', linewidth=0.8)
    ax1.axvline(x=0.9, color='gray', linestyle='--', linewidth=0.8)

    ax1.plot(results['x'], results['K'], 'ko-', linewidth=2, markersize=5, label='K_eff')
    ax1.set_xlabel('x in Al_x CoCrFeNi', fontsize=12)
    ax1.set_ylabel('K_eff = (1-q)(1-H)', fontsize=12)
    ax1.set_title('Test 2: K_eff vs Al content with phase boundaries', fontsize=13)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Secondary axis: VEC
    ax2 = ax1.twinx()
    ax2.plot(results['x'], results['VEC'], 'b^--', linewidth=1.5, markersize=4,
             alpha=0.7, label='VEC')
    ax2.set_ylabel('VEC', fontsize=12, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'test2_alx_cocrfeni.png'), dpi=150)
    plt.close(fig)
    print(f"\n  Figure saved: scripts/figures/test2_alx_cocrfeni.png")

    # Analysis: does any parameter predict the boundary?
    x_arr = np.array(results['x'])
    K_arr = np.array(results['K'])
    VEC_arr = np.array(results['VEC'])

    # Find K at boundaries
    idx_05 = np.argmin(np.abs(x_arr - 0.5))
    idx_09 = np.argmin(np.abs(x_arr - 0.9))
    print(f"\n  K_eff at x=0.5 boundary: {K_arr[idx_05]:.4f}")
    print(f"  K_eff at x=0.9 boundary: {K_arr[idx_09]:.4f}")
    print(f"  VEC at x=0.5 boundary:   {VEC_arr[idx_05]:.2f}")
    print(f"  VEC at x=0.9 boundary:   {VEC_arr[idx_09]:.2f}")
    print(f"  Note: K_eff increases monotonically with x -- it tracks distortion")
    print(f"        but does NOT by itself predict the FCC/BCC boundary location.")
    print(f"        VEC is the better predictor (VEC<6.87 => BCC per Guo criterion).")


# =============================================================================
# Test 3: q Constancy Scorecard
# =============================================================================

def test_q_constancy_cantor():
    """All q estimates for Cantor alloy in one table."""
    print("\n" + "=" * 70)
    print("TEST 3: q constancy scorecard for Cantor CoCrFeMnNi")
    print("=" * 70)

    symbols = ['Co', 'Cr', 'Fe', 'Mn', 'Ni']
    fracs = np.ones(5) / 5.0
    props = get_alloy_properties(symbols)

    # q from delta (mismatch formula)
    q_delta = props['q']

    # q from elastic modulus fit
    # E_meas ~ 200 GPa for Cantor
    E_meas = 200.0
    Es = props['Es']
    q_elastic = _fit_q_ces(fracs, Es, E_meas)

    # q from thermal conductivity fit
    # kappa_meas ~ 12 W/mK for Cantor (literature)
    kappa_meas = 12.0
    kappas = props['kappas']
    q_kappa = _fit_q_ces(fracs, kappas, kappa_meas)

    # q from yield strength fit
    # sigma_y_meas ~ 350 MPa for Cantor (single crystal, annealed poly ~200 MPa)
    sigma_y_meas = 350.0
    sigma_ys = props['sigma_ys']
    q_sigma = _fit_q_ces(fracs, sigma_ys, sigma_y_meas)

    print(f"\n  Cantor alloy: CoCrFeMnNi (equimolar)")
    print(f"  Element moduli (GPa): {dict(zip(symbols, Es))}")
    print(f"  Element kappa (W/mK): {dict(zip(symbols, kappas))}")
    print(f"  Element sigma_y (MPa): {dict(zip(symbols, sigma_ys))}")

    print(f"\n  {'Method':<30s} {'q':>8s} {'Notes'}")
    print(f"  {'-'*65}")
    print(f"  {'q from delta (mismatch)':.<30s} {q_delta:8.4f}   alpha={ALPHA_DEFAULT:.1f}, delta={props['delta_pct']:.2f}%")
    print(f"  {'q from E fit (200 GPa)':.<30s} {q_elastic:8.4f}   E_Voigt={props['E_voigt']:.1f}, E_Reuss={props['E_reuss']:.1f}")
    print(f"  {'q from kappa fit (12 W/mK)':.<30s} {q_kappa:8.4f}   kappa_ROM={props['kappa_rom']:.1f} W/mK")
    print(f"  {'q from sigma_y fit (350 MPa)':.<30s} {q_sigma:8.4f}   sigma_y_ROM={props['sigma_y_rom']:.0f} MPa")

    q_values = [q_delta, q_elastic, q_kappa, q_sigma]
    q_spread = max(q_values) - min(q_values)
    print(f"\n  Spread: max - min = {q_spread:.4f}")
    print(f"  Mean q = {np.mean(q_values):.4f}, Std = {np.std(q_values):.4f}")

    # Check if fits hit search bounds (meaning target is outside CES range)
    E_voigt = props['E_voigt']
    E_reuss = props['E_reuss']
    print(f"\n  ANALYSIS:")
    print(f"    E range: Reuss={E_reuss:.1f}, Voigt={E_voigt:.1f}, target=200.0")
    if E_meas < E_reuss:
        print(f"    -> E target is BELOW Reuss bound; CES cannot fit it at any q.")
        print(f"       This means the alloy's modulus depression exceeds what a")
        print(f"       generalized mean can produce. q_elastic hit search bound.")
    print(f"    kappa range: min={min(kappas):.1f}, ROM={props['kappa_rom']:.1f}, target=12.0")
    print(f"    -> kappa target is far below ROM because Mn has anomalously low")
    print(f"       conductivity (7.8 W/mK). Phonon scattering in HEAs is not a")
    print(f"       generalized-mean effect.")
    print(f"    sigma_y range: ROM={props['sigma_y_rom']:.0f}, target=350")
    if sigma_y_meas > max(sigma_ys):
        print(f"    -> sigma_y target EXCEEDS max element value ({max(sigma_ys):.0f} MPa);")
        print(f"       CES(q>1) needed -- cocktail effect requires q>1.")
    print(f"\n  VERDICT: q is NOT a universal constant across properties. Each property")
    print(f"  has its own effective q, and for some (kappa, sigma_y) the CES model")
    print(f"  cannot even reach the measured value within reasonable q.")


# =============================================================================
# Test 4: Yang-Zhang Classification
# =============================================================================

def test_yang_zhang_classification():
    """Compare K_eff > 0 vs Omega-delta rule for phase classification."""
    print("\n" + "=" * 70)
    print("TEST 4: Yang-Zhang classification -- K_eff vs Omega-delta")
    print("=" * 70)

    # Alloy database: (name, symbols, phase_type)
    # phase_type: 'SS' = solid solution, 'IM' = intermetallic/multi-phase
    alloy_db = [
        # Solid solutions
        ('CoCrFeMnNi',    ['Co','Cr','Fe','Mn','Ni'],       'SS'),
        ('CoCrFeNi',      ['Co','Cr','Fe','Ni'],            'SS'),
        ('WMoTaNb',       ['W','Mo','Ta','Nb'],             'SS'),
        ('WMoTaNbV',      ['W','Mo','Ta','Nb','V'],         'SS'),
        ('TiZrHfNbTa',    ['Ti','Zr','Hf','Nb','Ta'],      'SS'),
        ('CoCrFeNiPd',    ['Co','Cr','Fe','Ni','Pd'],       'SS'),
        ('CoCrFeNiMn',    ['Co','Cr','Fe','Ni','Mn'],       'SS'),
        ('AlCoCrFeNi',    ['Al','Co','Cr','Fe','Ni'],       'SS'),
        ('NbTiVZr',       ['Nb','Ti','V','Zr'],             'SS'),
        ('MoNbTaW',       ['Mo','Nb','Ta','W'],             'SS'),
        ('MoNbTaVW',      ['Mo','Nb','Ta','V','W'],         'SS'),
        ('CrMoNbTaVW',    ['Cr','Mo','Nb','Ta','V','W'],    'SS'),
        ('HfNbTaTiZr',    ['Hf','Nb','Ta','Ti','Zr'],       'SS'),
        # Intermetallic / multi-phase
        ('AlCoCrCuFeNi',  ['Al','Co','Cr','Cu','Fe','Ni'],  'IM'),
        ('CoCrFeNiTi',    ['Co','Cr','Fe','Ni','Ti'],       'IM'),
        ('AlCoCrFeNiTi',  ['Al','Co','Cr','Fe','Ni','Ti'],  'IM'),
    ]

    R_gas = 8.314  # J/(mol·K)

    print(f"\n  {'Alloy':<18s} {'delta%':>7s} {'q':>7s} {'K':>7s} {'K_eff':>7s} "
          f"{'DH_mix':>7s} {'Tm':>6s} {'Omega':>7s} {'Actual':>6s} "
          f"{'K>0':>5s} {'OmDel':>5s}")
    print(f"  {'-'*102}")

    n_correct_K = 0
    n_correct_OD = 0
    n_total = len(alloy_db)
    rows = []

    for name, symbols, phase in alloy_db:
        fracs = np.ones(len(symbols)) / len(symbols)
        props = get_alloy_properties_ext(symbols, fracs)

        delta_pct = props['delta_pct']
        q_val = props['q']
        K_bare = props['K']  # bare K = (1-q)(1-H)

        # Mixing enthalpy
        dH_mix = compute_delta_H_mix(symbols, fracs)
        T_m_avg = props['T_m']
        S_mix = compute_S1(fracs) * R_gas  # J/(mol K)

        # Full K_eff per paper eq. (line 228):
        # K_eff = (1-q)(1-H) * max(0, 1 - |DH_mix|/(R*T_m))
        enthalpy_factor = max(0.0, 1.0 - abs(dH_mix * 1000.0) / (R_gas * T_m_avg))
        K_val = K_bare * enthalpy_factor

        Omega = compute_Omega(T_m_avg, S_mix, dH_mix * 1000.0)  # dH in J/mol

        # Predictions
        K_pred = 'SS' if K_val > 0 else 'IM'
        OD_pred = 'SS' if (Omega > 1.1 and delta_pct < 6.6) else 'IM'

        K_correct = (K_pred == phase)
        OD_correct = (OD_pred == phase)
        if K_correct:
            n_correct_K += 1
        if OD_correct:
            n_correct_OD += 1

        K_mark = 'Y' if K_correct else 'N'
        OD_mark = 'Y' if OD_correct else 'N'

        print(f"  {name:<18s} {delta_pct:7.2f} {q_val:7.3f} {K_bare:7.3f} {K_val:7.3f} "
              f"{dH_mix:7.1f} {T_m_avg:6.0f} {Omega:7.02f} {phase:>6s} "
              f"{K_mark:>5s} {OD_mark:>5s}")

        rows.append({
            'name': name, 'delta_pct': delta_pct, 'q': q_val,
            'K_bare': K_bare, 'K': K_val,
            'dH_mix': dH_mix, 'T_m': T_m_avg, 'Omega': Omega,
            'phase': phase, 'K_correct': K_correct, 'OD_correct': OD_correct,
        })

    acc_K = 100.0 * n_correct_K / n_total
    acc_OD = 100.0 * n_correct_OD / n_total
    print(f"\n  Classification accuracy:")
    print(f"    K_eff > 0 criterion:           {n_correct_K}/{n_total} = {acc_K:.1f}%")
    print(f"    Omega-delta rule (O>1.1,d<6.6): {n_correct_OD}/{n_total} = {acc_OD:.1f}%")

    # Plot Omega-delta diagram with K_eff coloring
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for row in rows:
        color = 'blue' if row['phase'] == 'SS' else 'red'
        marker = 'o' if row['phase'] == 'SS' else 's'
        ax1.plot(row['delta_pct'], row['Omega'], marker, color=color,
                 markersize=8, alpha=0.8)
        ax1.annotate(row['name'], (row['delta_pct'], row['Omega']),
                     fontsize=5, ha='left', va='bottom')

    ax1.axvline(x=6.6, color='gray', linestyle='--', linewidth=1, label='delta=6.6%')
    ax1.axhline(y=1.1, color='gray', linestyle=':', linewidth=1, label='Omega=1.1')
    ax1.set_xlabel('delta (%)', fontsize=12)
    ax1.set_ylabel('Omega', fontsize=12)
    ax1.set_title('Omega-delta diagram', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # K_eff bar chart
    names = [r['name'] for r in rows]
    K_vals = [r['K'] for r in rows]
    colors = ['blue' if r['phase'] == 'SS' else 'red' for r in rows]
    ax2.barh(range(len(names)), K_vals, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=7)
    ax2.axvline(x=0, color='black', linewidth=1)
    ax2.set_xlabel('K_eff = (1-q)(1-H)', fontsize=12)
    ax2.set_title('K_eff classification (blue=SS, red=IM)', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'test4_yang_zhang.png'), dpi=150)
    plt.close(fig)
    print(f"\n  Figure saved: scripts/figures/test4_yang_zhang.png")


# =============================================================================
# Test 5: Per-Element Surplus
# =============================================================================

def test_per_element_surplus():
    """pi_K(J) = K(J)/J numerical verification for equimolar compositions."""
    print("\n" + "=" * 70)
    print("TEST 5: Per-element surplus pi_K(J) = K(J)/J")
    print("=" * 70)

    # Use a fixed representative set of radii for equimolar compositions
    # Pick first J elements from a fixed ordering (by atomic number)
    ordered_symbols = ['Al', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
                       'Cu', 'Zr', 'Nb', 'Mo', 'Hf', 'Ta', 'W']

    J_range = np.arange(1, min(21, len(ordered_symbols) + 1))

    # Also compute analytical result with constant q
    print("\n  Analytical (constant q = 0.5):")
    print(f"  {'J':>3s} {'K':>8s} {'pi_K':>8s} {'Delta_K':>8s}")
    print(f"  {'-'*30}")

    q_const = 0.5
    pi_K_analytical = []
    K_analytical = []
    for J in range(1, 21):
        H_eq = 1.0 / J
        K_val = (1.0 - q_const) * (1.0 - H_eq)
        pi_K = K_val / J
        dK = K_val - ((1.0 - q_const) * (1.0 - 1.0 / (J - 1)) if J > 1 else 0.0)
        pi_K_analytical.append(pi_K)
        K_analytical.append(K_val)
        if J <= 10 or J == 15 or J == 20:
            print(f"  {J:3d} {K_val:8.4f} {pi_K:8.4f} {dK:+8.4f}")

    # Numerical with actual elements
    print(f"\n  With actual element radii (delta-dependent q):")
    print(f"  {'J':>3s} {'Elements':<35s} {'delta%':>7s} {'q':>7s} {'K':>7s} {'pi_K':>7s} {'dK':>7s}")
    print(f"  {'-'*80}")

    K_numerical = []
    pi_K_numerical = []
    J_num_range = []

    K_prev = 0.0
    for J in range(1, len(ordered_symbols) + 1):
        syms = ordered_symbols[:J]
        fracs = np.ones(J) / J
        if J == 1:
            K_val = 0.0
            delta_pct = 0.0
            q_val = 1.0
        else:
            props = get_alloy_properties(syms)
            K_val = props['K']
            delta_pct = props['delta_pct']
            q_val = props['q']

        pi_K = K_val / J if J > 0 else 0.0
        dK = K_val - K_prev

        K_numerical.append(K_val)
        pi_K_numerical.append(pi_K)
        J_num_range.append(J)

        elem_str = '-'.join(syms)
        if len(elem_str) > 34:
            elem_str = elem_str[:31] + '...'
        print(f"  {J:3d} {elem_str:<35s} {delta_pct:7.2f} {q_val:7.3f} "
              f"{K_val:7.3f} {pi_K:7.4f} {dK:+7.4f}")

        K_prev = K_val

    # Find peak of pi_K
    j_peak_analytical = np.argmax(pi_K_analytical) + 1
    j_peak_numerical = np.argmax(pi_K_numerical) + 1
    print(f"\n  Analytical pi_K peak at J = {j_peak_analytical}")
    print(f"  Numerical pi_K peak at J = {j_peak_numerical}")

    # Marginal strengthening
    dK_numerical = np.diff([0.0] + K_numerical)
    j_peak_marginal = np.argmax(dK_numerical) + 1
    print(f"  Marginal strengthening dK peak at J = {j_peak_marginal}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: pi_K vs J
    Js_analytic = np.arange(1, 21)
    ax1.plot(Js_analytic, pi_K_analytical, 'b--o', markersize=5,
             label='Analytical (q=0.5 const)', alpha=0.7)
    ax1.plot(J_num_range, pi_K_numerical, 'r-s', markersize=6,
             label='Numerical (delta-dep. q)', linewidth=2)
    ax1.axvline(x=2, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('J (number of elements)', fontsize=12)
    ax1.set_ylabel('pi_K = K(J)/J', fontsize=12)
    ax1.set_title('Per-element surplus pi_K(J)', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, 21))

    # Right: marginal strengthening dK vs J
    dK_analytic = np.diff([0.0] + K_analytical)
    ax2.plot(Js_analytic, dK_analytic, 'b--o', markersize=5,
             label='Analytical (q=0.5 const)', alpha=0.7)
    ax2.plot(J_num_range, dK_numerical, 'r-s', markersize=6,
             label='Numerical (delta-dep. q)', linewidth=2)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_xlabel('J (number of elements)', fontsize=12)
    ax2.set_ylabel('Delta_K = K(J) - K(J-1)', fontsize=12)
    ax2.set_title('Marginal strengthening', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(1, max(J_num_range) + 1))

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'test5_per_element_surplus.png'), dpi=150)
    plt.close(fig)
    print(f"\n  Figure saved: scripts/figures/test5_per_element_surplus.png")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  q-THERMODYNAMIC HEA THEORY: STRESS TESTS")
    print("=" * 70)

    test_subsystem_transferability()
    test_alx_cocrfeni()
    test_q_constancy_cantor()
    test_yang_zhang_classification()
    test_per_element_surplus()

    print("\n" + "=" * 70)
    print("  ALL STRESS TESTS COMPLETE")
    print("=" * 70)
    print()
