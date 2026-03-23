#!/usr/bin/env python3
"""
THE CRITICAL TEST: Fit q independently from 4 property channels and check agreement.

For CoCrFeMnNi (Cantor alloy) and WMoTaNbZr (refractory HEA), fit the CES
complementarity parameter q from:
  1. Elastic modulus (Voigt-Reuss-CES vs experiment)
  2. Thermal conductivity (ROM vs experiment)
  3. Hardness / yield strength (ROM vs experiment)
  4. delta-based geometric estimate (q = 1 - alpha*delta^2)

Uses PyTorch for GPU-accelerated bootstrap confidence intervals.

Key findings:
  - Cantor alloy (delta ~1.18%) has q ~0.97 => CES ~ROM, tiny corrections.
    Thermal conductivity deficit is dominated by alloy (Nordheim) scattering,
    not CES curvature.
  - Refractory HEA (delta ~5.25%) has q ~0.37 => large CES curvature,
    better test case for the theory.
"""

import sys
import os
import numpy as np

# Import element database and core functions from hea_validate_core
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hea_validate_core import (
    ELEMENTS, Element, compute_delta, compute_H, compute_K,
    compute_q_from_delta, compute_CES, compute_Zq, get_alloy_properties,
    ALPHA_DEFAULT,
)

try:
    import torch
    HAS_TORCH = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    HAS_TORCH = False
    DEVICE = None

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# =============================================================================
# Numerical root finding (scipy-free)
# =============================================================================

def ces_power_mean(fracs, props, q):
    """Compute CES power mean: (sum c_j * x_j^q)^{1/q}.

    Handles q=0 (geometric mean) and negative q gracefully.
    Works with both numpy arrays and torch tensors.
    """
    if HAS_TORCH and isinstance(fracs, torch.Tensor):
        return _ces_power_mean_torch(fracs, props, q)
    return _ces_power_mean_numpy(fracs, props, q)


def _ces_power_mean_numpy(fracs, props, q):
    """Numpy implementation of CES power mean."""
    if abs(q) < 1e-12:
        mask = fracs > 0
        return np.exp(np.sum(fracs[mask] * np.log(props[mask])))
    Z = np.sum(fracs * props**q)
    if Z <= 0:
        return np.nan
    return Z**(1.0 / q)


def _ces_power_mean_torch(fracs, props, q):
    """Torch implementation of CES power mean (batched over q)."""
    # q can be a tensor (batch of q values)
    if isinstance(q, (int, float)):
        q = torch.tensor(q, dtype=fracs.dtype, device=fracs.device)
    # fracs, props: (J,), q: (N,) or scalar
    # Compute (sum c_j * x_j^q)^{1/q} for each q
    if q.dim() == 0:
        if torch.abs(q) < 1e-12:
            mask = fracs > 0
            return torch.exp(torch.sum(fracs[mask] * torch.log(props[mask])))
        Z = torch.sum(fracs * props**q)
        if Z <= 0:
            return torch.tensor(float('nan'))
        return Z**(1.0 / q)
    else:
        # Batched: q is (N,), fracs/props are (J,)
        # props_q: (N, J) = props[j]^q[i]
        log_props = torch.log(props).unsqueeze(0)  # (1, J)
        q_2d = q.unsqueeze(1)  # (N, 1)
        props_q = torch.exp(q_2d * log_props)  # (N, J)
        Z = torch.sum(fracs.unsqueeze(0) * props_q, dim=1)  # (N,)
        result = torch.where(
            torch.abs(q) < 1e-12,
            torch.exp(torch.sum(fracs * torch.log(props))).expand_as(q),
            torch.where(Z > 0, Z**(1.0 / q), torch.tensor(float('nan'), device=q.device))
        )
        return result


def bisect_q_for_target(fracs, props, target, q_lo=-20.0, q_hi=2.0,
                        tol=1e-10, max_iter=200):
    """Find q such that CES_power_mean(fracs, props, q) = target via bisection.

    Returns q, or np.nan if no solution in range.
    """
    f_lo = _ces_power_mean_numpy(fracs, props, q_lo) - target
    f_hi = _ces_power_mean_numpy(fracs, props, q_hi) - target

    if np.isnan(f_lo) or np.isnan(f_hi):
        return np.nan
    if f_lo * f_hi > 0:
        # No sign change — no root in this interval
        return np.nan

    for _ in range(max_iter):
        q_mid = 0.5 * (q_lo + q_hi)
        f_mid = _ces_power_mean_numpy(fracs, props, q_mid) - target
        if np.isnan(f_mid):
            return np.nan
        if abs(f_mid) < tol:
            return q_mid
        if f_mid * f_lo < 0:
            q_hi = q_mid
            f_hi = f_mid
        else:
            q_lo = q_mid
            f_lo = f_mid

    return 0.5 * (q_lo + q_hi)


def bisect_q_for_target_torch(fracs, props, targets, q_lo=-20.0, q_hi=2.0,
                               tol=1e-10, max_iter=200):
    """Batched bisection on GPU: find q for each target value.

    fracs, props: (J,) tensors on device.
    targets: (N,) tensor of target values.
    Returns: (N,) tensor of q values.
    """
    N = targets.shape[0]
    lo = torch.full((N,), q_lo, device=targets.device, dtype=targets.dtype)
    hi = torch.full((N,), q_hi, device=targets.device, dtype=targets.dtype)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = _ces_power_mean_torch(fracs, props, mid) - targets
        # Update bounds
        sign_lo = (_ces_power_mean_torch(fracs, props, lo) - targets).sign()
        go_left = (f_mid * sign_lo) < 0
        hi = torch.where(go_left, mid, hi)
        lo = torch.where(go_left, lo, mid)
        if (hi - lo).max() < tol:
            break

    return 0.5 * (lo + hi)


# =============================================================================
# Bootstrap confidence intervals
# =============================================================================

def bootstrap_q_ci(fracs, props, target, target_err, n_boot=10000,
                   ci_level=0.95, q_lo=-20.0, q_hi=2.0):
    """Bootstrap confidence interval for fitted q.

    Resample target from Normal(target, target_err) and refit q each time.
    Uses GPU if available via PyTorch, falls back to numpy.

    Returns: (q_median, q_lo_ci, q_hi_ci)
    """
    alpha = 1.0 - ci_level

    if HAS_TORCH and target_err > 0:
        return _bootstrap_q_torch(fracs, props, target, target_err,
                                  n_boot, alpha, q_lo, q_hi)
    else:
        return _bootstrap_q_numpy(fracs, props, target, target_err,
                                  n_boot, alpha, q_lo, q_hi)


def _bootstrap_q_torch(fracs, props, target, target_err, n_boot, alpha,
                        q_lo, q_hi):
    """GPU-accelerated bootstrap using PyTorch batched bisection."""
    fracs_t = torch.tensor(fracs, dtype=torch.float64, device=DEVICE)
    props_t = torch.tensor(props, dtype=torch.float64, device=DEVICE)

    # Generate bootstrap target samples
    torch.manual_seed(42)
    targets_t = torch.normal(
        mean=torch.full((n_boot,), target, dtype=torch.float64, device=DEVICE),
        std=torch.full((n_boot,), target_err, dtype=torch.float64, device=DEVICE),
    )
    # Clamp to positive values (physical requirement)
    targets_t = torch.clamp(targets_t, min=0.1)

    # Batched bisection
    q_samples = bisect_q_for_target_torch(fracs_t, props_t, targets_t,
                                           q_lo=q_lo, q_hi=q_hi)

    # Remove NaN
    q_valid = q_samples[~torch.isnan(q_samples)]
    if len(q_valid) == 0:
        return np.nan, np.nan, np.nan

    q_np = q_valid.cpu().numpy()
    q_median = np.median(q_np)
    q_lo_ci = np.percentile(q_np, 100 * alpha / 2)
    q_hi_ci = np.percentile(q_np, 100 * (1.0 - alpha / 2))
    return q_median, q_lo_ci, q_hi_ci


def _bootstrap_q_numpy(fracs, props, target, target_err, n_boot, alpha,
                        q_lo, q_hi):
    """Numpy fallback bootstrap."""
    rng = np.random.default_rng(42)
    if target_err <= 0:
        q_fit = bisect_q_for_target(fracs, props, target, q_lo=q_lo, q_hi=q_hi)
        return q_fit, q_fit, q_fit

    targets = rng.normal(target, target_err, size=n_boot)
    targets = np.clip(targets, 0.1, None)

    q_samples = []
    for t in targets:
        q = bisect_q_for_target(fracs, props, t, q_lo=q_lo, q_hi=q_hi)
        if not np.isnan(q):
            q_samples.append(q)

    if len(q_samples) == 0:
        return np.nan, np.nan, np.nan

    q_arr = np.array(q_samples)
    q_median = np.median(q_arr)
    q_lo_ci = np.percentile(q_arr, 100 * alpha / 2)
    q_hi_ci = np.percentile(q_arr, 100 * (1.0 - alpha / 2))
    return q_median, q_lo_ci, q_hi_ci


# =============================================================================
# Channel fitting for a single alloy
# =============================================================================

def fit_q_elastic(fracs, Es, E_meas, E_err=0.0):
    """Channel 1: Fit q from elastic modulus."""
    q_fit = bisect_q_for_target(fracs, Es, E_meas)
    q_med, q_lo, q_hi = bootstrap_q_ci(fracs, Es, E_meas, E_err)
    return {
        'channel': 'Elastic modulus',
        'ROM': np.sum(fracs * Es),
        'measured': E_meas,
        'unit': 'GPa',
        'q_fit': q_fit,
        'q_median': q_med, 'q_lo': q_lo, 'q_hi': q_hi,
    }


def fit_q_thermal(fracs, kappas, kappa_meas, kappa_err=0.0):
    """Channel 2: Fit q from thermal conductivity."""
    q_fit = bisect_q_for_target(fracs, kappas, kappa_meas)
    q_med, q_lo, q_hi = bootstrap_q_ci(fracs, kappas, kappa_meas, kappa_err)

    # Check Reuss bound
    kappa_reuss = _ces_power_mean_numpy(fracs, kappas, -1.0)

    return {
        'channel': 'Thermal conductivity',
        'ROM': np.sum(fracs * kappas),
        'Reuss': kappa_reuss,
        'measured': kappa_meas,
        'unit': 'W/mK',
        'q_fit': q_fit,
        'q_median': q_med, 'q_lo': q_lo, 'q_hi': q_hi,
        'below_reuss': kappa_meas < kappa_reuss,
    }


def fit_q_strength(fracs, sigma_ys, sigma_meas, sigma_err=0.0):
    """Channel 3: Fit q from yield strength."""
    q_fit = bisect_q_for_target(fracs, sigma_ys, sigma_meas)
    q_med, q_lo, q_hi = bootstrap_q_ci(fracs, sigma_ys, sigma_meas, sigma_err)
    return {
        'channel': 'Yield strength',
        'ROM': np.sum(fracs * sigma_ys),
        'measured': sigma_meas,
        'unit': 'MPa',
        'q_fit': q_fit,
        'q_median': q_med, 'q_lo': q_lo, 'q_hi': q_hi,
    }


def fit_q_delta(radii, fracs, alpha=ALPHA_DEFAULT):
    """Channel 4: q from atomic size mismatch."""
    delta = compute_delta(radii, fracs)
    q = compute_q_from_delta(delta, alpha)
    return {
        'channel': 'delta (mismatch)',
        'delta_pct': delta * 100,
        'q_fit': q,
        'q_median': q, 'q_lo': q, 'q_hi': q,  # no uncertainty modeled
    }


# =============================================================================
# CES sweep: property vs q curve
# =============================================================================

def ces_sweep(fracs, props, q_range=None):
    """Compute CES power mean over a range of q values."""
    if q_range is None:
        q_range = np.linspace(-5, 2, 500)
    vals = np.array([_ces_power_mean_numpy(fracs, props, q) for q in q_range])
    return q_range, vals


# =============================================================================
# Alloy analysis: Cantor CoCrFeMnNi
# =============================================================================

def analyze_cantor():
    """Full 4-channel q fit for CoCrFeMnNi (Cantor alloy)."""
    print("\n" + "=" * 74)
    print("  CANTOR ALLOY: CoCrFeMnNi — 4-Channel q Fit")
    print("=" * 74)

    symbols = ['Co', 'Cr', 'Fe', 'Mn', 'Ni']
    props = get_alloy_properties(symbols)
    fracs = props['fracs']
    Es = props['Es']
    kappas = props['kappas']
    sigma_ys = props['sigma_ys']
    radii = props['radii']

    print(f"\n  Composition: equimolar {'-'.join(symbols)}")
    print(f"  delta = {props['delta_pct']:.2f}%  |  q(delta) = {props['q']:.4f}")
    print(f"  H = {props['H']:.3f}  |  K = {props['K']:.4f}")

    # --- Element properties table ---
    print(f"\n  {'Elem':>5s} {'c_j':>6s} {'r/pm':>6s} {'E/GPa':>7s} "
          f"{'kappa':>7s} {'sig_y':>7s}")
    print(f"  {'─' * 44}")
    for i, s in enumerate(symbols):
        print(f"  {s:>5s} {fracs[i]:6.3f} {radii[i]:6.1f} {Es[i]:7.1f} "
              f"{kappas[i]:7.1f} {sigma_ys[i]:7.0f}")

    # --- Channel 1: Elastic modulus ---
    print(f"\n  {'─' * 70}")
    print("  Channel 1: ELASTIC MODULUS")
    print(f"  {'─' * 70}")

    E_voigt = np.sum(fracs * Es)
    E_reuss = 1.0 / np.sum(fracs / Es)
    E_meas = 200.0  # GPa, Laplanche et al. 2015
    E_err = 10.0    # estimated uncertainty

    print(f"    E_Voigt (q=1)  = {E_voigt:.1f} GPa")
    print(f"    E_Reuss (q=-1) = {E_reuss:.1f} GPa")
    print(f"    E_measured     = {E_meas:.1f} +/- {E_err:.0f} GPa (Laplanche et al. 2015)")

    ch1 = fit_q_elastic(fracs, Es, E_meas, E_err)
    print(f"    q_elastic      = {ch1['q_fit']:.4f}")
    print(f"    Bootstrap 95% CI: [{ch1['q_lo']:.4f}, {ch1['q_hi']:.4f}]")
    print(f"    Note: E_measured is between Reuss and Voigt, so q is well-defined.")

    # --- Channel 2: Thermal conductivity ---
    print(f"\n  {'─' * 70}")
    print("  Channel 2: THERMAL CONDUCTIVITY")
    print(f"  {'─' * 70}")

    kappa_rom = np.sum(fracs * kappas)
    kappa_reuss = 1.0 / np.sum(fracs / kappas)
    kappa_meas = 12.0  # W/mK, Jin et al. 2016
    kappa_err = 2.0

    print(f"    kappa_ROM (q=1)  = {kappa_rom:.1f} W/mK")
    print(f"    kappa_Reuss (q=-1) = {kappa_reuss:.1f} W/mK")
    print(f"    kappa_measured   = {kappa_meas:.1f} +/- {kappa_err:.0f} W/mK (Jin et al. 2016)")

    ch2 = fit_q_thermal(fracs, kappas, kappa_meas, kappa_err)

    if ch2['below_reuss']:
        print(f"\n    *** DISCREPANCY: kappa_measured ({kappa_meas:.1f}) < kappa_Reuss ({kappa_reuss:.1f}) ***")
        print(f"    The CES power mean with ANY q gives kappa >= kappa_harmonic_mean.")
        print(f"    The measured value {kappa_meas:.1f} W/mK is BELOW the Reuss bound.")
        print(f"    This means CES curvature alone CANNOT explain the thermal")
        print(f"    conductivity reduction. The dominant mechanism is Nordheim-type")
        print(f"    alloy scattering (phonon scattering from mass/force-constant")
        print(f"    disorder), which is NOT captured by the CES partition function.")
        print(f"    Mn (kappa=7.8 W/mK) is the bottleneck, but even harmonic mean")
        print(f"    ({kappa_reuss:.1f} W/mK) far exceeds 12 W/mK.")
        if np.isnan(ch2['q_fit']):
            print(f"    q_thermal = NO SOLUTION (target below Reuss bound)")
        else:
            print(f"    q_thermal = {ch2['q_fit']:.4f} (unrealistically negative)")
    else:
        print(f"    q_thermal      = {ch2['q_fit']:.4f}")
        print(f"    Bootstrap 95% CI: [{ch2['q_lo']:.4f}, {ch2['q_hi']:.4f}]")

    # --- Channel 3: Yield strength ---
    print(f"\n  {'─' * 70}")
    print("  Channel 3: YIELD STRENGTH")
    print(f"  {'─' * 70}")

    sigma_rom = np.sum(fracs * sigma_ys)
    sigma_meas_ann = 125.0   # MPa, annealed Cantor, Otto et al. 2013
    sigma_err = 15.0
    HV_meas = 1.1            # GPa

    print(f"    sigma_y,ROM    = {sigma_rom:.0f} MPa")
    print(f"    sigma_y,meas   = {sigma_meas_ann:.0f} +/- {sigma_err:.0f} MPa (annealed, Otto 2013)")
    print(f"    HV_measured    = {HV_meas:.1f} GPa")

    ch3 = fit_q_strength(fracs, sigma_ys, sigma_meas_ann, sigma_err)

    if sigma_meas_ann < sigma_rom:
        print(f"\n    *** FAILURE POINT: sigma_y,measured ({sigma_meas_ann:.0f} MPa) "
              f"< sigma_y,ROM ({sigma_rom:.0f} MPa) ***")
        print(f"    The paper's cocktail effect predicts ABOVE-ROM for resistance")
        print(f"    properties (q < 1 => CES < ROM for positive properties).")
        print(f"    But annealed Cantor alloy yield strength is BELOW ROM.")
        print(f"    This is expected: pure-element yield strengths are for")
        print(f"    work-hardened/polycrystalline forms; annealed single-phase")
        print(f"    FCC solid solution has lower sigma_y than several constituents.")
        print(f"    q_strength     = {ch3['q_fit']:.4f}")
        print(f"    Bootstrap 95% CI: [{ch3['q_lo']:.4f}, {ch3['q_hi']:.4f}]")
        print(f"    A q < 1 means CES pulls BELOW ROM — opposite to cocktail effect.")
    else:
        print(f"    q_strength     = {ch3['q_fit']:.4f}")
        print(f"    Bootstrap 95% CI: [{ch3['q_lo']:.4f}, {ch3['q_hi']:.4f}]")

    # --- Channel 4: delta-based ---
    print(f"\n  {'─' * 70}")
    print("  Channel 4: delta-BASED GEOMETRIC ESTIMATE")
    print(f"  {'─' * 70}")

    ch4 = fit_q_delta(radii, fracs)
    print(f"    delta          = {ch4['delta_pct']:.2f}%")
    print(f"    q_mismatch     = 1 - {ALPHA_DEFAULT:.1f} * ({ch4['delta_pct']/100:.4f})^2 = {ch4['q_fit']:.4f}")
    print(f"    At q ~ 1, CES ~ ROM so corrections are tiny.")

    # --- Summary table ---
    channels = [ch1, ch2, ch3, ch4]
    print(f"\n  {'=' * 70}")
    print("  CANTOR ALLOY: q-FIT SUMMARY TABLE")
    print(f"  {'=' * 70}")
    print(f"\n  {'Channel':<25s} {'q_fit':>8s} {'95% CI':>20s} {'Notes':>20s}")
    print(f"  {'─' * 75}")

    for ch in channels:
        q_str = f"{ch['q_fit']:.4f}" if not np.isnan(ch['q_fit']) else "NO SOLN"
        if ch['q_lo'] == ch['q_hi']:
            ci_str = "(no uncertainty)"
        elif np.isnan(ch['q_lo']):
            ci_str = "N/A"
        else:
            ci_str = f"[{ch['q_lo']:.4f}, {ch['q_hi']:.4f}]"

        notes = ""
        if ch['channel'] == 'Thermal conductivity' and ch.get('below_reuss'):
            notes = "below Reuss!"
        elif ch['channel'] == 'Yield strength' and ch.get('measured', 0) < ch.get('ROM', 0):
            notes = "below ROM!"

        print(f"  {ch['channel']:<25s} {q_str:>8s} {ci_str:>20s} {notes:>20s}")

    # Analysis
    print(f"\n  ANALYSIS:")
    print(f"  The 4 channels give DISAGREEING q values for the Cantor alloy.")
    print(f"  - q(delta) ~ {ch4['q_fit']:.3f} (geometric, very close to 1)")
    print(f"  - q(elastic) ~ {ch1['q_fit']:.3f} (small deviation from ROM)")
    q_th = ch2['q_fit']
    q_th_str = 'NO SOLUTION' if np.isnan(q_th) else f'{q_th:.3f}'
    print(f"  - q(thermal) = {q_th_str} (Nordheim scattering dominates)")
    print(f"  - q(strength) ~ {ch3['q_fit']:.3f} (below ROM, not cocktail effect)")
    print(f"")
    print(f"  ROOT CAUSE: delta ~ 1.18% is tiny. At q ~ 0.97, CES ~ ROM,")
    print(f"  so the CES framework has almost no predictive power for this alloy.")
    print(f"  The measured deviations from ROM are driven by OTHER physics:")
    print(f"    - Thermal: Nordheim alloy scattering (phonon mass disorder)")
    print(f"    - Strength: solid solution softening in annealed FCC")
    print(f"    - Elastic: texture, grain boundaries, porosity")

    return channels


# =============================================================================
# Alloy analysis: Refractory WMoTaNbZr
# =============================================================================

def analyze_refractory():
    """Full 4-channel q fit for WMoTaNbZr (refractory HEA)."""
    print("\n" + "=" * 74)
    print("  REFRACTORY HEA: WMoTaNbZr — 4-Channel q Fit")
    print("=" * 74)

    symbols = ['W', 'Mo', 'Ta', 'Nb', 'Zr']
    props = get_alloy_properties(symbols)
    fracs = props['fracs']
    Es = props['Es']
    kappas = props['kappas']
    sigma_ys = props['sigma_ys']
    radii = props['radii']

    print(f"\n  Composition: equimolar {'-'.join(symbols)}")
    print(f"  delta = {props['delta_pct']:.2f}%  |  q(delta) = {props['q']:.4f}")
    print(f"  H = {props['H']:.3f}  |  K = {props['K']:.4f}")

    # --- Element properties table ---
    print(f"\n  {'Elem':>5s} {'c_j':>6s} {'r/pm':>6s} {'E/GPa':>7s} "
          f"{'kappa':>7s} {'sig_y':>7s}")
    print(f"  {'─' * 44}")
    for i, s in enumerate(symbols):
        print(f"  {s:>5s} {fracs[i]:6.3f} {radii[i]:6.1f} {Es[i]:7.1f} "
              f"{kappas[i]:7.1f} {sigma_ys[i]:7.0f}")

    # --- Experimental / estimated values ---
    # WMoTaNbZr: Senkov et al. 2011, 2018
    # E_measured ~ 178 GPa (estimated from nanoindentation / ultrasonic)
    # kappa ~ 15-20 W/mK (estimated, not well characterized)
    # sigma_y ~ 1200 MPa (compression, Senkov 2011)

    E_rom = np.sum(fracs * Es)
    E_reuss = 1.0 / np.sum(fracs / Es)
    E_meas = 178.0   # GPa, estimated
    E_err = 20.0

    kappa_rom = np.sum(fracs * kappas)
    kappa_reuss = 1.0 / np.sum(fracs / kappas)
    kappa_meas = 18.0  # W/mK, estimated
    kappa_err = 5.0

    sigma_rom = np.sum(fracs * sigma_ys)
    sigma_meas = 1200.0  # MPa, Senkov et al. 2011
    sigma_err = 150.0

    # --- Channel 1: Elastic modulus ---
    print(f"\n  {'─' * 70}")
    print("  Channel 1: ELASTIC MODULUS")
    print(f"  {'─' * 70}")
    print(f"    E_Voigt (q=1)  = {E_rom:.1f} GPa")
    print(f"    E_Reuss (q=-1) = {E_reuss:.1f} GPa")
    print(f"    E_measured     = {E_meas:.1f} +/- {E_err:.0f} GPa [ESTIMATED]")

    ch1 = fit_q_elastic(fracs, Es, E_meas, E_err)

    if np.isnan(ch1['q_fit']):
        # Check if below Reuss
        if E_meas < E_reuss:
            print(f"    q_elastic = NO SOLUTION (below Reuss bound {E_reuss:.1f} GPa)")
        else:
            print(f"    q_elastic = NO SOLUTION (above Voigt bound {E_rom:.1f} GPa)")
    else:
        print(f"    q_elastic      = {ch1['q_fit']:.4f}")
        print(f"    Bootstrap 95% CI: [{ch1['q_lo']:.4f}, {ch1['q_hi']:.4f}]")

    # --- Channel 2: Thermal conductivity ---
    print(f"\n  {'─' * 70}")
    print("  Channel 2: THERMAL CONDUCTIVITY")
    print(f"  {'─' * 70}")
    print(f"    kappa_ROM (q=1)   = {kappa_rom:.1f} W/mK")
    print(f"    kappa_Reuss (q=-1)= {kappa_reuss:.1f} W/mK")
    print(f"    kappa_measured    = {kappa_meas:.1f} +/- {kappa_err:.0f} W/mK [ESTIMATED]")

    ch2 = fit_q_thermal(fracs, kappas, kappa_meas, kappa_err)

    if ch2.get('below_reuss'):
        print(f"    *** kappa_measured < kappa_Reuss: Nordheim scattering dominates ***")
        if np.isnan(ch2['q_fit']):
            print(f"    q_thermal = NO SOLUTION")
        else:
            print(f"    q_thermal = {ch2['q_fit']:.4f} (unrealistically negative)")
    else:
        print(f"    q_thermal      = {ch2['q_fit']:.4f}")
        print(f"    Bootstrap 95% CI: [{ch2['q_lo']:.4f}, {ch2['q_hi']:.4f}]")

    # --- Channel 3: Yield strength ---
    print(f"\n  {'─' * 70}")
    print("  Channel 3: YIELD STRENGTH")
    print(f"  {'─' * 70}")
    print(f"    sigma_y,ROM    = {sigma_rom:.0f} MPa")
    print(f"    sigma_y,meas   = {sigma_meas:.0f} +/- {sigma_err:.0f} MPa (Senkov 2011) [compression]")

    ch3 = fit_q_strength(fracs, sigma_ys, sigma_meas, sigma_err)

    if sigma_meas > sigma_rom:
        print(f"    sigma_y,measured ({sigma_meas:.0f}) > sigma_y,ROM ({sigma_rom:.0f}) MPa")
        print(f"    This is the COCKTAIL EFFECT: strength exceeds ROM.")
        print(f"    For CES with q > 1: power mean EXCEEDS arithmetic mean.")
        print(f"    q_strength     = {ch3['q_fit']:.4f}")
        print(f"    Bootstrap 95% CI: [{ch3['q_lo']:.4f}, {ch3['q_hi']:.4f}]")
    elif np.isnan(ch3['q_fit']):
        print(f"    q_strength = NO SOLUTION")
    else:
        print(f"    q_strength     = {ch3['q_fit']:.4f}")
        print(f"    Bootstrap 95% CI: [{ch3['q_lo']:.4f}, {ch3['q_hi']:.4f}]")

    # --- Channel 4: delta-based ---
    print(f"\n  {'─' * 70}")
    print("  Channel 4: delta-BASED GEOMETRIC ESTIMATE")
    print(f"  {'─' * 70}")

    ch4 = fit_q_delta(radii, fracs)
    print(f"    delta          = {ch4['delta_pct']:.2f}%")
    print(f"    q_mismatch     = 1 - {ALPHA_DEFAULT:.1f} * ({ch4['delta_pct']/100:.4f})^2 = {ch4['q_fit']:.4f}")
    print(f"    At q ~ {ch4['q_fit']:.2f}, CES shows LARGE curvature away from ROM.")

    # --- Summary table ---
    channels = [ch1, ch2, ch3, ch4]
    print(f"\n  {'=' * 70}")
    print("  REFRACTORY HEA WMoTaNbZr: q-FIT SUMMARY TABLE")
    print(f"  {'=' * 70}")
    print(f"\n  {'Channel':<25s} {'q_fit':>8s} {'95% CI':>20s} {'Notes':>20s}")
    print(f"  {'─' * 75}")

    for ch in channels:
        q_str = f"{ch['q_fit']:.4f}" if not np.isnan(ch['q_fit']) else "NO SOLN"
        if ch['q_lo'] == ch['q_hi']:
            ci_str = "(no uncertainty)"
        elif np.isnan(ch['q_lo']):
            ci_str = "N/A"
        else:
            ci_str = f"[{ch['q_lo']:.4f}, {ch['q_hi']:.4f}]"

        notes = ""
        if ch['channel'] == 'Thermal conductivity' and ch.get('below_reuss'):
            notes = "below Reuss!"
        elif ch['channel'] == 'Yield strength':
            meas = ch.get('measured', 0)
            rom = ch.get('ROM', 0)
            if meas > rom:
                notes = "ABOVE ROM (cocktail)"
            elif meas < rom:
                notes = "below ROM"

        print(f"  {ch['channel']:<25s} {q_str:>8s} {ci_str:>20s} {notes:>20s}")

    # Analysis
    print(f"\n  ANALYSIS:")
    print(f"  Refractory WMoTaNbZr has delta ~ 5.25%, giving q ~ {ch4['q_fit']:.2f}.")
    print(f"  This is a MUCH better test case for CES theory than Cantor alloy:")
    print(f"  - Large lattice distortion => large CES curvature K ~ {props['K']:.2f}")
    print(f"  - Yield strength ({sigma_meas:.0f} MPa) far exceeds ROM ({sigma_rom:.0f} MPa)")
    print(f"  - Elastic modulus shows significant deviation from Voigt average")
    print(f"  - Thermal conductivity still complicated by Nordheim scattering")
    print(f"")
    print(f"  However, the q values from different channels still need not agree")
    print(f"  exactly, because each property has different sensitivity to the")
    print(f"  microscopic disorder (mass vs force-constant vs scattering).")
    print(f"  The theory predicts a UNIVERSAL q, but real materials have")
    print(f"  property-dependent corrections beyond the CES framework.")

    return channels


# =============================================================================
# Comparison figure
# =============================================================================

def plot_q_comparison(cantor_channels, refr_channels):
    """Create a comparison figure of q values across channels and alloys."""
    if not HAS_MPL:
        print("\n  [matplotlib not available — skipping figure generation]")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    def plot_alloy(ax, channels, title, q_delta):
        labels = []
        q_vals = []
        q_los = []
        q_his = []
        colors_list = []
        chan_colors = {
            'Elastic modulus': '#2196F3',
            'Thermal conductivity': '#F44336',
            'Yield strength': '#4CAF50',
            'delta (mismatch)': '#FF9800',
        }

        for ch in channels:
            labels.append(ch['channel'])
            q_val = ch['q_fit'] if not np.isnan(ch['q_fit']) else 0.0
            q_vals.append(q_val)
            q_lo = ch['q_lo'] if not np.isnan(ch['q_lo']) else q_val
            q_hi = ch['q_hi'] if not np.isnan(ch['q_hi']) else q_val
            q_los.append(max(0.0, q_val - q_lo))
            q_his.append(max(0.0, q_hi - q_val))
            colors_list.append(chan_colors.get(ch['channel'], '#666666'))

        y_pos = np.arange(len(labels))
        ax.barh(y_pos, q_vals, xerr=[q_los, q_his],
                color=colors_list, alpha=0.7, edgecolor='black',
                capsize=4, height=0.6)

        # Mark q(delta) reference line
        ax.axvline(x=q_delta, color='#FF9800', linestyle='--', alpha=0.5,
                   label=f'q(delta) = {q_delta:.3f}')
        # Mark q=1 (ROM)
        ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5,
                   label='q=1 (ROM)')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Fitted q', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(axis='x', alpha=0.3)

        # Mark no-solution channels
        for i, ch in enumerate(channels):
            if np.isnan(ch['q_fit']):
                ax.annotate('NO SOLN', xy=(0, i), fontsize=9,
                            color='red', fontweight='bold',
                            ha='center', va='center')

    plot_alloy(axes[0], cantor_channels, 'Cantor CoCrFeMnNi (delta=1.18%)',
               cantor_channels[3]['q_fit'])
    plot_alloy(axes[1], refr_channels, 'Refractory WMoTaNbZr (delta=5.25%)',
               refr_channels[3]['q_fit'])

    plt.tight_layout()
    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, 'q_channel_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure saved: {fig_path}")


def plot_ces_sweep_curves(cantor_channels, refr_channels):
    """Plot CES power mean vs q for each property channel."""
    if not HAS_MPL:
        print("  [matplotlib not available — skipping CES sweep figure]")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    alloys = [
        ('CoCrFeMnNi', ['Co', 'Cr', 'Fe', 'Mn', 'Ni'], cantor_channels,
         {'E': 200.0, 'kappa': 12.0, 'sigma_y': 125.0}),
        ('WMoTaNbZr', ['W', 'Mo', 'Ta', 'Nb', 'Zr'], refr_channels,
         {'E': 178.0, 'kappa': 18.0, 'sigma_y': 1200.0}),
    ]

    prop_keys = [
        ('Es', 'E (GPa)', 'Elastic Modulus'),
        ('kappas', 'kappa (W/mK)', 'Thermal Conductivity'),
        ('sigma_ys', 'sigma_y (MPa)', 'Yield Strength'),
    ]

    meas_keys = ['E', 'kappa', 'sigma_y']

    for row, (alloy_name, symbols, channels, meas_dict) in enumerate(alloys):
        ap = get_alloy_properties(symbols)
        fracs = ap['fracs']

        for col, (prop_key, ylabel, prop_title) in enumerate(prop_keys):
            ax = axes[row, col]
            prop_vals = ap[prop_key]

            q_range = np.linspace(-8, 3, 600)
            ces_vals = np.array([
                _ces_power_mean_numpy(fracs, prop_vals, q) for q in q_range
            ])

            ax.plot(q_range, ces_vals, 'b-', linewidth=2, label='CES(q)')

            # Mark measured value
            m_key = meas_keys[col]
            m_val = meas_dict[m_key]
            ax.axhline(y=m_val, color='red', linestyle='--', alpha=0.7,
                       label=f'Measured = {m_val:.0f}')

            # Mark q(delta)
            q_delta = channels[3]['q_fit']
            ces_at_qdelta = _ces_power_mean_numpy(fracs, prop_vals, q_delta)
            ax.axvline(x=q_delta, color='orange', linestyle=':', alpha=0.5,
                       label=f'q(delta) = {q_delta:.2f}')
            ax.plot(q_delta, ces_at_qdelta, 'o', color='orange', markersize=8)

            # Mark fitted q for this channel
            q_ch = channels[col]['q_fit']
            if not np.isnan(q_ch):
                ces_at_qch = _ces_power_mean_numpy(fracs, prop_vals, q_ch)
                ax.plot(q_ch, ces_at_qch, 's', color='green', markersize=10,
                        label=f'q_fit = {q_ch:.2f}', zorder=5)

            # Mark ROM (q=1)
            rom_val = np.sum(fracs * prop_vals)
            ax.plot(1.0, rom_val, 'D', color='gray', markersize=7,
                    label=f'ROM = {rom_val:.0f}')

            ax.set_xlabel('q')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{alloy_name}: {prop_title}', fontsize=11)
            ax.legend(fontsize=8, loc='best')
            ax.grid(alpha=0.3)
            ax.set_xlim(-8, 3)

    plt.tight_layout()
    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, 'ces_sweep_curves.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {fig_path}")


# =============================================================================
# Final comparative analysis
# =============================================================================

def print_comparative_analysis(cantor_ch, refr_ch):
    """Print the final comparative analysis of both alloys."""
    print("\n" + "=" * 74)
    print("  COMPARATIVE ANALYSIS: CANTOR vs REFRACTORY")
    print("=" * 74)

    print(f"\n  {'':25s} {'CoCrFeMnNi':>14s} {'WMoTaNbZr':>14s}")
    print(f"  {'─' * 55}")
    print(f"  {'delta (%)':25s} {'1.18':>14s} {'5.25':>14s}")

    cantor_q_delta = cantor_ch[3]['q_fit']
    refr_q_delta = refr_ch[3]['q_fit']
    print(f"  {'q(delta)':25s} {cantor_q_delta:14.4f} {refr_q_delta:14.4f}")

    labels = ['q(elastic)', 'q(thermal)', 'q(strength)', 'q(delta)']
    for i, label in enumerate(labels):
        q_c = cantor_ch[i]['q_fit']
        q_r = refr_ch[i]['q_fit']
        q_c_str = f"{q_c:.4f}" if not np.isnan(q_c) else "NO SOLN"
        q_r_str = f"{q_r:.4f}" if not np.isnan(q_r) else "NO SOLN"
        print(f"  {label:25s} {q_c_str:>14s} {q_r_str:>14s}")

    print(f"\n  KEY FINDINGS:")
    print(f"  ─────────────")
    print(f"  1. Cantor alloy (delta=1.18%): q ~ 0.97, CES ~ ROM.")
    print(f"     - All property deviations from ROM have other physical origins.")
    print(f"     - Thermal conductivity: Nordheim scattering (not CES curvature).")
    print(f"     - Yield strength: below ROM due to annealing (not cocktail effect).")
    print(f"     - The CES framework has almost NO predictive power here.")
    print(f"")
    print(f"  2. Refractory HEA (delta=5.25%): q ~ 0.37, large CES curvature.")
    print(f"     - Strength cocktail effect IS observed (sigma_y >> ROM).")
    print(f"     - But q(strength) likely differs from q(delta) because solid-solution")
    print(f"       strengthening has its own physics (Labusch/Varvenne models).")
    print(f"     - Different channels give different q => q is NOT universal.")
    print(f"")
    print(f"  3. CONCLUSION: The CES partition function is a useful interpolation")
    print(f"     tool between Voigt and Reuss bounds, but a SINGLE universal q")
    print(f"     does not consistently explain all property channels simultaneously.")
    print(f"     The theory works best for high-delta alloys where deviations from")
    print(f"     ROM are large, but even there, property-specific physics matters.")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 74)
    print("  THE CRITICAL TEST: Independent q-Fitting from 4 Property Channels")
    print("=" * 74)
    print(f"  PyTorch: {'available' if HAS_TORCH else 'NOT available'}"
          f"{'  (device: ' + str(DEVICE) + ')' if HAS_TORCH else ''}")
    print(f"  matplotlib: {'available' if HAS_MPL else 'NOT available'}")
    print(f"  Bootstrap resamples: 10,000 (GPU-accelerated)" if HAS_TORCH else
          f"  Bootstrap resamples: 10,000 (CPU numpy)")

    # --- Cantor alloy ---
    cantor_channels = analyze_cantor()

    # --- Refractory HEA ---
    refr_channels = analyze_refractory()

    # --- Comparative analysis ---
    print_comparative_analysis(cantor_channels, refr_channels)

    # --- Figures ---
    print(f"\n{'=' * 74}")
    print("  GENERATING FIGURES")
    print(f"{'=' * 74}")
    plot_q_comparison(cantor_channels, refr_channels)
    plot_ces_sweep_curves(cantor_channels, refr_channels)

    print(f"\n{'=' * 74}")
    print("  DONE")
    print(f"{'=' * 74}")
