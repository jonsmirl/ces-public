#!/usr/bin/env python3
"""
Monte Carlo lattice simulation to test whether q < 1 emerges from
nearest-neighbor correlations in a model HEA.

Simple cubic lattice with Lennard-Jones nearest-neighbor interactions.
Kawasaki dynamics (atom swaps) at fixed composition.  GPU-accelerated
via PyTorch when CUDA is available.

Key question: does the effective q parameter that best reproduces the
per-element energy partition deviate from 1, and does it track the
paper's formula q = 1 - alpha * delta^2?
"""

import time
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

# ---------------------------------------------------------------------------
# Element database (subset relevant to Cantor-family alloys)
# ---------------------------------------------------------------------------

ELEMENTS = {
    'Ni': {'r_pm': 124.0, 'mass': 58.69, 'T_m': 1728},
    'Co': {'r_pm': 125.0, 'mass': 58.93, 'T_m': 1768},
    'Cr': {'r_pm': 128.0, 'mass': 52.00, 'T_m': 2180},
    'Fe': {'r_pm': 126.0, 'mass': 55.85, 'T_m': 1811},
    'Mn': {'r_pm': 127.0, 'mass': 54.94, 'T_m': 1519},
}

# Ordered list for sub-selection by J
ELEMENT_ORDER = ['Ni', 'Co', 'Cr', 'Fe', 'Mn']

# ---------------------------------------------------------------------------
# CES / q-thermo helpers (pure numpy, used only for analysis)
# ---------------------------------------------------------------------------

def compute_delta(radii: np.ndarray, fracs: np.ndarray) -> float:
    """Atomic size mismatch delta = sqrt(sum c_i (1 - r_i/r_bar)^2)."""
    r_bar = np.sum(fracs * radii)
    return np.sqrt(np.sum(fracs * (1.0 - radii / r_bar) ** 2))


def q_from_delta(delta: float, alpha: float = 100.0) -> float:
    """q = 1 - alpha * delta^2."""
    return 1.0 - alpha * delta ** 2


def ces_energy(fracs: np.ndarray, energies: np.ndarray, q: float) -> float:
    """CES aggregate E = (sum c_j |E_j|^q)^{1/q}, sign preserved."""
    signs = np.sign(energies)
    abs_e = np.abs(energies)
    abs_e = np.maximum(abs_e, 1e-30)
    if abs(q) < 1e-10:
        return np.exp(np.sum(fracs * np.log(abs_e)))
    Z = np.sum(fracs * abs_e ** q)
    if Z <= 0:
        return np.nan
    # Return with average sign
    avg_sign = np.sign(np.sum(fracs * energies))
    return avg_sign * Z ** (1.0 / q)


# ---------------------------------------------------------------------------
# Lattice setup (PyTorch)
# ---------------------------------------------------------------------------

def build_neighbor_table(L: int, device: torch.device) -> torch.Tensor:
    """
    For a simple cubic L^3 lattice with periodic boundaries, return a
    (N, 6) tensor of neighbor indices.
    """
    N = L ** 3
    idx = torch.arange(N, device=device)
    x = idx % L
    y = (idx // L) % L
    z = idx // (L * L)

    def flat(xx, yy, zz):
        return (zz % L) * L * L + (yy % L) * L + (xx % L)

    neighbors = torch.stack([
        flat(x + 1, y, z), flat(x - 1, y, z),
        flat(x, y + 1, z), flat(x, y - 1, z),
        flat(x, y, z + 1), flat(x, y, z - 1),
    ], dim=1)  # (N, 6)
    return neighbors


def init_lattice(L: int, J: int, device: torch.device) -> torch.Tensor:
    """
    Create equimolar random assignment of J element types on L^3 sites.
    Returns integer tensor of shape (N,) with values in [0, J).
    """
    N = L ** 3
    # Build exactly equimolar then shuffle
    base = torch.arange(J, device=device).repeat((N + J - 1) // J)[:N]
    perm = torch.randperm(N, device=device)
    return base[perm]


# ---------------------------------------------------------------------------
# Energy computation
# ---------------------------------------------------------------------------

def build_pair_tables(J: int, sigma: torch.Tensor, eps: torch.Tensor,
                      a_lat: float, device: torch.device):
    """
    Precompute pair energy e_ij for every (i, j) element pair.
    sigma, eps are 1-D tensors of length J.
    Returns (J, J) tensor of pair energies at distance a_lat.
    """
    sig_ij = 0.5 * (sigma.unsqueeze(1) + sigma.unsqueeze(0))  # (J, J)
    eps_ij = torch.sqrt(eps.unsqueeze(1) * eps.unsqueeze(0))   # (J, J)
    ratio = sig_ij / a_lat
    r6 = ratio ** 6
    r12 = r6 ** 2
    e_pair = 4.0 * eps_ij * (r12 - r6)  # (J, J)
    return e_pair


def total_site_energies(lattice: torch.Tensor, neighbors: torch.Tensor,
                        e_pair: torch.Tensor) -> torch.Tensor:
    """
    Compute per-site energy (sum over 6 neighbors, double-counting factor
    handled later).  Returns (N,) tensor.
    """
    nb_types = lattice[neighbors]  # (N, 6)
    site_types = lattice.unsqueeze(1).expand_as(nb_types)  # (N, 6)
    pair_e = e_pair[site_types, nb_types]  # (N, 6)
    return pair_e.sum(dim=1)  # (N,)


# ---------------------------------------------------------------------------
# Monte Carlo sweep (batched Kawasaki swaps)
# ---------------------------------------------------------------------------

def mc_sweep(lattice: torch.Tensor, neighbors: torch.Tensor,
             e_pair: torch.Tensor, beta: float, batch_size: int,
             device: torch.device):
    """
    One sweep: attempt batch_size swap proposals sequentially in
    mini-batches to avoid conflicts.  For speed we do the whole batch
    at once (some swaps may conflict; this is an approximation that
    becomes exact in the large-N limit).
    """
    N = lattice.shape[0]

    # Pick random pairs
    idx_a = torch.randint(0, N, (batch_size,), device=device)
    idx_b = torch.randint(0, N, (batch_size,), device=device)

    type_a = lattice[idx_a]
    type_b = lattice[idx_b]

    # Skip same-type swaps (no effect)
    diff_mask = type_a != type_b

    if diff_mask.sum() == 0:
        return 0

    idx_a = idx_a[diff_mask]
    idx_b = idx_b[diff_mask]
    type_a = type_a[diff_mask]
    type_b = type_b[diff_mask]

    # Energy change for swapping a <-> b
    # For site a: neighbors don't move, but site type changes a_type -> b_type
    # dE_a = sum_over_nb [ e(b_type, nb) - e(a_type, nb) ]
    nb_a = neighbors[idx_a]  # (M, 6)
    nb_b = neighbors[idx_b]  # (M, 6)
    nb_types_a = lattice[nb_a]  # (M, 6)
    nb_types_b = lattice[nb_b]  # (M, 6)

    # New pair energies minus old
    dE_a = (e_pair[type_b.unsqueeze(1).expand_as(nb_types_a), nb_types_a]
            - e_pair[type_a.unsqueeze(1).expand_as(nb_types_a), nb_types_a]).sum(dim=1)
    dE_b = (e_pair[type_a.unsqueeze(1).expand_as(nb_types_b), nb_types_b]
            - e_pair[type_b.unsqueeze(1).expand_as(nb_types_b), nb_types_b]).sum(dim=1)

    # Correction: if a and b are neighbors, the above double-counts their
    # mutual interaction.  For simplicity in this model we ignore this
    # (probability of being neighbors is 6/N ~ 0.1%).
    dE = dE_a + dE_b

    # Metropolis acceptance
    accept = dE <= 0
    if beta < 1e30:
        boltz = torch.rand(dE.shape[0], device=device)
        accept = accept | (boltz < torch.exp(-beta * dE))
    else:
        pass  # T=0: accept only downhill

    # Apply accepted swaps
    acc_a = idx_a[accept]
    acc_b = idx_b[accept]
    acc_type_a = type_a[accept]
    acc_type_b = type_b[accept]

    lattice[acc_a] = acc_type_b
    lattice[acc_b] = acc_type_a

    return int(accept.sum().item())


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def measure(lattice: torch.Tensor, neighbors: torch.Tensor,
            e_pair: torch.Tensor, J: int):
    """
    Compute per-element average energy, energy std-dev, and
    Warren-Cowley short-range order parameters.
    """
    N = lattice.shape[0]
    site_e = total_site_energies(lattice, neighbors, e_pair)
    # Per-element means
    elem_energy = torch.zeros(J, device=lattice.device, dtype=torch.float64)
    elem_estd = torch.zeros(J, device=lattice.device, dtype=torch.float64)
    elem_count = torch.zeros(J, device=lattice.device, dtype=torch.float64)

    for j in range(J):
        mask = lattice == j
        if mask.sum() == 0:
            continue
        ej = site_e[mask].double()
        elem_energy[j] = ej.mean()
        elem_estd[j] = ej.std() if ej.numel() > 1 else 0.0
        elem_count[j] = mask.sum().double()

    # Warren-Cowley SRO: alpha_ij = 1 - P(j|i) / c_j
    # P(j|i) = (# i-j nn pairs) / (# i nn pairs)
    fracs = elem_count / N
    nb_types = lattice[neighbors]  # (N, 6)
    alpha_wc = torch.zeros(J, J, device=lattice.device, dtype=torch.float64)

    for i in range(J):
        mask_i = (lattice == i)
        if mask_i.sum() == 0:
            continue
        nb_of_i = nb_types[mask_i]  # (n_i, 6)
        total_nb_i = nb_of_i.numel()
        for j_idx in range(J):
            p_j_given_i = (nb_of_i == j_idx).sum().double() / total_nb_i
            c_j = fracs[j_idx]
            if c_j > 1e-10:
                alpha_wc[i, j_idx] = 1.0 - p_j_given_i / c_j

    total_E = site_e.sum().item() / 2.0  # remove double-counting

    return {
        'total_E': total_E,
        'elem_energy': elem_energy.cpu().numpy(),
        'elem_estd': elem_estd.cpu().numpy(),
        'elem_count': elem_count.cpu().numpy(),
        'alpha_wc': alpha_wc.cpu().numpy(),
    }


def run_pure_element_energies(L: int, J: int, sigma: torch.Tensor,
                              eps: torch.Tensor, a_lat: float,
                              device: torch.device) -> np.ndarray:
    """
    For each element type j in [0, J), run a pure lattice (all atoms are j)
    and return per-atom energy. Returns array of shape (J,).
    """
    neighbors = build_neighbor_table(L, device)
    e_pair = build_pair_tables(J, sigma, eps, a_lat, device)
    N = L ** 3
    pure_energies = np.zeros(J)
    for j in range(J):
        lattice = torch.full((N,), j, dtype=torch.long, device=device)
        site_e = total_site_energies(lattice, neighbors, e_pair)
        # Total energy with double-counting removed, per atom
        pure_energies[j] = site_e.sum().item() / (2.0 * N)
    return pure_energies


def fit_q(fracs: np.ndarray, pure_energies: np.ndarray,
          mixed_E_per_atom: float) -> float:
    """
    Find q such that CES(c_j, E_pure_j, q) best matches E_mixed per atom.

    pure_energies: per-atom energy of each element in a pure lattice.
    mixed_E_per_atom: per-atom energy of the mixed system.

    q=1 means rule-of-mixtures (no interaction effect).
    q<1 means mixing reduces energy (favorable interactions).
    q>1 means mixing enhances energy.
    """
    abs_e = np.abs(pure_energies)
    if np.min(abs_e) < 1e-30:
        return 1.0  # degenerate

    best_q = 1.0
    best_err = 1e30

    for q_try in np.linspace(-2.0, 3.0, 5001):
        try:
            pred = ces_energy(fracs, pure_energies, q_try)
            err = (pred - mixed_E_per_atom) ** 2
            if err < best_err:
                best_err = err
                best_q = q_try
        except Exception:
            continue

    return best_q


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_simulation(J: int, T_over_eps: float, L: int, n_sweeps: int,
                   n_equil: int, device: torch.device) -> dict:
    """
    Run MC simulation for a J-component equimolar alloy on L^3 simple
    cubic lattice at temperature T/epsilon.
    """
    symbols = ELEMENT_ORDER[:J]
    radii_pm = np.array([ELEMENTS[s]['r_pm'] for s in symbols])

    # LJ parameters: sigma proportional to atomic radius, epsilon = 1 for all
    # (we study T/epsilon so epsilon scale is arbitrary)
    sigma_base = radii_pm / radii_pm.mean()  # normalized around 1
    eps_base = np.ones(J)

    sigma_t = torch.tensor(sigma_base, dtype=torch.float32, device=device)
    eps_t = torch.tensor(eps_base, dtype=torch.float32, device=device)

    # Lattice constant = mean sigma (in reduced units)
    a_lat = 1.0  # reduced units

    N = L ** 3
    beta = 1.0 / T_over_eps if T_over_eps > 1e-10 else 1e30

    # Build system
    neighbors = build_neighbor_table(L, device)
    e_pair = build_pair_tables(J, sigma_t, eps_t, a_lat, device)
    lattice = init_lattice(L, J, device)

    batch_size = N  # one sweep = N attempted swaps

    # Equilibration
    for sw in range(n_equil):
        mc_sweep(lattice, neighbors, e_pair, beta, batch_size, device)

    # Production
    measurements = []
    for sw in range(n_sweeps):
        n_acc = mc_sweep(lattice, neighbors, e_pair, beta, batch_size, device)
        if (sw + 1) % max(1, n_sweeps // 5) == 0:
            m = measure(lattice, neighbors, e_pair, J)
            measurements.append(m)

    # Final detailed measurement
    final = measure(lattice, neighbors, e_pair, J)

    # Compute pure-element reference energies
    pure_energies = run_pure_element_energies(L, J, sigma_t, eps_t, a_lat, device)

    # Compute q_emergent: fit CES of pure-element energies to mixed energy
    fracs = np.ones(J) / J
    mixed_E_per_atom = final['total_E'] / N
    q_emergent = fit_q(fracs, pure_energies, mixed_E_per_atom)

    # q from formula
    delta = compute_delta(radii_pm, fracs)
    q_formula = q_from_delta(delta)

    return {
        'J': J, 'T_over_eps': T_over_eps, 'L': L, 'N': N,
        'symbols': symbols,
        'delta': delta, 'delta_pct': delta * 100,
        'q_formula': q_formula,
        'q_emergent': q_emergent,
        'total_E': final['total_E'],
        'mixed_E_per_atom': mixed_E_per_atom,
        'pure_energies': pure_energies,
        'elem_energy': final['elem_energy'],
        'elem_estd': final['elem_estd'],
        'alpha_wc': final['alpha_wc'],
    }


def main():
    t0 = time.time()

    # Device selection
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU (will be slower)")

    # Simulation parameters
    L = 20              # lattice size -> 8000 atoms
    n_sweeps = 1000     # production sweeps
    n_equil = 200       # equilibration sweeps
    J_values = [2, 3, 4, 5]
    T_values = [0.1, 0.5, 1.0, 2.0]

    print(f"\nLattice: {L}^3 = {L**3} sites (simple cubic, PBC)")
    print(f"Sweeps: {n_equil} equilibration + {n_sweeps} production")
    print(f"J values: {J_values}")
    print(f"T/epsilon values: {T_values}")
    print()

    results = []
    total_runs = len(J_values) * len(T_values)
    run_idx = 0

    for J in J_values:
        for T in T_values:
            run_idx += 1
            symbols = ELEMENT_ORDER[:J]
            print(f"[{run_idx}/{total_runs}] J={J} ({'-'.join(symbols)}), "
                  f"T/eps={T:.1f} ...", end=' ', flush=True)

            t1 = time.time()
            res = run_simulation(J, T, L, n_sweeps, n_equil, device)
            dt = time.time() - t1

            print(f"done ({dt:.1f}s)  q_em={res['q_emergent']:.4f}  "
                  f"q_form={res['q_formula']:.4f}  "
                  f"delta={res['delta_pct']:.2f}%")
            results.append(res)

    # -----------------------------------------------------------------------
    # Print results table
    # -----------------------------------------------------------------------
    print(f"\n{'='*78}")
    print("  RESULTS: q_emergent(J, T/eps) vs q_formula(delta)")
    print(f"{'='*78}")
    print(f"\n  {'J':>3s}  {'Elements':<20s}  {'delta%':>7s}  {'q_form':>7s}  ", end='')
    for T in T_values:
        print(f"{'T='+str(T):>8s}", end='  ')
    print()
    print(f"  {'':>3s}  {'':>20s}  {'':>7s}  {'':>7s}  ", end='')
    for T in T_values:
        print(f"{'q_em':>8s}", end='  ')
    print()
    print(f"  {'-'*74}")

    for J in J_values:
        j_results = [r for r in results if r['J'] == J]
        syms = '-'.join(j_results[0]['symbols'])
        delta_pct = j_results[0]['delta_pct']
        q_form = j_results[0]['q_formula']
        print(f"  {J:3d}  {syms:<20s}  {delta_pct:7.2f}  {q_form:7.4f}  ", end='')
        for T in T_values:
            r = [x for x in j_results if abs(x['T_over_eps'] - T) < 0.01][0]
            print(f"{r['q_emergent']:8.4f}", end='  ')
        print()

    # -----------------------------------------------------------------------
    # Warren-Cowley SRO summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*78}")
    print("  SHORT-RANGE ORDER (Warren-Cowley alpha, diagonal = self)")
    print(f"{'='*78}")

    for J in J_values:
        # Show SRO for lowest temperature
        r = [x for x in results if x['J'] == J
             and abs(x['T_over_eps'] - T_values[0]) < 0.01][0]
        print(f"\n  J={J} ({'-'.join(r['symbols'])}), T/eps={T_values[0]}:")
        alpha = r['alpha_wc']
        header = "       " + "  ".join(f"{s:>6s}" for s in r['symbols'])
        print(header)
        for i, si in enumerate(r['symbols']):
            row = f"  {si:>4s} " + "  ".join(f"{alpha[i,j]:6.3f}"
                                               for j in range(J))
            print(row)

    # -----------------------------------------------------------------------
    # Energy spread (lattice distortion proxy)
    # -----------------------------------------------------------------------
    print(f"\n{'='*78}")
    print("  PER-ELEMENT ENERGY SPREAD (std dev, proxy for lattice distortion)")
    print(f"{'='*78}")

    for J in J_values:
        r_low = [x for x in results if x['J'] == J
                 and abs(x['T_over_eps'] - T_values[0]) < 0.01][0]
        r_high = [x for x in results if x['J'] == J
                  and abs(x['T_over_eps'] - T_values[-1]) < 0.01][0]
        print(f"\n  J={J} ({'-'.join(r_low['symbols'])}):")
        print(f"    {'Element':<6s}  {'<E>_low':>10s}  {'std_low':>10s}  "
              f"{'<E>_high':>10s}  {'std_high':>10s}")
        for idx, s in enumerate(r_low['symbols']):
            print(f"    {s:<6s}  {r_low['elem_energy'][idx]:10.4f}  "
                  f"{r_low['elem_estd'][idx]:10.4f}  "
                  f"{r_high['elem_energy'][idx]:10.4f}  "
                  f"{r_high['elem_estd'][idx]:10.4f}")

    # -----------------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------------
    print(f"\n{'='*78}")
    print("  INTERPRETATION")
    print(f"{'='*78}")

    q_em_all = [r['q_emergent'] for r in results]
    q_form_all = [r['q_formula'] for r in results]
    mean_em = np.mean(q_em_all)
    std_em = np.std(q_em_all)
    mean_form = np.mean(q_form_all)

    print(f"\n  q_emergent: mean = {mean_em:.4f}, std = {std_em:.4f}")
    print(f"  q_formula:  mean = {mean_form:.4f}")

    if abs(mean_em - 1.0) < 0.05:
        print("\n  >> q_emergent ~ 1 for all cases.")
        print("     NN pair correlations alone do NOT generate q < 1.")
        print("     The paper's q < 1 may require many-body or elastic effects.")
    elif abs(mean_em - mean_form) < 0.1:
        print("\n  >> q_emergent tracks q_formula reasonably well!")
        print("     NN correlations partially explain q < 1.")
    else:
        print(f"\n  >> q_emergent ({mean_em:.3f}) differs from "
              f"q_formula ({mean_form:.3f}).")
        print("     The relationship is non-trivial.")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Monte Carlo Lattice Simulation: Emergent q Parameter',
                 fontsize=14, fontweight='bold')

    # Panel 1: q_emergent vs J for different T
    ax = axes[0, 0]
    for T in T_values:
        q_em = [r['q_emergent'] for r in results if abs(r['T_over_eps'] - T) < 0.01]
        ax.plot(J_values, q_em, 'o-', label=f'T/eps={T}', markersize=6)
    # Also plot q_formula
    q_form = [results[i * len(T_values)]['q_formula'] for i in range(len(J_values))]
    ax.plot(J_values, q_form, 's--', color='black', label='q_formula(delta)',
            markersize=8, zorder=5)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='q=1 (rule of mixtures)')
    ax.set_xlabel('Number of elements J')
    ax.set_ylabel('q parameter')
    ax.set_title('q_emergent vs J')
    ax.legend(fontsize=8)
    ax.set_xticks(J_values)
    ax.grid(True, alpha=0.3)

    # Panel 2: q_emergent vs T for different J
    ax = axes[0, 1]
    for J in J_values:
        q_em = [r['q_emergent'] for r in results if r['J'] == J]
        T_list = [r['T_over_eps'] for r in results if r['J'] == J]
        syms = '-'.join(ELEMENT_ORDER[:J])
        ax.plot(T_list, q_em, 'o-', label=f'J={J} ({syms})', markersize=6)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('T / epsilon')
    ax.set_ylabel('q_emergent')
    ax.set_title('q_emergent vs Temperature')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Energy spread vs J (lattice distortion proxy)
    ax = axes[1, 0]
    for T in T_values:
        spreads = []
        for J in J_values:
            r = [x for x in results if x['J'] == J
                 and abs(x['T_over_eps'] - T) < 0.01][0]
            # Average std over elements
            spreads.append(np.mean(r['elem_estd']))
        ax.plot(J_values, spreads, 'o-', label=f'T/eps={T}', markersize=6)
    ax.set_xlabel('Number of elements J')
    ax.set_ylabel('Mean per-element energy std dev')
    ax.set_title('Energy Spread (Distortion Proxy)')
    ax.legend(fontsize=8)
    ax.set_xticks(J_values)
    ax.grid(True, alpha=0.3)

    # Panel 4: Warren-Cowley SRO (off-diagonal mean) vs J
    ax = axes[1, 1]
    for T in T_values:
        sro_vals = []
        for J in J_values:
            r = [x for x in results if x['J'] == J
                 and abs(x['T_over_eps'] - T) < 0.01][0]
            alpha = r['alpha_wc']
            # Mean absolute off-diagonal SRO
            mask = ~np.eye(J, dtype=bool)
            sro_vals.append(np.mean(np.abs(alpha[mask])))
        ax.plot(J_values, sro_vals, 'o-', label=f'T/eps={T}', markersize=6)
    ax.set_xlabel('Number of elements J')
    ax.set_ylabel('Mean |alpha_WC| (off-diagonal)')
    ax.set_title('Short-Range Order Magnitude')
    ax.legend(fontsize=8)
    ax.set_xticks(J_values)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, 'lattice_mc.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n  Figure saved to: {fig_path}")

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
