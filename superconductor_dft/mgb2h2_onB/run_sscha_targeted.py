#!/usr/bin/env python3
"""
Targeted SSCHA check for MgB2H2 soft acoustic modes.

The harmonic calculation shows 3/8 q-points with acoustic softening
(lambda_acoustic ~ 11.2 at Q3, Q5, Q7). This script checks whether
anharmonic effects stiffen these modes (stabilizing the structure)
or leave them soft (confirming CDW tendency).

Strategy: Instead of full SSCHA (days), do targeted anharmonic
sampling at the soft q-vector only.

Prerequisites:
  pip install python-sscha  (or from github)
  Completed el-ph calculation (for force constants)

Run AFTER GPU el-ph and CDW check:
  python run_sscha_targeted.py
"""

import os
import sys
import numpy as np

# Check if sscha is installed
try:
    import sscha
    print(f"SSCHA loaded OK")
except ImportError:
    print("SSCHA not installed. Install with:")
    print("  pip install python-sscha")
    print("  # or: pip install git+https://github.com/SSCHAcode/python-sscha.git")
    print("\nAlso needs:")
    print("  pip install cellconstructor")
    print("  pip install ase")
    sys.exit(1)

import cellconstructor as CC
import cellconstructor.Phonons
import sscha.Ensemble
import sscha.SchaMinimizer

# ============================================================
# CONFIGURATION
# ============================================================

# QE settings
QE_BIN = "/tmp/q-e-qe-7.4.1/bin"
PSEUDO_DIR = "/home/jonsmirl/thesis/superconductor_dft/pseudo"
WORK_DIR = "/home/jonsmirl/thesis/superconductor_dft/mgb2h2_onB/sscha_work"
os.makedirs(WORK_DIR, exist_ok=True)

# SSCHA parameters
N_CONFIGS = 50       # Number of stochastic configurations (start small)
T_TARGET = 300.0     # Temperature in Kelvin (room temperature)
N_STEPS = 20         # SSCHA minimization steps
SUPERCELL = [2, 2, 2]  # Must match the 2x2x2 q-grid from el-ph

# Structure
CELL_PARAMS = np.array([
    [3.213, 0.000, 0.000],
    [-1.6065, 2.781924, 0.000],
    [0.000, 0.000, 3.941]
])
POSITIONS_FRAC = np.array([
    [-0.00000, 0.00000, -0.00352],
    [0.33333, 0.66667, 0.40412],
    [0.66667, 0.33333, 0.40412],
    [0.33333, 0.66667, 0.73764],
    [0.66667, 0.33333, 0.73764],
])
SPECIES = ['Mg', 'B', 'B', 'H', 'H']
MASSES = [24.305, 10.811, 10.811, 1.008, 1.008]

print("=" * 60)
print("TARGETED SSCHA — Anharmonic check for soft modes")
print("=" * 60)
print(f"  Supercell: {SUPERCELL}")
print(f"  Temperature: {T_TARGET} K")
print(f"  Configurations: {N_CONFIGS}")
print(f"  Steps: {N_STEPS}")

# ============================================================
# STEP 1: Load harmonic phonons from QE output
# ============================================================
print("\n--- Step 1: Loading harmonic dynamical matrices ---")

DYN_PREFIX = "/home/jonsmirl/thesis/superconductor_dft/mgb2h2_onB/mgb2h2r.dyn"
try:
    # Load all dynamical matrices from the 2x2x2 q-grid
    harmonic_phonons = CC.Phonons.Phonons(DYN_PREFIX, nqirr=8)
    print(f"  Loaded {harmonic_phonons.nqirr} irreducible q-points")

    # Check for imaginary frequencies
    w, pols = harmonic_phonons.DiagonalizeSupercell()
    n_imag = np.sum(w < 0)
    print(f"  Supercell frequencies: {len(w)} modes")
    print(f"  Imaginary modes: {n_imag}")
    if n_imag > 0:
        print(f"  Lowest frequency: {w[0]:.2f} cm-1")
        print(f"  These will be stabilized by SSCHA if anharmonicity is strong enough")
except Exception as e:
    print(f"  Error loading dynamical matrices: {e}")
    print(f"  Make sure the el-ph calculation has finished and dyn files exist")
    print(f"  Expected files: {DYN_PREFIX}1 through {DYN_PREFIX}8")
    sys.exit(1)

# ============================================================
# STEP 2: Generate stochastic ensemble
# ============================================================
print("\n--- Step 2: Generating stochastic ensemble ---")

# Force positive definite (soft acoustic modes have tiny imaginary freq)
harmonic_phonons.Symmetrize()
harmonic_phonons.ForcePositiveDefinite()

ensemble = sscha.Ensemble.Ensemble(harmonic_phonons, T_TARGET, SUPERCELL)
ensemble.generate(N_CONFIGS)
print(f"  Generated {N_CONFIGS} configurations at {T_TARGET} K")
print(f"  Supercell atoms: {ensemble.structures[0].N_atoms}")

# ============================================================
# STEP 3: Compute forces with QE (or CHGNet for speed)
# ============================================================
print("\n--- Step 3: Computing forces ---")
print("  Option A: QE (accurate, slow ~6-12 hours)")
print("  Option B: CHGNet (approximate, fast ~10 min)")
print()

USE_CHGNET = True  # Set to False for QE (production quality)

if USE_CHGNET:
    print("  Using CHGNet for fast screening...")
    from ase.calculators.calculator import Calculator
    from chgnet.model.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator

    model = CHGNet.load()
    calc = CHGNetCalculator(model)

    for i, structure in enumerate(ensemble.structures):
        # Convert to ASE
        from ase import Atoms
        atoms = Atoms(
            symbols=[SPECIES[j % len(SPECIES)] for j in range(structure.N_atoms)],
            positions=structure.coords,
            cell=structure.unit_cell,
            pbc=True
        )
        atoms.calc = calc
        forces = atoms.get_forces()
        energy = atoms.get_potential_energy()
        # Store in ensemble
        ensemble.energies[i] = energy
        ensemble.forces[i] = forces
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{N_CONFIGS} done")

    print(f"  All {N_CONFIGS} configurations computed")
else:
    print("  QE force calculations — writing input files...")
    # Write QE inputs for each configuration
    for i, structure in enumerate(ensemble.structures):
        config_dir = os.path.join(WORK_DIR, f"config_{i:04d}")
        os.makedirs(config_dir, exist_ok=True)
        # Write QE SCF input
        with open(os.path.join(config_dir, "scf.in"), "w") as f:
            f.write(f"""&CONTROL
  calculation = 'scf'
  prefix = 'sscha_{i:04d}'
  pseudo_dir = '{PSEUDO_DIR}/'
  outdir = './tmp/'
  tprnfor = .true.
/
&SYSTEM
  ibrav = 0, nat = {structure.N_atoms}, ntyp = 3
  ecutwfc = 60.0, ecutrho = 480.0
  occupations = 'smearing', smearing = 'mp', degauss = 0.02
  nosym = .true.
/
&ELECTRONS
  conv_thr = 1.0d-8, mixing_beta = 0.5
/
CELL_PARAMETERS {{angstrom}}
""")
            for row in structure.unit_cell:
                f.write(f"  {row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}\n")
            f.write("ATOMIC_SPECIES\n")
            f.write("  Mg  24.305  Mg.pbe-n-kjpaw_psl.0.3.0.UPF\n")
            f.write("  B   10.811  B.pbe-n-kjpaw_psl.1.0.0.UPF\n")
            f.write("  H    1.008  H.pbe-kjpaw.UPF\n")
            f.write("ATOMIC_POSITIONS {angstrom}\n")
            for j in range(structure.N_atoms):
                spec = SPECIES[j % len(SPECIES)]
                f.write(f"  {spec}  {structure.coords[j,0]:.6f}  {structure.coords[j,1]:.6f}  {structure.coords[j,2]:.6f}\n")
            f.write("K_POINTS {automatic}\n  6 6 8  0 0 0\n")

    print(f"  Wrote {N_CONFIGS} QE inputs in {WORK_DIR}/config_XXXX/")
    print(f"  Run them with:")
    print(f"    for d in {WORK_DIR}/config_*/; do")
    print(f"      cd $d && mkdir -p tmp")
    print(f"      mpirun -np 1 {QE_BIN}/pw.x -in scf.in > scf.out 2>&1")
    print(f"      cd -")
    print(f"    done")
    print(f"  Then re-run this script to continue from step 4")
    sys.exit(0)

# ============================================================
# STEP 4: SSCHA minimization
# ============================================================
print("\n--- Step 4: SSCHA minimization ---")

minim = sscha.SchaMinimizer.SSCHA_Minimizer(ensemble)
minim.init()
minim.set_minimization_step(0.1)

# Run minimization
minim.run(N_STEPS)

# Get anharmonic phonons
anharmonic_phonons = minim.dyn

print("\n--- Results ---")
w_harm, _ = harmonic_phonons.DiagonalizeSupercell()
w_anharm, _ = anharmonic_phonons.DiagonalizeSupercell()

print(f"\n  {'Mode':>4} {'Harmonic':>12} {'Anharmonic':>12} {'Shift':>10}")
print(f"  {'-'*4} {'-'*12} {'-'*12} {'-'*10}")
for i in range(min(20, len(w_harm))):
    shift = w_anharm[i] - w_harm[i]
    flag = " <-- STIFFENED" if w_harm[i] < 50 and w_anharm[i] > w_harm[i] + 10 else ""
    flag = " <-- STILL SOFT" if w_anharm[i] < 0 else flag
    print(f"  {i+1:>4} {w_harm[i]:>12.2f} {w_anharm[i]:>12.2f} {shift:>+10.2f}{flag}")

n_imag_harm = np.sum(w_harm < 0)
n_imag_anharm = np.sum(w_anharm < 0)
print(f"\n  Imaginary modes: {n_imag_harm} (harmonic) → {n_imag_anharm} (anharmonic)")

if n_imag_anharm == 0 and n_imag_harm > 0:
    print(f"\n  ANHARMONICITY STABILIZES THE STRUCTURE!")
    print(f"  The soft modes are stiffened by zero-point motion.")
    print(f"  The hexagonal MgB2H2 phase is stable at {T_TARGET}K.")
elif n_imag_anharm > 0:
    print(f"\n  Soft modes persist even with anharmonicity.")
    print(f"  CDW distortion is likely the true ground state.")
else:
    print(f"\n  No imaginary modes in either case — structure is stable.")
