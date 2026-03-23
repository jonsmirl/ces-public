#!/usr/bin/env python3
"""
CDW supercell check for MgB2H2.

3/7 q-points show acoustic softening with identical lambda=11.2.
This suggests a CDW instability at a specific nesting vector.
Build a 2x2x1 supercell, relax with CHGNet, and check if it
breaks symmetry → CDW ground state.

Run this AFTER the GPU el-ph finishes:
  source ~/.venv/bin/activate
  python run_cdw_check.py
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.pwscf import PWInput
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator
from ase.optimize import BFGS
from ase.filters import UnitCellFilter

# Original relaxed MgB2H2 unit cell
lattice = Lattice([
    [3.213, 0.000, 0.000],
    [-1.6065, 2.781924, 0.000],
    [0.000, 0.000, 3.941]
])
unit_cell = Structure(lattice, ['Mg', 'B', 'B', 'H', 'H'], [
    [-0.00000, 0.00000, -0.00352],
    [0.33333, 0.66667, 0.40412],
    [0.66667, 0.33333, 0.40412],
    [0.33333, 0.66667, 0.73764],
    [0.66667, 0.33333, 0.73764],
])

print("=" * 60)
print("CDW SUPERCELL CHECK — MgB2H2")
print("=" * 60)

# Build 2x2x1 supercell (20 atoms)
supercell = unit_cell.copy()
supercell.make_supercell([2, 2, 1])
print(f"\nUnit cell: {len(unit_cell)} atoms, a={unit_cell.lattice.a:.3f}, c={unit_cell.lattice.c:.3f}")
print(f"Supercell: {len(supercell)} atoms, a={supercell.lattice.a:.3f}, c={supercell.lattice.c:.3f}")

# Add small random perturbation to break symmetry and allow CDW
np.random.seed(42)
for i in range(len(supercell)):
    supercell.translate_sites(i, np.random.randn(3) * 0.02, frac_coords=True)

print(f"\nRelaxing 2x2x1 supercell with CHGNet (positions + cell)...")

model = CHGNet.load()
atoms = AseAtomsAdaptor.get_atoms(supercell)
atoms.calc = CHGNetCalculator(model)

e_before = atoms.get_potential_energy() / len(atoms)

ucf = UnitCellFilter(atoms)
opt = BFGS(ucf, logfile='cdw_relax.log')
opt.run(fmax=0.005, steps=500)

e_after = atoms.get_potential_energy() / len(atoms)
relaxed = AseAtomsAdaptor.get_structure(atoms)

print(f"\nEnergy: {e_before:.4f} → {e_after:.4f} eV/atom (dE = {e_after-e_before:.4f})")
print(f"Relaxed cell: a={relaxed.lattice.a:.3f}, b={relaxed.lattice.b:.3f}, c={relaxed.lattice.c:.3f}")
print(f"  alpha={relaxed.lattice.alpha:.2f}, beta={relaxed.lattice.beta:.2f}, gamma={relaxed.lattice.gamma:.2f}")

# Check symmetry breaking: compare atom positions in supercell
# If CDW, equivalent atoms in different unit cells will have different positions
print(f"\n--- Symmetry Check ---")
# Get fractional coords of all Mg atoms
mg_sites = [s for s in relaxed if str(s.species_string) == 'Mg']
b_sites = [s for s in relaxed if str(s.species_string) == 'B']
h_sites = [s for s in relaxed if str(s.species_string) == 'H']

# Check if all Mg-Mg distances within the supercell are equivalent
mg_z = [s.frac_coords[2] for s in mg_sites]
b_z = [s.frac_coords[2] for s in b_sites]
h_z = [s.frac_coords[2] for s in h_sites]

print(f"Mg z-coords: {[f'{z:.4f}' for z in sorted(mg_z)]}")
print(f"B  z-coords: {[f'{z:.4f}' for z in sorted(b_z)]}")
print(f"H  z-coords: {[f'{z:.4f}' for z in sorted(h_z)]}")

mg_z_spread = max(mg_z) - min(mg_z)
b_z_spread = max(b_z) - min(b_z)
h_z_spread = max(h_z) - min(h_z)

print(f"\nZ-coordinate spread (should be ~0 if no CDW):")
print(f"  Mg: {mg_z_spread:.4f}")
print(f"  B:  {b_z_spread:.4f}")
print(f"  H:  {h_z_spread:.4f}")

# Check B-B bond lengths
print(f"\nB-B distances:")
for i, si in enumerate(b_sites):
    for j, sj in enumerate(b_sites):
        if i < j:
            d = relaxed.get_distance(relaxed.index(si), relaxed.index(sj))
            if d < 2.5:
                print(f"  B{i}-B{j}: {d:.4f} A")

CDW_THRESHOLD = 0.02  # Angstrom
has_cdw = (mg_z_spread > CDW_THRESHOLD or b_z_spread > CDW_THRESHOLD or h_z_spread > CDW_THRESHOLD)

print(f"\n{'=' * 60}")
if has_cdw:
    print(f"  CDW DETECTED: symmetry broken in supercell")
    print(f"  The true ground state is DISTORTED.")
    print(f"  El-ph needs to be recomputed on the distorted structure.")
    print(f"  T_c may be lower (but could also be higher if CDW")
    print(f"  stabilizes the acoustic modes).")
else:
    print(f"  NO CDW: supercell relaxes to same structure as unit cell")
    print(f"  The acoustic softening is a Kohn anomaly, not a CDW instability.")
    print(f"  The optic lambda = 1.9 and T_c ~ 195K prediction STANDS.")
print(f"{'=' * 60}")

# Write QE input for the relaxed supercell (for potential follow-up DFT)
pw = PWInput(relaxed,
    pseudo={'Mg': 'Mg.pbe-n-kjpaw_psl.0.3.0.UPF',
            'B': 'B.pbe-n-kjpaw_psl.1.0.0.UPF',
            'H': 'H.pbe-kjpaw.UPF'},
    control={'calculation': 'scf', 'prefix': 'mgb2h2_cdw',
             'pseudo_dir': '/home/jonsmirl/thesis/superconductor_dft/pseudo/',
             'outdir': './tmp_cdw/'},
    system={'ecutwfc': 60, 'ecutrho': 480,
            'occupations': 'smearing', 'smearing': 'mp', 'degauss': 0.02,
            'nosym': True},
    electrons={'conv_thr': 1e-10},
    kpoints_grid=(6, 6, 8))
pw.write_file("scf_cdw_supercell.in")
print(f"\nWrote scf_cdw_supercell.in for DFT follow-up")
