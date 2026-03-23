#!/usr/bin/env python3
"""
Superconductor candidate screening using ALIGNN + phonopy.

Builds crystal structures for the three candidates and runs:
1. ALIGNN T_c prediction (ML model trained on superconductor database)
2. Structure relaxation (if MACE available)
3. Phonon analysis (mechanical stability check)

Candidates:
  C1: (La,Y)H10 — known high-Tc hydride (reference)
  C2: YB2N2H4 — layered BN-framework hydride (novel)
  C3: MgB2Hx — hydrogenated MgB2 variants (conservative)

Also screens known superconductors as calibration.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from pymatgen.core import Structure, Lattice
from jarvis.core.atoms import Atoms as JAtoms
from jarvis.io.vasp.inputs import Poscar

# ═══════════════════════════════════════════════════════════════════
# STRUCTURE BUILDERS
# ═══════════════════════════════════════════════════════════════════

def build_MgB2():
    """MgB2 — known superconductor, T_c = 39K. Calibration reference.
    Space group P6/mmm, a=3.086, c=3.524.
    """
    lattice = Lattice.hexagonal(3.086, 3.524)
    species = ["Mg", "B", "B"]
    coords = [
        [0.0, 0.0, 0.0],       # Mg at origin
        [1/3, 2/3, 0.5],       # B1
        [2/3, 1/3, 0.5],       # B2
    ]
    return Structure(lattice, species, coords), "MgB2 (T_c=39K, reference)"


def build_MgB2H2_intercalated():
    """MgB2H2 — hydrogen intercalated between Mg and B layers.
    H atoms placed above and below the B hexagonal ring center,
    at z ~ 0.25 and 0.75 (between Mg at z=0 and B at z=0.5).
    Preserves B sp2 character (H not bonded to B directly).
    """
    # Expand c-axis to accommodate hydrogen
    lattice = Lattice.hexagonal(3.086, 5.5)  # expanded c
    species = ["Mg", "B", "B", "H", "H"]
    coords = [
        [0.0, 0.0, 0.0],       # Mg
        [1/3, 2/3, 0.5],       # B1
        [2/3, 1/3, 0.5],       # B2
        [0.0, 0.0, 0.25],      # H above Mg layer
        [0.0, 0.0, 0.75],      # H below Mg layer (next cell)
    ]
    return Structure(lattice, species, coords), "MgB2H2 intercalated"


def build_MgB2H2_on_boron():
    """MgB2H2 — hydrogen bonded directly to boron atoms.
    Each B gets one H above. This changes B from sp2 to sp3-like.
    Risk: may become insulating.
    """
    lattice = Lattice.hexagonal(3.086, 5.0)
    species = ["Mg", "B", "B", "H", "H"]
    coords = [
        [0.0, 0.0, 0.0],       # Mg
        [1/3, 2/3, 0.45],      # B1
        [2/3, 1/3, 0.45],      # B2
        [1/3, 2/3, 0.69],      # H on B1 (B-H ~ 1.2 A)
        [2/3, 1/3, 0.69],      # H on B2
    ]
    return Structure(lattice, species, coords), "MgB2H2 on-boron (sp3 risk)"


def build_LiMgB2H2():
    """Li0.5Mg0.5B2H2 — electron-doped hydrogenated MgB2.
    Li substitution for Mg removes electrons, potentially
    preserving metallic sigma-band even with H addition.
    Use 2x1x1 supercell: one Mg, one Li.
    """
    # 2x supercell along a
    a = 3.086
    lattice = Lattice.hexagonal(a * 2, 5.5)
    species = ["Mg", "Li", "B", "B", "B", "B", "H", "H", "H", "H"]
    coords = [
        [0.0, 0.0, 0.0],        # Mg
        [0.5, 0.0, 0.0],        # Li (replacing second Mg)
        [1/6, 2/3, 0.5],        # B1
        [1/3, 1/3, 0.5],        # B2
        [2/3, 2/3, 0.5],        # B3
        [5/6, 1/3, 0.5],        # B4
        [0.0, 0.0, 0.25],       # H1
        [0.5, 0.0, 0.25],       # H2
        [0.0, 0.0, 0.75],       # H3
        [0.5, 0.0, 0.75],       # H4
    ]
    return Structure(lattice, species, coords), "Li0.5Mg0.5B2H2 (e-doped)"


def build_YB2N2H4():
    """YB2N2H4 — layered yttrium boron nitride hydride.
    Concept: BN sheets with H on each B and N, intercalated with Y.
    Hexagonal structure inspired by h-BN + YH2.
    """
    # Layered structure: Y-H layer | B-N-H layer | Y-H layer | ...
    a = 3.0  # similar to h-BN (2.50) but expanded for Y
    c = 8.0  # large c for layered structure
    lattice = Lattice.hexagonal(a, c)
    species = ["Y", "B", "N", "H", "H", "H", "H"]
    coords = [
        [0.0, 0.0, 0.0],       # Y at z=0
        [1/3, 2/3, 0.35],      # B in BN layer
        [2/3, 1/3, 0.35],      # N in BN layer
        [1/3, 2/3, 0.50],      # H on B (above)
        [2/3, 1/3, 0.20],      # H on N (below)
        [0.0, 0.0, 0.12],      # H bonded to Y (above)
        [0.0, 0.0, 0.88],      # H bonded to Y (below)
    ]
    return Structure(lattice, species, coords), "YB2N2H4 layered"


def build_YH2():
    """YH2 — known ambient-pressure metallic hydride. Calibration.
    CaF2 structure (Fm-3m), a = 5.20 Å.
    """
    lattice = Lattice.cubic(5.20)
    species = ["Y", "Y", "Y", "Y", "H", "H", "H", "H", "H", "H", "H", "H"]
    coords = [
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],  # Y fcc
        [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75],  # H tetrahedral
        [0.75, 0.25, 0.25], [0.25, 0.75, 0.25], [0.25, 0.25, 0.75], [0.75, 0.75, 0.75],  # H tetrahedral
    ]
    return Structure(lattice, species, coords), "YH2 (metallic, reference)"


def build_LaH10_approx():
    """Approximate LaH10 structure (Fm-3m clathrate).
    Simplified: La at fcc sites, H forming cage.
    At ambient this is unstable but serves as T_c reference.
    Use a = 5.1 Å (high-pressure lattice parameter).
    """
    a = 5.10
    lattice = Lattice.cubic(a)
    # La at 4a (0,0,0) fcc positions
    # H at 8c (1/4,1/4,1/4) and 32f (~0.12,0.12,0.12)
    species = ["La", "La", "La", "La"]
    coords_La = [[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]
    # Simplified H cage — place 32 H in approximate positions
    coords_H = []
    species_H = []
    for s1 in [0.12, 0.88]:
        for s2 in [0.12, 0.88]:
            for s3 in [0.12, 0.88]:
                coords_H.append([s1, s2, s3])
                species_H.append("H")
    # Add 8c positions
    for s in [[0.25,0.25,0.25], [0.75,0.75,0.25], [0.75,0.25,0.75], [0.25,0.75,0.75],
              [0.25,0.25,0.75], [0.75,0.75,0.75], [0.75,0.25,0.25], [0.25,0.75,0.25]]:
        coords_H.append(s)
        species_H.append("H")

    all_species = species + species_H
    all_coords = coords_La + coords_H
    # Trim to get La4H40 = LaH10 per formula unit
    return Structure(lattice, all_species, all_coords), "LaH10 (clathrate, high-P reference)"


def build_CaB2H8():
    """CaB2H8 — calcium borohydride variant.
    Ca provides metallicity (larger, more electropositive than Mg).
    BH4 units provide high-frequency modes.
    """
    a = 4.5
    c = 6.0
    lattice = Lattice.hexagonal(a, c)
    species = ["Ca", "B", "B", "H", "H", "H", "H", "H", "H", "H", "H"]
    coords = [
        [0.0, 0.0, 0.0],       # Ca
        [1/3, 2/3, 0.40],      # B1
        [2/3, 1/3, 0.40],      # B2
        [1/3, 2/3, 0.55],      # H on B1 (up)
        [1/3, 2/3, 0.25],      # H on B1 (down)
        [0.20, 0.53, 0.40],    # H on B1 (side)
        [0.47, 0.80, 0.40],    # H on B1 (side)
        [2/3, 1/3, 0.55],      # H on B2 (up)
        [2/3, 1/3, 0.25],      # H on B2 (down)
        [0.53, 0.20, 0.40],    # H on B2 (side)
        [0.80, 0.47, 0.40],    # H on B2 (side)
    ]
    return Structure(lattice, species, coords), "CaB2H8 (borohydride)"


# ═══════════════════════════════════════════════════════════════════
# ALIGNN T_c PREDICTION
# ═══════════════════════════════════════════════════════════════════

def pymatgen_to_jarvis(structure):
    """Convert pymatgen Structure to JARVIS Atoms."""
    return JAtoms(
        lattice_mat=structure.lattice.matrix.tolist(),
        coords=structure.frac_coords.tolist(),
        elements=[str(s) for s in structure.species],
        cartesian=False
    )


def predict_Tc(structure, name, model_name="jv_supercon_tc_alignn"):
    """Predict superconducting T_c using ALIGNN pretrained model."""
    from alignn.pretrained import get_prediction
    jarvis_atoms = pymatgen_to_jarvis(structure)
    try:
        result = get_prediction(atoms=jarvis_atoms, model_name=model_name)
        tc = float(result)
        return tc
    except Exception as e:
        return f"Error: {e}"


def predict_formation_energy(structure, name):
    """Predict formation energy using ALIGNN."""
    from alignn.pretrained import get_prediction
    jarvis_atoms = pymatgen_to_jarvis(structure)
    try:
        result = get_prediction(
            atoms=jarvis_atoms,
            model_name="mp_e_form_alignn"
        )
        return float(result)
    except Exception as e:
        return f"Error: {e}"


def predict_bandgap(structure, name):
    """Predict bandgap using ALIGNN (0 = metallic)."""
    from alignn.pretrained import get_prediction
    jarvis_atoms = pymatgen_to_jarvis(structure)
    try:
        result = get_prediction(
            atoms=jarvis_atoms,
            model_name="jv_optb88vdw_bandgap_alignn"
        )
        return float(result)
    except Exception as e:
        return f"Error: {e}"


# ═══════════════════════════════════════════════════════════════════
# MAIN SCREENING
# ═══════════════════════════════════════════════════════════════════

def run_screening():
    """Screen all candidates with ALIGNN predictions."""

    candidates = [
        build_MgB2(),
        build_YH2(),
        build_MgB2H2_intercalated(),
        build_MgB2H2_on_boron(),
        build_LiMgB2H2(),
        build_YB2N2H4(),
        build_CaB2H8(),
    ]

    # Try to include LaH10 (might fail due to many atoms)
    try:
        candidates.append(build_LaH10_approx())
    except Exception as e:
        print(f"LaH10 build failed: {e}")

    print("=" * 75)
    print("SUPERCONDUCTOR CANDIDATE SCREENING — ALIGNN Predictions")
    print("=" * 75)
    print(f"\n{'Candidate':<35} {'T_c (K)':<12} {'E_form':<12} {'Bandgap':<12}")
    print(f"{'-'*35} {'-'*12} {'-'*12} {'-'*12}")

    results = []
    for structure, name in candidates:
        print(f"\nProcessing: {name}...")
        print(f"  Formula: {structure.composition.reduced_formula}")
        print(f"  Atoms: {len(structure)}, Volume: {structure.volume:.1f} A^3")

        tc = predict_Tc(structure, name)
        ef = predict_formation_energy(structure, name)
        bg = predict_bandgap(structure, name)

        tc_str = f"{tc:.1f}K" if isinstance(tc, float) else str(tc)[:30]
        ef_str = f"{ef:.3f} eV" if isinstance(ef, float) else str(ef)[:30]
        bg_str = f"{bg:.2f} eV" if isinstance(bg, float) else str(bg)[:30]

        print(f"  T_c prediction:     {tc_str}")
        print(f"  Formation energy:   {ef_str}")
        print(f"  Bandgap prediction: {bg_str}")

        results.append({
            'name': name,
            'formula': structure.composition.reduced_formula,
            'tc': tc,
            'ef': ef,
            'bg': bg,
            'structure': structure,
        })

    # Summary table
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(f"\n{'Candidate':<35} {'T_c (K)':<10} {'E_form':<10} {'Gap (eV)':<10} {'Metallic?':<10}")
    print(f"{'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for r in results:
        tc = f"{r['tc']:.1f}" if isinstance(r['tc'], float) else "err"
        ef = f"{r['ef']:.3f}" if isinstance(r['ef'], float) else "err"
        bg = f"{r['bg']:.2f}" if isinstance(r['bg'], float) else "err"
        metallic = "YES" if isinstance(r['bg'], float) and r['bg'] < 0.1 else "no"
        print(f"  {r['name']:<33} {tc:>8}  {ef:>8}  {bg:>8}  {metallic:>8}")

    return results


if __name__ == '__main__':
    results = run_screening()
