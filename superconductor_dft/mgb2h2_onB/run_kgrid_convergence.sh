#!/bin/bash
# K-grid convergence test for MgB2H2
# Run AFTER GPU el-ph finishes
# Tests 12x12x8 → 16x16x12 → 20x20x14 → 24x24x16
# Compares N(E_F) at each grid size

export NVHPC=$HOME/nvhpc
export NVHPC_ROOT=$NVHPC/Linux_x86_64/24.11
export PATH=$NVHPC_ROOT/comm_libs/12.6/openmpi4/openmpi-4.1.5/bin:$NVHPC_ROOT/compilers/bin:$PATH
export LD_LIBRARY_PATH=$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/cuda/12.6/lib64:$NVHPC_ROOT/comm_libs/12.6/openmpi4/openmpi-4.1.5/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
QE=/tmp/q-e-qe-7.4.1/bin
PSEUDO=/home/jonsmirl/thesis/superconductor_dft/pseudo
cd /home/jonsmirl/thesis/superconductor_dft/mgb2h2_onB

for GRID in "12 12 8" "16 16 12" "20 20 14" "24 24 16"; do
    TAG=$(echo $GRID | tr ' ' 'x')
    echo "=== K-grid: $TAG ==="

    cat > scf_k${TAG}.in << ENDINPUT
&CONTROL
  calculation = 'scf'
  prefix = 'mgb2h2_k${TAG}'
  pseudo_dir = '${PSEUDO}/'
  outdir = './tmp_k${TAG}/'
/
&SYSTEM
  ibrav = 0, nat = 5, ntyp = 3
  ecutwfc = 60.0, ecutrho = 480.0
  occupations = 'smearing', smearing = 'mp', degauss = 0.02
  nosym = .true.
/
&ELECTRONS
  conv_thr = 1.0d-10, mixing_beta = 0.5
/
CELL_PARAMETERS {angstrom}
  3.213000  0.000000  0.000000
 -1.606500  2.781924  0.000000
  0.000000  0.000000  3.941000
ATOMIC_SPECIES
  Mg  24.305  Mg.pbe-n-kjpaw_psl.0.3.0.UPF
  B   10.811  B.pbe-n-kjpaw_psl.1.0.0.UPF
  H    1.008  H.pbe-kjpaw.UPF
ATOMIC_POSITIONS {crystal}
  Mg -0.00000  0.00000 -0.00352
  B   0.33333  0.66667  0.40412
  B   0.66667  0.33333  0.40412
  H   0.33333  0.66667  0.73764
  H   0.66667  0.33333  0.73764
K_POINTS {automatic}
  ${GRID}  0 0 0
ENDINPUT

    mkdir -p tmp_k${TAG}
    mpirun --oversubscribe -np 1 $QE/pw.x -in scf_k${TAG}.in > scf_k${TAG}.out 2>&1

    cat > dos_k${TAG}.in << ENDINPUT
&DOS
  prefix = 'mgb2h2_k${TAG}', outdir = './tmp_k${TAG}/'
  fildos = 'dos_k${TAG}.dat', DeltaE = 0.05, degauss = 0.02
/
ENDINPUT
    mpirun --oversubscribe -np 1 $QE/dos.x -in dos_k${TAG}.in > dos_k${TAG}.out 2>&1

    EF=$(grep "the Fermi energy" scf_k${TAG}.out | awk '{print $5}')
    echo "  E_F = $EF eV"
done

# Compare all
python3 << 'PYEOF'
import numpy as np, re, glob

print("\nK-grid Convergence Summary:")
print(f"  {'Grid':>12} {'E_F (eV)':>10} {'N(E_F)':>10}")
print(f"  {'-'*12} {'-'*10} {'-'*10}")

for tag in ['12x12x8', '16x16x12', '20x20x14', '24x24x16']:
    try:
        ef = float(re.search(r'Fermi energy is\s+([\d.]+)',
            open(f'scf_k{tag}.out').read()).group(1))
        d = np.loadtxt(f'dos_k{tag}.dat')
        idx = np.argmin(np.abs(d[:,0] - ef))
        print(f"  {tag:>12} {ef:>10.4f} {d[idx,1]:>10.3f}")
    except:
        print(f"  {tag:>12} {'FAILED':>10}")

print("\n  If N(E_F) varies <10% across grids → CONVERGED")
PYEOF
