#!/bin/bash
# Carbon-doped MgB2H2: Mg(B1-xCx)2H2
# C is lighter than B (12 vs 11 — negligible), but C-C bonds are STIFFER
# → higher phonon frequencies → higher omega_log → higher Tc
# Also C has one more electron → shifts band filling
#
# Quick SCF + DOS screening for x = 0.25, 0.50
# Uses 2x1x1 supercell (4 B sites → replace 1 or 2 with C)
#
# Run on GPU after el-ph finishes

export NVHPC=$HOME/nvhpc
export NVHPC_ROOT=$NVHPC/Linux_x86_64/24.11
export PATH=$NVHPC_ROOT/comm_libs/12.6/openmpi4/openmpi-4.1.5/bin:$NVHPC_ROOT/compilers/bin:$PATH
export LD_LIBRARY_PATH=$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/cuda/12.6/lib64:$NVHPC_ROOT/comm_libs/12.6/openmpi4/openmpi-4.1.5/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
QE=/tmp/q-e-qe-7.4.1/bin
PSEUDO=/home/jonsmirl/thesis/superconductor_dft/pseudo

# Make sure C pseudo exists
if [ ! -f $PSEUDO/C.pbe-n-kjpaw_psl.1.0.0.UPF ]; then
    echo "Need C pseudopotential. Copying..."
    find /usr/share/espresso/pseudo -name "C.pbe*kjpaw*" -exec cp {} $PSEUDO/ \; 2>/dev/null
    # If not found in system, try to get from a completed calc
    find /home/jonsmirl/thesis/superconductor_dft -name "C.pbe*UPF" -exec cp {} $PSEUDO/ \; 2>/dev/null
    ls $PSEUDO/C.* 2>/dev/null || echo "WARNING: No C pseudo found. May need to download."
fi

cd /home/jonsmirl/thesis/superconductor_dft

echo "============================================================"
echo "CARBON DOPING SCREEN — Mg(B1-xCx)2H2"
echo "============================================================"

# Reference: pure MgB2H2 (already computed)
echo ""
echo "=== x=0.00 (pure MgB2H2) — reference ==="
echo "  E_F = 7.189 eV, N(E_F) = 0.700, lambda_optic = 1.70"

# x=0.25: replace 1 of 4 B with C in 2x1x1 supercell
echo ""
echo "=== x=0.25: Mg2(B3C1)H4 ==="
mkdir -p mgbc_x25 && cd mgbc_x25

cat > scf.in << 'EOF'
&CONTROL
  calculation = 'scf'
  prefix = 'mgbc25'
  pseudo_dir = '/home/jonsmirl/thesis/superconductor_dft/pseudo/'
  outdir = './tmp/'
/
&SYSTEM
  ibrav = 0, nat = 10, ntyp = 4
  ecutwfc = 60.0, ecutrho = 480.0
  occupations = 'smearing', smearing = 'mp', degauss = 0.02
  nosym = .true.
/
&ELECTRONS
  conv_thr = 1.0d-10, mixing_beta = 0.3
/
CELL_PARAMETERS {angstrom}
  6.426  0.000  0.000
 -3.213  5.564  0.000
  0.000  0.000  3.941
ATOMIC_SPECIES
  Mg  24.305  Mg.pbe-n-kjpaw_psl.0.3.0.UPF
  B   10.811  B.pbe-n-kjpaw_psl.1.0.0.UPF
  C   12.011  C.pbe-n-kjpaw_psl.1.0.0.UPF
  H    1.008  H.pbe-kjpaw.UPF
ATOMIC_POSITIONS {crystal}
  Mg  0.000  0.000 -0.004
  Mg  0.500  0.500 -0.004
  B   0.167  0.333  0.404
  B   0.333  0.167  0.404
  B   0.667  0.833  0.404
  C   0.833  0.667  0.404
  H   0.167  0.333  0.738
  H   0.333  0.167  0.738
  H   0.667  0.833  0.738
  H   0.833  0.667  0.738
K_POINTS {automatic}
  6 6 8  0 0 0
EOF

rm -rf tmp && mkdir tmp
mpirun --oversubscribe -np 1 $QE/pw.x -in scf.in > scf.out 2>&1
echo "  EXIT: $?"
grep "the Fermi energy" scf.out
grep "JOB DONE" scf.out

# DOS
cat > dos.in << 'EOF'
&DOS
  prefix = 'mgbc25', outdir = './tmp/'
  fildos = 'dos.dat', DeltaE = 0.05, degauss = 0.02
/
EOF
mpirun --oversubscribe -np 1 $QE/dos.x -in dos.in > dos.out 2>&1

python3 -c "
import numpy as np, re
ef = float(re.search(r'Fermi energy is\s+([\d.]+)', open('scf.out').read()).group(1))
d = np.loadtxt('dos.dat')
idx = np.argmin(np.abs(d[:,0]-ef))
print(f'  N(E_F) = {d[idx,1]:.3f} st/eV/cell ({d[idx,1]/10:.4f}/atom)')
" 2>/dev/null

# Spin check
cat > scf_spin.in << 'EOF'
&CONTROL
  calculation = 'scf'
  prefix = 'mgbc25_sp'
  pseudo_dir = '/home/jonsmirl/thesis/superconductor_dft/pseudo/'
  outdir = './tmp/'
/
&SYSTEM
  ibrav = 0, nat = 10, ntyp = 4
  ecutwfc = 60.0, ecutrho = 480.0
  occupations = 'smearing', smearing = 'mp', degauss = 0.02
  nosym = .true., nspin = 2
/
&ELECTRONS
  conv_thr = 1.0d-8, mixing_beta = 0.3
/
CELL_PARAMETERS {angstrom}
  6.426  0.000  0.000
 -3.213  5.564  0.000
  0.000  0.000  3.941
ATOMIC_SPECIES
  Mg  24.305  Mg.pbe-n-kjpaw_psl.0.3.0.UPF
  B   10.811  B.pbe-n-kjpaw_psl.1.0.0.UPF
  C   12.011  C.pbe-n-kjpaw_psl.1.0.0.UPF
  H    1.008  H.pbe-kjpaw.UPF
ATOMIC_POSITIONS {crystal}
  Mg  0.000  0.000 -0.004
  Mg  0.500  0.500 -0.004
  B   0.167  0.333  0.404
  B   0.333  0.167  0.404
  B   0.667  0.833  0.404
  C   0.833  0.667  0.404
  H   0.167  0.333  0.738
  H   0.333  0.167  0.738
  H   0.667  0.833  0.738
  H   0.833  0.667  0.738
K_POINTS {automatic}
  6 6 8  0 0 0
EOF
rm -rf tmp && mkdir tmp
mpirun --oversubscribe -np 1 $QE/pw.x -in scf_spin.in > scf_spin.out 2>&1
echo "  Stoner check:"
grep "total magnetization" scf_spin.out | tail -1
grep "absolute magnetization" scf_spin.out | tail -1

cd /home/jonsmirl/thesis/superconductor_dft

# x=0.50: replace 2 of 4 B with C (MgBCH2)
echo ""
echo "=== x=0.50: Mg2(B2C2)H4 = MgBCH2 ==="
mkdir -p mgbc_x50 && cd mgbc_x50

cat > scf.in << 'EOF'
&CONTROL
  calculation = 'scf'
  prefix = 'mgbc50'
  pseudo_dir = '/home/jonsmirl/thesis/superconductor_dft/pseudo/'
  outdir = './tmp/'
/
&SYSTEM
  ibrav = 0, nat = 10, ntyp = 4
  ecutwfc = 60.0, ecutrho = 480.0
  occupations = 'smearing', smearing = 'mp', degauss = 0.02
  nosym = .true.
/
&ELECTRONS
  conv_thr = 1.0d-10, mixing_beta = 0.3
/
CELL_PARAMETERS {angstrom}
  6.426  0.000  0.000
 -3.213  5.564  0.000
  0.000  0.000  3.941
ATOMIC_SPECIES
  Mg  24.305  Mg.pbe-n-kjpaw_psl.0.3.0.UPF
  B   10.811  B.pbe-n-kjpaw_psl.1.0.0.UPF
  C   12.011  C.pbe-n-kjpaw_psl.1.0.0.UPF
  H    1.008  H.pbe-kjpaw.UPF
ATOMIC_POSITIONS {crystal}
  Mg  0.000  0.000 -0.004
  Mg  0.500  0.500 -0.004
  B   0.167  0.333  0.404
  C   0.333  0.167  0.404
  B   0.667  0.833  0.404
  C   0.833  0.667  0.404
  H   0.167  0.333  0.738
  H   0.333  0.167  0.738
  H   0.667  0.833  0.738
  H   0.833  0.667  0.738
K_POINTS {automatic}
  6 6 8  0 0 0
EOF

rm -rf tmp && mkdir tmp
mpirun --oversubscribe -np 1 $QE/pw.x -in scf.in > scf.out 2>&1
echo "  EXIT: $?"
grep "the Fermi energy" scf.out
grep "JOB DONE" scf.out

# DOS
cat > dos.in << 'EOF'
&DOS
  prefix = 'mgbc50', outdir = './tmp/'
  fildos = 'dos.dat', DeltaE = 0.05, degauss = 0.02
/
EOF
mpirun --oversubscribe -np 1 $QE/dos.x -in dos.in > dos.out 2>&1

python3 -c "
import numpy as np, re
ef = float(re.search(r'Fermi energy is\s+([\d.]+)', open('scf.out').read()).group(1))
d = np.loadtxt('dos.dat')
idx = np.argmin(np.abs(d[:,0]-ef))
print(f'  N(E_F) = {d[idx,1]:.3f} st/eV/cell ({d[idx,1]/10:.4f}/atom)')
" 2>/dev/null

# Spin check
sed 's/mgbc50/mgbc50_sp/;s/nosym = .true./nosym = .true., nspin = 2/' scf.in > scf_spin.in
rm -rf tmp && mkdir tmp
mpirun --oversubscribe -np 1 $QE/pw.x -in scf_spin.in > scf_spin.out 2>&1
echo "  Stoner check:"
grep "total magnetization" scf_spin.out | tail -1
grep "absolute magnetization" scf_spin.out | tail -1

cd /home/jonsmirl/thesis/superconductor_dft

echo ""
echo "============================================================"
echo "CARBON DOPING SUMMARY"
echo "============================================================"
echo "  x=0.00 (MgB2H2):   E_F=7.189, N(E_F)=0.700, non-magnetic"
for x in 25 50; do
    dir="mgbc_x${x}"
    ef=$(grep "Fermi energy" $dir/scf.out 2>/dev/null | awk '{print $5}')
    mag=$(grep "total magnetization" $dir/scf_spin.out 2>/dev/null | tail -1 | awk '{print $4}')
    nef=$(python3 -c "
import numpy as np, re
ef=float(re.search(r'Fermi energy is\s+([\d.]+)',open('$dir/scf.out').read()).group(1))
d=np.loadtxt('$dir/dos.dat')
idx=np.argmin(np.abs(d[:,0]-ef))
print(f'{d[idx,1]:.3f}')
" 2>/dev/null)
    echo "  x=0.${x}: E_F=$ef, N(E_F)=$nef, mag=$mag"
done
echo ""
echo "  C doping shifts E_F (adds electrons to sigma band)."
echo "  If N(E_F) increases and no magnetism → promising for higher Tc."
echo "  If E_F moves past van Hove singularity → N(E_F) drops."
echo "============================================================"
