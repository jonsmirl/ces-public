#!/bin/bash
# Full el-ph calculation for best carbon-doped candidate
# Runs AFTER run_carbon_doping.sh identifies which x is best
#
# Picks the x with highest N(E_F) and no magnetism, then runs
# phonon + el-ph on that structure (same pipeline as MgB2H2)
#
# Estimated time: ~8-10 hours on GPU (larger supercell = more modes)

export NVHPC=$HOME/nvhpc
export NVHPC_ROOT=$NVHPC/Linux_x86_64/24.11
export PATH=$NVHPC_ROOT/comm_libs/12.6/openmpi4/openmpi-4.1.5/bin:$NVHPC_ROOT/compilers/bin:$PATH
export LD_LIBRARY_PATH=$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/cuda/12.6/lib64:$NVHPC_ROOT/comm_libs/12.6/openmpi4/openmpi-4.1.5/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
QE=/tmp/q-e-qe-7.4.1/bin
PSEUDO=/home/jonsmirl/thesis/superconductor_dft/pseudo
cd /home/jonsmirl/thesis/superconductor_dft

echo "============================================================"
echo "CARBON-DOPED MgB2H2 — Full el-ph calculation"
echo "============================================================"

# Pick best candidate from screening
echo "Checking screening results..."
BEST=""
BEST_NEF=0
for x in 25 50; do
    dir="mgbc_x${x}"
    if [ ! -f "$dir/scf.out" ]; then
        echo "  x=0.$x: no SCF results — skipping"
        continue
    fi

    # Check magnetism
    mag=$(grep "total magnetization" $dir/scf_spin.out 2>/dev/null | tail -1 | awk '{printf "%.2f", $4}')
    if [ -z "$mag" ]; then mag="0.00"; fi
    mag_abs=$(echo "$mag" | tr -d '-')

    if (( $(echo "$mag_abs > 0.1" | bc -l 2>/dev/null || echo 0) )); then
        echo "  x=0.$x: MAGNETIC (mag=$mag) — skipping"
        continue
    fi

    # Get N(E_F)
    nef=$(python3 -c "
import numpy as np, re
ef=float(re.search(r'Fermi energy is\s+([\d.]+)',open('$dir/scf.out').read()).group(1))
d=np.loadtxt('$dir/dos.dat')
idx=np.argmin(np.abs(d[:,0]-ef))
print(f'{d[idx,1]:.3f}')
" 2>/dev/null)

    echo "  x=0.$x: N(E_F)=$nef, mag=$mag — non-magnetic"

    if [ -n "$nef" ] && (( $(echo "$nef > $BEST_NEF" | bc -l 2>/dev/null || echo 0) )); then
        BEST="$x"
        BEST_NEF="$nef"
    fi
done

if [ -z "$BEST" ]; then
    echo "No suitable candidate found. Exiting."
    exit 1
fi

echo ""
echo "BEST CANDIDATE: x=0.$BEST (N(E_F)=$BEST_NEF)"
echo ""

WORKDIR="mgbc_x${BEST}"
cd $WORKDIR

# Step 1: Dense SCF for el-ph
echo "--- Step 1: Dense k-grid SCF ---"
cat > scf_dense.in << EOF
&CONTROL
  calculation = 'scf'
  prefix = 'mgbc${BEST}'
  pseudo_dir = '${PSEUDO}/'
  outdir = './tmp/'
  la2F = .true.
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
$(grep -A5 "CELL_PARAMETERS" scf.in)
$(grep -A6 "ATOMIC_SPECIES" scf.in)
$(grep -A12 "ATOMIC_POSITIONS" scf.in)
K_POINTS {automatic}
  12 12 8  0 0 0
EOF

rm -rf tmp && mkdir tmp
echo "Running dense SCF on GPU..."
mpirun --oversubscribe -np 1 $QE/pw.x -in scf_dense.in > scf_dense.out 2>&1
echo "SCF EXIT: $?"
grep "the Fermi energy" scf_dense.out

# Step 2: Phonon + el-ph
echo ""
echo "--- Step 2: Phonon + el-ph (2x2x2 q-grid) ---"
cat > elph.in << EOF
El-ph for Mg(B1-xCx)2H2 x=0.${BEST}
&INPUTPH
  prefix = 'mgbc${BEST}'
  outdir = './tmp/'
  fildyn = 'mgbc.dyn'
  fildvscf = 'dvscf'
  ldisp = .true.
  nq1 = 2, nq2 = 2, nq3 = 2
  tr2_ph = 1.0d-12
  search_sym = .false.
  electron_phonon = 'interpolated'
  el_ph_sigma = 0.01
  el_ph_nsigma = 10
/
EOF

echo "Running el-ph on GPU (this will take ~8-10 hours)..."
echo "Started at $(date)"
mpirun --oversubscribe -np 1 $QE/ph.x -in elph.in > elph.out 2>&1
echo "El-ph EXIT: $?"
echo "Finished at $(date)"

# Step 3: Extract results
echo ""
echo "--- Step 3: Results ---"
python3 << 'PYEOF'
import re
blocks, current = [], []
with open("elph.out") as fh:
    for line in fh:
        m = re.match(r'\s+lambda\(\s*(\d+)\)=\s*([-\d.]+)\s+gamma=\s*([\d.]+)', line)
        if m: current.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
        elif current and 'lambda' not in line:
            if len(current) > 0: blocks.append(current)
            current = []

modes_per_q = 30  # 10 atoms * 3
qpts = 0
for b in blocks:
    if len(b) == modes_per_q:
        qpts += 1

# Use whatever mode count we find
if blocks:
    nmodes = len(blocks[0])
    sets = len(blocks) // 10  # 10 smearing widths per q-point
    print(f"Modes per q-point: {nmodes}")
    print(f"Q-points with data: {sets}")

    optic_all = []
    for q in range(sets):
        b = blocks[q * 10]
        # acoustic = first 3, optic = rest
        ac = sum(l for mode, l, _ in b if mode <= 3 and l > 0)
        op = sum(l for mode, l, _ in b if mode > 3 and l > 0)
        tot = ac + op
        note = " SOFT" if ac > 2 else ""
        optic_all.append(op)
        print(f"  Q{q+1}: total={tot:6.2f}  ac={ac:5.2f}  op={op:5.2f}{note}")

    if optic_all:
        oavg = sum(optic_all)/len(optic_all)
        print(f"\n  Optic lambda avg: {oavg:.2f}")
        print(f"  T_c (strong-coupling, w_log~800K): {0.18*800*oavg**0.5:.0f} K")
        print(f"  Compare MgB2H2: optic lambda = 1.70, T_c = 93-185K")
else:
    print("No lambda data found — check elph.out for errors")
PYEOF

echo ""
echo "============================================================"
echo "CARBON-DOPED EL-PH COMPLETE"
echo "============================================================"
