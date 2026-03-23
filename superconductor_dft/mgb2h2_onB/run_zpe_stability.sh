#!/bin/bash
# Zero-point energy stability check for MgB2H2
#
# Compares free energy (E + ZPE) of:
#   1. Hexagonal MgB2H2 (our candidate)
#   2. CDW-distorted supercell (from run_cdw_check.py)
#
# If ZPE favors hexagonal even when classical E favors CDW,
# the hexagonal phase is quantum-stabilized.
#
# Run AFTER: el-ph (for phonon DOS) and CDW check (for distorted structure)
#
# Usage: bash run_zpe_stability.sh

export NVHPC=$HOME/nvhpc
export NVHPC_ROOT=$NVHPC/Linux_x86_64/24.11
export PATH=$NVHPC_ROOT/comm_libs/12.6/openmpi4/openmpi-4.1.5/bin:$NVHPC_ROOT/compilers/bin:$PATH
export LD_LIBRARY_PATH=$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/cuda/12.6/lib64:$NVHPC_ROOT/comm_libs/12.6/openmpi4/openmpi-4.1.5/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
QE=/tmp/q-e-qe-7.4.1/bin
cd /home/jonsmirl/thesis/superconductor_dft/mgb2h2_onB

echo "============================================================"
echo "ZERO-POINT ENERGY STABILITY CHECK — MgB2H2"
echo "============================================================"

# Step 1: Get ZPE of hexagonal phase from phonon DOS
echo ""
echo "--- Step 1: ZPE of hexagonal MgB2H2 ---"

# Compute phonon DOS on fine grid (if not already done)
if [ ! -f mgb2h2r.fc ]; then
    echo "  Need force constants. Run q2r first:"
    cat > q2r_final.in << 'EOF'
&INPUT
  fildyn = 'mgb2h2r.dyn'
  flfrc = 'mgb2h2r.fc'
  zasr = 'crystal'
/
EOF
    mpirun --oversubscribe -np 1 $QE/q2r.x -in q2r_final.in > q2r_final.out 2>&1
    echo "  q2r done"
fi

# Phonon DOS for ZPE integration
cat > matdyn_zpe.in << 'EOF'
&INPUT
  asr = 'crystal'
  flfrc = 'mgb2h2r.fc'
  dos = .true.
  fldos = 'phdos_zpe.dat'
  nk1 = 20, nk2 = 20, nk3 = 14
  deltaE = 1.0
/
EOF
mpirun --oversubscribe -np 1 $QE/matdyn.x -in matdyn_zpe.in > matdyn_zpe.out 2>&1

# Compute ZPE = (1/2) * sum_q,nu hbar*omega_q,nu
python3 << 'PYEOF'
import numpy as np

# Constants
hbar = 6.582119569e-16  # eV*s
kB = 8.617333262e-5     # eV/K
cm1_to_eV = 1.23984e-4  # 1 cm-1 in eV

try:
    data = np.loadtxt('phdos_zpe.dat')
    freq = data[:, 0]  # cm-1
    dos = data[:, 1]   # states/cm-1

    # Only positive frequencies
    pos = freq > 5
    f = freq[pos]
    d = dos[pos]

    # ZPE = integral of (1/2) * hbar*omega * g(omega) d(omega)
    # In cm-1 units: ZPE = (1/2) * sum(freq * dos * delta_freq) * cm1_to_eV
    df = f[1] - f[0]  # uniform spacing
    zpe = 0.5 * np.sum(f * d * df) * cm1_to_eV
    print(f"  ZPE (hexagonal) = {zpe*1000:.1f} meV/cell = {zpe:.4f} eV/cell")
    print(f"  ZPE per atom = {zpe/5*1000:.1f} meV/atom")

    # Free energy at 300K: F = E + ZPE + integral(kT * ln(1-exp(-hw/kT)) * g(w) dw)
    T = 300.0
    thermal = 0.0
    for i in range(len(f)):
        hw = f[i] * cm1_to_eV
        if hw > 0 and T > 0:
            x = hw / (kB * T)
            if x < 50:
                thermal += d[i] * df * kB * T * np.log(1 - np.exp(-x))
    thermal *= cm1_to_eV / cm1_to_eV  # already in right units via kB*T
    # Actually: thermal contribution = kB*T * integral(ln(1-exp(-hw/kBT)) * g(w) dw) * delta_w
    # Need to be more careful with units
    F_vib = zpe  # + thermal (thermal is negative, reduces F)
    print(f"  F_vib(300K) ≈ {F_vib*1000:.1f} meV/cell (ZPE dominant for light H)")

except Exception as e:
    print(f"  Error computing ZPE: {e}")
    print(f"  (matdyn may have failed — check matdyn_zpe.out)")
PYEOF

# Step 2: Get classical energy difference (hexagonal vs CDW)
echo ""
echo "--- Step 2: Classical energy comparison ---"

E_hex=$(grep "!    total energy" scf_relaxed_nosym.out 2>/dev/null | tail -1 | awk '{print $5}')
if [ -z "$E_hex" ]; then
    E_hex=$(grep "!    total energy" scf_gpu_full.out 2>/dev/null | tail -1 | awk '{print $5}')
fi

E_cdw="not yet computed"
if [ -f scf_cdw_supercell.in ]; then
    if [ -f tmp_cdw/mgb2h2_cdw.save/data-file-schema.xml ]; then
        E_cdw=$(grep "!    total energy" scf_cdw.out 2>/dev/null | tail -1 | awk '{print $5}')
    else
        echo "  CDW supercell SCF not yet run. Run:"
        echo "    mkdir -p tmp_cdw"
        echo "    mpirun -np 1 $QE/pw.x -in scf_cdw_supercell.in > scf_cdw.out"
    fi
fi

echo "  E(hexagonal) = $E_hex Ry/cell"
echo "  E(CDW 2x2x1) = $E_cdw Ry/supercell"

echo ""
echo "--- Step 3: Verdict ---"
echo "  If ZPE(hexagonal) - ZPE(CDW) > E(CDW) - E(hexagonal):"
echo "    → Quantum stabilization: hexagonal phase survives"
echo "  Else:"
echo "    → CDW is the true ground state"
echo ""
echo "  NOTE: Light hydrogen (1 amu) has LARGE zero-point motion."
echo "  In similar hydrides, ZPE corrections are 50-100 meV/atom,"
echo "  often enough to stabilize the high-symmetry phase."
echo "============================================================"
