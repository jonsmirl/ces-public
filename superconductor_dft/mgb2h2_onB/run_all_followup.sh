#!/bin/bash
# Run all follow-up calculations sequentially
# Chained to run after k-grid convergence finishes
#
# Order: quick checks first, long calculations last
# Total: ~30 min quick + ~8-10 hours long

echo "=== $(date) === CARBON DOPING SCREEN (quick, ~10 min) ==="
cd /home/jonsmirl/thesis/superconductor_dft/mgb2h2_onB
bash run_carbon_doping.sh 2>&1

echo ""
echo "=== $(date) === ZPE STABILITY CHECK (quick, ~5 min) ==="
bash run_zpe_stability.sh 2>&1

echo ""
echo "=== $(date) === SSCHA TARGETED CHECK (~30 min) ==="
source /home/jonsmirl/thesis/.venv/bin/activate
pip install python-sscha cellconstructor 2>&1 | tail -3
python run_sscha_targeted.py 2>&1

echo ""
echo "=== $(date) === CARBON-DOPED EL-PH (LONG, ~8-10 hours) ==="
bash run_carbon_elph.sh 2>&1

echo ""
echo "=== $(date) === ALL CALCULATIONS COMPLETE ==="
