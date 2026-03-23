#!/bin/bash
# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install numpy matplotlib scipy
pip install torch --index-url https://download.pytorch.org/whl/cu121
echo "Done. Activate with: source venv/bin/activate"
