#!/usr/bin/env bash
# One-time setup on Linux/macOS (Python 3.11+): creates the virtual environment and installs the dependencies.
# For a CPU-only PyTorch (much smaller download on Linux), run first:  export DADS_CPU_TORCH=1
set -e
cd "$(dirname "$0")"
[ -d venv ] || python3 -m venv venv
. venv/bin/activate
python -m pip install --upgrade pip
if [ "${DADS_CPU_TORCH:-0}" = "1" ]; then
    pip install "$(grep '^torch==' requirements.txt)" --index-url https://download.pytorch.org/whl/cpu
fi
# Exact, tested versions: requirements.txt (DADS's packages) + constraints.txt (what they pull in).
pip install -r requirements.txt -c constraints.txt
echo
echo "Setup finished. Start DADS with ./run_local.sh"
