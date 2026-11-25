#!/usr/bin/env bash
# Starts DADS on this computer and opens http://localhost:8000
cd "$(dirname "$0")"
[ -x venv/bin/python ] || { echo "Run ./setup_local.sh first."; exit 1; }
exec venv/bin/python main.py
