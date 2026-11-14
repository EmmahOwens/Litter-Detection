#!/usr/bin/env bash
# Repeatable One-Command Launcher for Capstone Deliverable 3 PC Deployment
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

echo "======================================================================"
echo "  Urban Litter Instance Segmentation & Municipal Severity System (D3) "
echo "  Target Platform: User-End PC / Laptop Web Interface                 "
echo "  Python Runtime : $PYTHON_BIN                                        "
echo "======================================================================"

exec "$PYTHON_BIN" proposed/app_d3_deployed.py
