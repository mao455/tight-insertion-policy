#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Reuse an explicitly activated Isaac Sim 5.1 Python installation.
# The new venv owns every course install and all downloaded project resources.
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python -m course.check_environment
