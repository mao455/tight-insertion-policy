#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
.venv/bin/python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.lock
mkdir -p .deps
git clone https://github.com/isaac-sim/IsaacLab.git .deps/IsaacLab
git -C .deps/IsaacLab checkout 090aed18163b2194d5551c7919f7539283677743
.venv/bin/python -m pip install -e .deps/IsaacLab/source/isaaclab
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m course.check_environment
