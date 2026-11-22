## Environment

Tested on Ubuntu 24.04.4, Python 3.11.15, RTX 4090 D (24 GB), driver 590.48.01, Isaac Sim 5.1.0 and Isaac Lab `090aed18` (2.3.2). Use the declared SDK environment (`scripts/install-sdk.sh` supplies a fresh-install recipe), then:
```bash
conda activate isaaclab
bash scripts/environment.sh
gh auth login
.venv/bin/python -m course.download demo
.venv/bin/python -m pip install --no-deps --no-build-isolation -e resources/sdk/IsaacLab/source/isaaclab
```

## Success rate

Default **A_130_50 DP + matched Transformer: 5/24 (20.83%)** strict autonomous complete insertions on frozen24 (reserved8 + fresh16); [episode results](evidence/frozen24/rollouts.csv). This is the highest observed value among four existing paired comparisons, not a fresh estimate after model selection.

## Train

Real-data workflow check (one DP update → new nominal for three full episodes → two Transformer updates → matched-pair save/reload; **not full retraining**):
```bash
.venv/bin/python -m course.download smoke
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m course.train smoke --root outputs/smoke
```
Complete selected72-parent continuation on the original 130-train/6-validation pool and unchanged family split (distinct from the historical 400-episode archive):
```bash
.venv/bin/python -m course.download training
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m course.train all --root outputs/full-A --role A
```
The full command prepares real data, trains DP, rebuilds nominal using its selected checkpoint, trains its matching Transformer, and writes `outputs/full-A/deploy/`. Use `--role B` for the command-target comparison.

## Run the checkpoint in Isaac Sim

From the repository root in a graphical desktop session:
```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m course.demo
```
This opens Isaac Sim, loads the released pair and native scene, runs the predetermined seed 28853000 at 120/60/5 Hz, and prints `NATIVE EVALUATOR: SUCCESS` or `FAILURE`. A timeout is a valid terminal; this acceptance episode does not change the historical success rate. Results are saved under `outputs/demo`; preserve that directory and use `--output outputs/demo-next` for another execution.
