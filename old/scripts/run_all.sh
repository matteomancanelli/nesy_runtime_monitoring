#!/usr/bin/env bash
# Run every timing experiment in sequence.
#
# Safe to re-run: each experiment resumes from results/*.csv, skipping cells
# already computed. So if a run is killed, just invoke this again.
set -euo pipefail

python experiments/exp1_single_trace.py
python experiments/exp2_formula_complexity.py
python experiments/exp3_batch_size.py
python experiments/exp5_depth_microbench.py
python experiments/exp6_state_scaling.py
python experiments/exp7_state_blowup.py
