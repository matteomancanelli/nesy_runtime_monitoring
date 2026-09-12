"""Regenerate every figure from the two Colab sessions in ``results/``.

``results/cpu/`` and ``results/gpu/`` hold one CSV per experiment, produced by a
CPU Colab runtime and a Tesla-T4 runtime with *identical* sweep configs. They
stay separate on disk: ``plots.load_timing`` takes a list of paths, concatenates
them, and splits curves by the truthful per-monitor ``device`` stamp — a CPU-only
monitor (Symbolic, and the original structured RuleRunner when built on CPU)
reports ``cpu`` even inside the GPU CSV, and duplicate (monitor, x, config) rows
are averaged rather than double-drawn. Merging the files by hand would destroy
exactly the information the ``config`` split relies on.

Three figure sets are written:

* ``figures/``          merged CPU+GPU (solid = CPU, dashed = GPU)
* ``figures/gpu_only/`` GPU session alone — the clean lead figures for the paper
* ``figures/device/``   per-monitor CPU-vs-GPU + the GPU-speedup summary

Exp 6's CPU sweep is partial (only Symbolic, RuleRunner, and 4 of 9 structured
points; DeepDFA and the progression monitors never ran), so it is excluded from
the merged and device sets and reported GPU-only.

Run:  python experiments/make_all_plots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments import plots

RESULTS = ROOT / "results"
CPU, GPU = RESULTS / "cpu", RESULTS / "gpu"
FIGS = RESULTS / "figures"

# Timing experiments: (name, csv basename, plot fn, merged?, device-compare?)
TIMING = [
    ("exp1", "exp1_single_trace.csv", plots.plot_exp1, True, True),
    ("exp2", "exp2_formula_complexity.csv", plots.plot_exp2, True, True),
    ("exp3", "exp3_batch_size.csv", plots.plot_exp3, True, True),
    ("exp5", "exp5_depth_microbench.csv", plots.plot_exp5, True, True),
    # exp6's CPU sweep never finished (no DeepDFA / progression rows) — a merged
    # figure would silently show a 3-monitor CPU set against an 8-monitor GPU set.
    ("exp6", "exp6_state_scaling.csv", plots.plot_exp6, False, False),
    ("exp7_stateblowup", "exp7_stateblowup.csv", plots.plot_exp7_stateblowup, True, False),
]


def _both(name: str) -> list[Path]:
    return [CPU / name, GPU / name]


def main() -> None:
    merged, gpu_only, device = FIGS, FIGS / "gpu_only", FIGS / "device"
    for d in (merged, gpu_only, device):
        d.mkdir(parents=True, exist_ok=True)

    for tag, csv, fn, do_merge, do_device in TIMING:
        if do_merge:
            fn(_both(csv), merged)
        fn([GPU / csv], gpu_only)
        if do_device:
            plots.plot_device_comparison(_both(csv), tag, device)
            plots.plot_device_speedup(_both(csv), tag, device)

    # Cost of correctness: exp2 (flat IJCNN family, where the ORIGINAL RuleRunner
    # is also correct, so the ratio isolates throughput, not the verdict fix).
    plots.plot_correctness_cost(_both("exp2_formula_complexity.csv"), merged)
    plots.plot_correctness_cost([GPU / "exp2_formula_complexity.csv"], gpu_only)


if __name__ == "__main__":
    main()
