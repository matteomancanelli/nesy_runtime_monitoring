"""Explicit readers for archived pre-E0 benchmark artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _paths(value: str | Path | list[str | Path]) -> list[Path]:
    if isinstance(value, (str, Path)):
        value = [value]
    return [Path(path) for path in value]


def _config_label(device: str, gpu_name: str | float) -> str:
    if str(device) == "cuda":
        missing = gpu_name is None or isinstance(gpu_name, float)
        name = "" if missing else str(gpu_name)
        return f"GPU ({name})" if name else "GPU"
    return "CPU"


def load_legacy_timing(
    csv_paths: str | Path | list[str | Path],
) -> pd.DataFrame:
    """Load archived July CSVs and migrate their overloaded ``n_leaves`` axis."""
    frames = [pd.read_csv(path) for path in _paths(csv_paths) if path.exists()]
    if not frames:
        raise FileNotFoundError(f"no legacy timing CSVs found among {csv_paths}")
    df = pd.concat(frames, ignore_index=True)
    state_family = df["formula_name"].str.startswith(("boundedresp_", "kthlast_"))
    if "n_leaves" in df:
        legacy_atoms = pd.Series(
            np.where(state_family, 2, df["n_leaves"]), index=df.index
        )
        legacy_states = pd.Series(
            np.where(state_family, df["n_leaves"], np.nan), index=df.index
        )
        df["n_atoms"] = (
            legacy_atoms if "n_atoms" not in df else df["n_atoms"].fillna(legacy_atoms)
        )
        df["dfa_states"] = (
            legacy_states
            if "dfa_states" not in df
            else df["dfa_states"].fillna(legacy_states)
        )
    elif "dfa_states" not in df:
        df["dfa_states"] = np.nan
    if "device" not in df:
        df["device"] = "cpu"
    if "gpu_name" not in df:
        df["gpu_name"] = ""
    df["gpu_name"] = df["gpu_name"].fillna("")
    df["config"] = [
        _config_label(device, gpu) for device, gpu in zip(df["device"], df["gpu_name"])
    ]
    keys = ["monitor_name", "formula_name", "trace_length", "n_traces", "config"]
    aggregations: dict[str, str] = {
        "mean_s_per_cell": "mean",
        "std_s_per_cell": "mean",
        "n_atoms": "first",
        "dfa_states": "first",
    }
    for column in (
        "ast_nodes",
        "distinct_subformulae",
        "ast_depth",
        "temporal_depth",
        "sweep_value",
    ):
        if column in df:
            aggregations[column] = "first"
    df = df.groupby(keys, as_index=False).agg(aggregations)
    df["mean_s_per_trace"] = df["mean_s_per_cell"] * df["trace_length"]
    df["std_s_per_trace"] = df["std_s_per_cell"] * df["trace_length"]
    df["us_per_cell"] = df["mean_s_per_cell"] * 1e6
    df["us_err"] = df["std_s_per_cell"] * 1e6
    return df
