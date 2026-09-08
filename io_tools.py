"""Portable result storage for L2-MMS runs."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np


ARRAY_FIELDS = (
    "times",
    "energies",
    "perimeters",
    "areas",
    "mesh_ratios",
    "min_edges",
    "lambdas",
    "newton_iters",
)


def save_result(path: str | Path, result: dict) -> Path:
    """Save a result dictionary as compressed NPZ without object arrays."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for field in ARRAY_FIELDS:
        if field in result:
            arrays[field] = np.asarray(result[field])

    snapshots = result.get("snapshots", {})
    snapshot_times = np.asarray(sorted(float(t) for t in snapshots), dtype=float)
    if len(snapshot_times):
        snapshot_curves = np.stack([np.asarray(snapshots[t]) for t in snapshot_times])
    else:
        snapshot_curves = np.empty((0, 0, 2), dtype=float)

    metadata = {
        key: value
        for key, value in result.items()
        if key not in ARRAY_FIELDS and key != "snapshots"
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    arrays["snapshot_times"] = snapshot_times
    arrays["snapshot_curves"] = snapshot_curves
    np.savez_compressed(destination, **arrays)
    return destination


def load_result(path: str | Path) -> dict:
    """Load a result written by save_result with allow_pickle disabled."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        result = json.loads(str(archive["metadata_json"].item()))
        for field in ARRAY_FIELDS:
            if field in archive:
                result[field] = archive[field].copy()
        snapshot_times = archive["snapshot_times"].astype(float)
        snapshot_curves = archive["snapshot_curves"]
        result["snapshots"] = {
            float(time): snapshot_curves[index].copy()
            for index, time in enumerate(snapshot_times)
        }
    return result


def convert_legacy_pickle(source: str | Path, destination: str | Path) -> Path:
    """Convert a trusted legacy cache produced by the original scripts."""
    source_path = Path(source)
    with source_path.open("rb") as stream:
        legacy = pickle.load(stream)

    if legacy.get("aniso_type") == "kfold":
        legacy["aniso_type"] = "qfold"
        parameters = dict(legacy.get("aniso_params", {}))
        if "q" not in parameters and "k" in parameters:
            parameters["q"] = parameters.pop("k")
        legacy["aniso_params"] = parameters

    legacy.setdefault(
        "flow",
        "apcsf" if legacy.get("A_target") is not None or "areas" in legacy else "csf",
    )
    return save_result(destination, legacy)
