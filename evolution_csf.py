"""
Evolution experiments for anisotropic curve shortening flow (CSF).

This script implements the semi-implicit L2-MMS PFEM used in the paper.
It covers isotropic, weakly anisotropic, and strongly anisotropic cases.
For strong anisotropy, gamma is split as

    gamma = gamma_c - gamma_e,
    gamma_c = gamma + C |.|,    gamma_e = C |.|.

Outputs for each selected case:
  * result.npz       complete numerical history and curve snapshots
  * evolution.png    curve snapshots with mesh nodes
  * diagnostics.png  normalized energy, mesh ratio, Newton iterations

Edit only the Parameters block below for normal use.
"""

from pathlib import Path

from io_tools import load_result, save_result
from mms import solve_csf
from plotting import plot_diagnostics, plot_evolution


# =========================================================
# Parameters
# =========================================================
REUSE_DATA = True

CASES_TO_RUN = [
    "qfold_q4_weak",
    "riemannian",
    "l4",
    "qfold_q4_strong",
]

CASES = {
    "isotropic": {
        "flow": "csf",
        "aniso_type": "isotropic",
        "aniso_params": {},
        "initial_shape": "circle",
        "N": 128,
        "dt": 5.0e-4,
        "T_final": 0.48,
        "snapshot_times": [0.0, 0.15, 0.30, 0.47],
        "axis_range": 3.0,
    },
    "qfold_q4_weak": {
        "flow": "csf",
        "aniso_type": "qfold",
        "aniso_params": {"q": 4, "beta": 0.03},
        "initial_shape": "circle",
        "N": 128,
        "dt": 5.0e-4,
        "T_final": 0.50,
        "snapshot_times": [0.0, 0.20, 0.30, 0.50],
        "axis_range": 3.0,
    },
    "riemannian": {
        "flow": "csf",
        "aniso_type": "riemannian",
        "aniso_params": {"G_list": [[1.0, 0.0, 2.0]]},
        "initial_shape": "circle",
        "N": 128,
        "dt": 5.0e-4,
        "T_final": 0.41,
        "snapshot_times": [0.0, 0.15, 0.30, 0.41],
        "axis_range": 3.0,
    },
    "l4": {
        "flow": "csf",
        "aniso_type": "l4",
        "aniso_params": {},
        "initial_shape": "circle",
        "N": 128,
        "dt": 5.0e-4,
        "T_final": 0.52,
        "snapshot_times": [0.0, 0.10, 0.40, 0.52],
        "axis_range": 3.0,
    },
    "qfold_q4_strong": {
        "flow": "csf",
        "aniso_type": "qfold",
        "aniso_params": {"q": 4, "beta": 0.10},
        "initial_shape": "circle",
        "N": 128,
        "dt": 5.0e-4,
        "T_final": 0.50,
        "snapshot_times": [0.0, 0.10, 0.20, 0.50],
        "axis_range": 3.0,
    },
    "qfold_q2_strong_flower": {
        "flow": "csf",
        "aniso_type": "qfold",
        "aniso_params": {"q": 2, "beta": 0.40},
        "initial_shape": "flower",
        "N": 128,
        "dt": 5.0e-4,
        "T_final": 0.50,
        "snapshot_times": [0.0, 0.10, 0.30, 0.50],
        "axis_range": 3.0,
    },
}

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output"


# =========================================================
# Run selected cases
# =========================================================
for case_name in CASES_TO_RUN:
    if case_name not in CASES:
        raise ValueError(f"Unknown CSF case: {case_name}")

    case_output = OUTPUT_ROOT / f"csf_{case_name}"
    case_output.mkdir(parents=True, exist_ok=True)
    data_path = case_output / "result.npz"

    if REUSE_DATA and data_path.exists():
        result = load_result(data_path)
        print(f"Loaded {data_path}")
    else:
        result = solve_csf(case_name, CASES[case_name])
        save_result(data_path, result)
        print(f"Saved {data_path}")

    evolution_path = plot_evolution(result, case_output / "evolution.png")
    diagnostic_path = plot_diagnostics(result, case_output / "diagnostics.png")
    print(f"Saved {evolution_path}")
    print(f"Saved {diagnostic_path}")
