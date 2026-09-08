"""
Evolution experiments for area-preserving anisotropic CSF (AP-CSF).

The constrained L2-MMS step is solved for the curve and one real-valued
Lagrange multiplier. The multiplier enforces the signed enclosed area exactly
up to the nonlinear solver tolerance.

Outputs for each selected case:
  * result.npz       complete history and curve snapshots
  * evolution.png    curve snapshots with mesh nodes
  * diagnostics.png  energy, area error, mesh ratio, Newton iterations

Edit only the Parameters block below for normal use.
"""

from pathlib import Path

from io_tools import load_result, save_result
from mms import solve_apcsf
from plotting import plot_diagnostics, plot_evolution


# =========================================================
# Parameters
# =========================================================
REUSE_DATA = True

CASES_TO_RUN = [
    "isotropic_ellipse",
    "qfold_q2_strong_ellipse",
]

CASES = {
    "isotropic_ellipse": {
        "flow": "apcsf",
        "aniso_type": "isotropic",
        "aniso_params": {},
        "initial_shape": "ellipse",
        "N": 128,
        "dt": 1.0e-4,
        "T_final": 2.50,
        "snapshot_times": [0.0, 0.50, 1.00, 2.50],
        "axis_range": 5.0,
    },
    "qfold_q4_weak_ellipse": {
        "flow": "apcsf",
        "aniso_type": "qfold",
        "aniso_params": {"q": 4, "beta": 0.03},
        "initial_shape": "ellipse",
        "N": 128,
        "dt": 5.0e-4,
        "T_final": 2.50,
        "snapshot_times": [0.0, 0.50, 1.00, 2.00],
        "axis_range": 5.0,
    },
    "riemannian_ellipse": {
        "flow": "apcsf",
        "aniso_type": "riemannian",
        "aniso_params": {"G_list": [[1.0, 0.0, 2.0]]},
        "initial_shape": "ellipse",
        "N": 128,
        "dt": 5.0e-4,
        "T_final": 2.50,
        "snapshot_times": [0.0, 0.50, 1.00, 2.00],
        "axis_range": 5.0,
    },
    "l4_ellipse": {
        "flow": "apcsf",
        "aniso_type": "l4",
        "aniso_params": {},
        "initial_shape": "ellipse",
        "N": 128,
        "dt": 5.0e-4,
        "T_final": 2.50,
        "snapshot_times": [0.0, 0.50, 1.00, 2.00],
        "axis_range": 5.0,
    },
    "qfold_q2_strong_ellipse": {
        "flow": "apcsf",
        "aniso_type": "qfold",
        "aniso_params": {"q": 2, "beta": 0.40},
        "initial_shape": "ellipse",
        "N": 128,
        "dt": 5.0e-4,
        "T_final": 3.50,
        "snapshot_times": [0.0, 0.10, 0.30, 0.50, 1.00, 1.50, 2.00, 3.50],
        "axis_range": 5.0,
    },
}

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output"


# =========================================================
# Run selected cases
# =========================================================
for case_name in CASES_TO_RUN:
    if case_name not in CASES:
        raise ValueError(f"Unknown AP-CSF case: {case_name}")

    case_output = OUTPUT_ROOT / f"apcsf_{case_name}"
    case_output.mkdir(parents=True, exist_ok=True)
    data_path = case_output / "result.npz"

    if REUSE_DATA and data_path.exists():
        result = load_result(data_path)
        print(f"Loaded {data_path}")
    else:
        result = solve_apcsf(case_name, CASES[case_name])
        save_result(data_path, result)
        print(f"Saved {data_path}")

    evolution_path = plot_evolution(result, case_output / "evolution.png")
    diagnostic_path = plot_diagnostics(result, case_output / "diagnostics.png")
    print(f"Saved {evolution_path}")
    print(f"Saved {diagnostic_path}")
