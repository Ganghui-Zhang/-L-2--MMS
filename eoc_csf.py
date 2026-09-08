"""
Coupled space-time convergence tests for anisotropic CSF.

For each refinement N, the time step is chosen as tau = h^2, with a small
adjustment so all comparison times are reached exactly. The error is the area
of the symmetric difference between curves from adjacent refinements.

Outputs:
  * output/eoc_<case>.dat
  * output/eoc_<case>_order.dat

Edit only the Parameters block below for normal use.
"""

from pathlib import Path

from convergence import run_convergence


# =========================================================
# Parameters
# =========================================================
REFINEMENTS = [16, 32, 64, 128, 256, 512]
DT_FACTOR = 1.0

CASES_TO_RUN = [
    "isotropic_circle",
    "qfold_q4_weak_circle",
    "riemannian_circle",
    "l4_circle",
]

CASES = {
    "isotropic_circle": {
        "flow": "csf",
        "aniso_type": "isotropic",
        "aniso_params": {},
        "initial_shape": "circle",
        "test_times": [0.05, 0.10, 0.30],
    },
    "qfold_q4_weak_circle": {
        "flow": "csf",
        "aniso_type": "qfold",
        "aniso_params": {"q": 4, "beta": 0.03},
        "initial_shape": "circle",
        "test_times": [0.05, 0.10, 0.30],
    },
    "riemannian_circle": {
        "flow": "csf",
        "aniso_type": "riemannian",
        "aniso_params": {"G_list": [[1.0, 0.0, 2.0]]},
        "initial_shape": "circle",
        "test_times": [0.05, 0.10, 0.30],
    },
    "l4_circle": {
        "flow": "csf",
        "aniso_type": "l4",
        "aniso_params": {},
        "initial_shape": "circle",
        "test_times": [0.05, 0.10, 0.30],
    },
    "qfold_q2_strong_circle": {
        "flow": "csf",
        "aniso_type": "qfold",
        "aniso_params": {"q": 2, "beta": 0.40},
        "initial_shape": "circle",
        "C_override": 0.20,
        "test_times": [0.01, 0.02, 0.03],
    },
}

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"


# =========================================================
# Run selected cases
# =========================================================
for case_name in CASES_TO_RUN:
    if case_name not in CASES:
        raise ValueError(f"Unknown CSF convergence case: {case_name}")
    result = run_convergence(
        "csf",
        case_name,
        CASES[case_name],
        refinements=REFINEMENTS,
        dt_factor=DT_FACTOR,
        output_directory=OUTPUT_DIR,
    )
    for path in result["paths"]:
        print(f"Saved {path}")
