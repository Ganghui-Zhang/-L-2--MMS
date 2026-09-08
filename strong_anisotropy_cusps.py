"""
Mesh-refinement comparison for strongly anisotropic q-fold CSF.

The experiment uses (q,beta)=(2,0.4), T=0.03, and the splitting constants
C=0.2, 1, 10. Each panel contains a magnified view of the upper cusp.
Precomputed curves are included. Set RUN_SOLVER_IF_MISSING=True to regenerate
them with Firedrake.

Outputs:
  * output/cusp_comparison_N=128.pdf
  * output/cusp_comparison_N=256.pdf
  * output/cusp_comparison_N=512.pdf
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

from convergence import aligned_time_step


# =========================================================
# Parameters
# =========================================================
N_VALUES = [128, 256, 512]
C_VALUES = [0.2, 1.0, 10.0]
T_FINAL = 0.03
RUN_SOLVER_IF_MISSING = False

ZOOM_HALF_WIDTH = 0.15
ZOOM_FACTOR = 5
AXIS_RANGE = 2.6

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "precomputed" / "cusp_comparison"
OUTPUT_DIR = ROOT / "output"


def compute_curves(num_elements):
    """Regenerate one mesh-resolution data set."""
    from mms import solve_csf

    mesh_size = 1.0 / num_elements
    time_step = aligned_time_step([T_FINAL], mesh_size**2)
    curves = []
    for c_value in C_VALUES:
        config = {
            "flow": "csf",
            "aniso_type": "qfold",
            "aniso_params": {"q": 2, "beta": 0.40},
            "initial_shape": "circle",
            "C_override": c_value,
            "N": num_elements,
            "dt": time_step,
            "T_final": T_FINAL,
            "snapshot_times": [T_FINAL],
            "collapse_perimeter": None,
        }
        result = solve_csf(
            f"q2_beta0p4_C{c_value:g}_N{num_elements}",
            config,
            record_diagnostics=False,
        )
        curves.append(result["snapshots"][T_FINAL])
    return np.stack(curves)


def load_or_compute(num_elements):
    """Load portable data, with an optional Firedrake fallback."""
    path = DATA_DIR / f"cusp_data_N{num_elements}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            stored_c = archive["C_values"]
            if not np.allclose(stored_c, C_VALUES):
                raise ValueError(f"Unexpected C values in {path}")
            return archive["curves"].copy()
    if not RUN_SOLVER_IF_MISSING:
        raise FileNotFoundError(
            f"{path} is missing. Set RUN_SOLVER_IF_MISSING=True to regenerate it."
        )
    curves = compute_curves(num_elements)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, C_values=np.asarray(C_VALUES), curves=curves)
    return curves


def plot_comparison(num_elements, curves):
    """Write the three-C cusp comparison at one mesh resolution."""
    reference_curve = curves[0]
    upper_tip = reference_curve[np.argmax(reference_curve[:-1, 1])]
    half_axis = AXIS_RANGE / 2.0
    figure, axes = plt.subplots(1, 3, figsize=(11.4, 4.1))

    for axis, c_value, coordinates in zip(axes, C_VALUES, curves):
        axis.plot(coordinates[:, 0], coordinates[:, 1], "r-", linewidth=1.6)
        axis.plot(coordinates[:-1, 0], coordinates[:-1, 1], "r.", markersize=3.5)
        axis.set_aspect("equal")
        axis.set_xlim(-half_axis, half_axis)
        axis.set_ylim(-half_axis, half_axis)
        axis.tick_params(labelsize=24)
        axis.set_title(rf"$C={c_value:g}$", fontsize=24, pad=8)

        inset = zoomed_inset_axes(axis, ZOOM_FACTOR, loc="lower right", borderpad=1.0)
        inset.plot(coordinates[:, 0], coordinates[:, 1], "r-", linewidth=1.6)
        inset.plot(coordinates[:-1, 0], coordinates[:-1, 1], "r.", markersize=5)
        inset.set_xlim(upper_tip[0] - ZOOM_HALF_WIDTH, upper_tip[0] + ZOOM_HALF_WIDTH)
        inset.set_ylim(upper_tip[1] - ZOOM_HALF_WIDTH, upper_tip[1] + ZOOM_HALF_WIDTH)
        inset.set_aspect("equal")
        inset.set_xticks([])
        inset.set_yticks([])
        mark_inset(axis, inset, loc1=2, loc2=4, fc="none", ec="black", linestyle="--")

    figure.tight_layout()
    output_path = OUTPUT_DIR / f"cusp_comparison_N={num_elements}.pdf"
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output_path}")


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for N in N_VALUES:
    plot_comparison(N, load_or_compute(N))
