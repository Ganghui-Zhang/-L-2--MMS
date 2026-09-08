"""
Compare long-time AP-CSF curves with equal-area Wulff shapes.

The included numerical curves were computed at T=2 with N=128 and tau=h^2.
One square PDF is produced per anisotropy, matching the four subfigures in the
paper.

Outputs:
  * output/wulff_compare_iso.pdf
  * output/wulff_compare_kfold.pdf
  * output/wulff_compare_riemannian.pdf
  * output/wulff_compare_l4.pdf
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from convergence import manifold_distance, polygon_area
from wulff import cahn_hoffman_curve, rescale_to_area


# =========================================================
# Parameters
# =========================================================
FONT = 24
FIGSIZE = (3.8, 3.8)
N_WULFF = 720

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "precomputed" / "wulff_comparison"
OUTPUT_DIR = ROOT / "output"

CASES = [
    (
        "iso",
        DATA_DIR / "apcsf_T2_iso_ellipse.dat",
        "isotropic",
        {},
    ),
    (
        "kfold",
        DATA_DIR / "apcsf_T2_qfold_q4_weak_ellipse.dat",
        "qfold",
        {"q": 4, "beta": 0.03},
    ),
    (
        "riemannian",
        DATA_DIR / "apcsf_T2_riemannian_ellipse.dat",
        "riemannian",
        {"G_list": [[1.0, 0.0, 2.0]]},
    ),
    (
        "l4",
        DATA_DIR / "apcsf_T2_l4_ellipse.dat",
        "l4",
        {},
    ),
]


# =========================================================
# Build equal-area comparisons
# =========================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
comparisons = []

for output_tag, data_path, aniso_type, aniso_params in CASES:
    numerical_curve = np.loadtxt(data_path)
    wulff_curve = cahn_hoffman_curve(
        aniso_type,
        aniso_params,
        samples=N_WULFF,
    )
    scaled_wulff = rescale_to_area(wulff_curve, polygon_area(numerical_curve))
    error = manifold_distance(numerical_curve, scaled_wulff)
    relative_error = error / polygon_area(numerical_curve)
    print(
        f"{output_tag:11s}: manifold distance={error:.6e}, "
        f"relative={relative_error:.6e}"
    )
    comparisons.append((output_tag, numerical_curve, scaled_wulff))

all_points = np.vstack(
    [np.vstack([numerical, theoretical]) for _, numerical, theoretical in comparisons]
)
axis_limit = 1.12 * np.abs(all_points).max()
plt.rcParams.update({"font.size": FONT, "axes.linewidth": 1.1})

for output_tag, numerical_curve, scaled_wulff in comparisons:
    figure, axis = plt.subplots(figsize=FIGSIZE)
    axis.plot(
        scaled_wulff[:, 0],
        scaled_wulff[:, 1],
        "-",
        color="black",
        linewidth=2.8,
    )
    axis.plot(
        numerical_curve[:, 0],
        numerical_curve[:, 1],
        "--",
        color="tab:red",
        linewidth=2.8,
    )
    axis.set_xlim(-axis_limit, axis_limit)
    axis.set_ylim(-axis_limit, axis_limit)
    axis.set_aspect("equal")
    axis.set_xticks([-1, 0, 1])
    axis.set_yticks([-1, 0, 1])
    axis.tick_params(labelsize=FONT)
    axis.grid(True, linestyle=":", alpha=0.5)
    figure.tight_layout(pad=0.3)
    output_path = OUTPUT_DIR / f"wulff_compare_{output_tag}.pdf"
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output_path}")
