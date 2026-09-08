"""
Strongly anisotropic AP-CSF evolution with the Wulff curve overlay.

The experiment uses an ellipse, (q,beta)=(2,0.4), C=0.2, N=128, and
tau=5e-4. The full Cahn-Hoffman curve is scaled so that its convex hull has
the area preserved by AP-CSF.

Output:
  * output/apcsf_k2_strong_wulff.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

from io_tools import load_result
from wulff import cahn_hoffman_curve


# =========================================================
# Parameters
# =========================================================
SNAPSHOT_TIMES = [0.0, 0.30, 1.50, 3.50]
N_WULFF = 720

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "precomputed" / "apcsf_strong" / "apcsf_q2_strong_ellipse.npz"
OUTPUT_DIR = ROOT / "output"


def polygon_signed_area(coordinates):
    x_coordinate = coordinates[:, 0]
    y_coordinate = coordinates[:, 1]
    return 0.5 * np.sum(
        x_coordinate[:-1] * y_coordinate[1:]
        - x_coordinate[1:] * y_coordinate[:-1]
    )


data = load_result(DATA_FILE)
snapshots = data["snapshots"]
target_area = abs(float(data["A_target"]))

raw_wulff = cahn_hoffman_curve(
    "qfold",
    {"q": 2, "beta": 0.40},
    samples=N_WULFF,
)
hull = ConvexHull(raw_wulff[:-1])
hull_curve = raw_wulff[:-1][hull.vertices]
hull_curve = np.vstack([hull_curve, hull_curve[0]])
hull_area = abs(polygon_signed_area(hull_curve))
scaled_wulff = raw_wulff * np.sqrt(target_area / hull_area)

all_coordinates = np.vstack(
    [snapshots[time] for time in SNAPSHOT_TIMES] + [scaled_wulff]
)
center = 0.5 * (
    all_coordinates.min(axis=0) + all_coordinates.max(axis=0)
)
half_width = 0.55 * np.ptp(all_coordinates, axis=0).max()

figure, axes = plt.subplots(1, 4, figsize=(15.2, 4.3))
for axis, time in zip(axes, SNAPSHOT_TIMES):
    coordinates = snapshots[time]
    axis.plot(coordinates[:, 0], coordinates[:, 1], "r-", linewidth=1.6)
    axis.plot(coordinates[:-1, 0], coordinates[:-1, 1], "r.", markersize=3.5)
    if time == SNAPSHOT_TIMES[-1]:
        axis.plot(
            scaled_wulff[:, 0],
            scaled_wulff[:, 1],
            "k-",
            linewidth=2.0,
            label="Wulff shape",
        )
    axis.set_aspect("equal")
    axis.set_xlim(center[0] - half_width, center[0] + half_width)
    axis.set_ylim(center[1] - half_width, center[1] + half_width)
    axis.tick_params(labelsize=24)
    axis.text(
        0.95,
        0.95,
        rf"$t={time:.2f}$",
        transform=axis.transAxes,
        fontsize=24,
        horizontalalignment="right",
        verticalalignment="top",
    )

axes[-1].legend(
    fontsize=18,
    loc="lower left",
    framealpha=0.85,
    edgecolor="0.3",
    fancybox=True,
)
figure.tight_layout(pad=1.5)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = OUTPUT_DIR / "apcsf_k2_strong_wulff.png"
figure.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close(figure)
print(f"Saved {output_path}")
