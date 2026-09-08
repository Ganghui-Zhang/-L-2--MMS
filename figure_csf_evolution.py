"""
Reproduce Fig. fig:csf_evolution from the four saved CSF histories.

The precomputed data are included so this plotting script does not rerun the
Firedrake simulations. To regenerate the histories, run evolution_csf.py and
change DATA_FILES below to the corresponding output result.npz files.

Outputs:
  * output/csf_combined_evolution.png
  * output/csf_combined_quantities.png
"""

from pathlib import Path

from io_tools import load_result
from plotting import plot_csf_evolution_summary


# =========================================================
# Parameters
# =========================================================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "precomputed" / "csf_evolution"
OUTPUT_DIR = ROOT / "output"

DATA_FILES = [
    DATA_DIR / "qfold_q4_weak.npz",
    DATA_DIR / "riemannian.npz",
    DATA_DIR / "l4.npz",
    DATA_DIR / "qfold_q4_strong.npz",
]


# =========================================================
# Plot the two manuscript panels
# =========================================================
results = [load_result(path) for path in DATA_FILES]
evolution_path, quantities_path = plot_csf_evolution_summary(results, OUTPUT_DIR)
print(f"Saved {evolution_path}")
print(f"Saved {quantities_path}")
