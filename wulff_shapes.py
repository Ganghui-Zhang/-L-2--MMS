"""
Generate the Cahn-Hoffman/Wulff boundary data used in the paper.

For a unit normal N, the weakly anisotropic Wulff boundary is traced by

    xi(N) = grad gamma_hat(N).

Outputs are two-column text files suitable for pgfplots.
"""

from pathlib import Path

from wulff import cahn_hoffman_curve, write_curve


# =========================================================
# Parameters
# =========================================================
N_SAMPLES = 720

CASES = {
    "iso": ("isotropic", {}),
    "kfold_k4_weak": ("qfold", {"q": 4, "beta": 0.03}),
    "riemannian": ("riemannian", {"G_list": [[1.0, 0.0, 2.0]]}),
    "l4": ("l4", {}),
}

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"


# =========================================================
# Generate tables
# =========================================================
for output_tag, (aniso_type, aniso_params) in CASES.items():
    curve = cahn_hoffman_curve(
        aniso_type,
        aniso_params,
        samples=N_SAMPLES,
    )
    path = write_curve(
        OUTPUT_DIR / f"wulff_{output_tag}.dat",
        curve,
        header=(
            f"Wulff boundary: type={aniso_type}, params={aniso_params}; "
            "columns x y"
        ),
    )
    print(f"Saved {path}")
