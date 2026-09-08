"""NumPy utilities for Cahn-Hoffman and Wulff curves."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from convergence import polygon_area


def gamma_hat_numpy(points, aniso_type: str, aniso_params: dict | None = None):
    """Evaluate the solver's one-homogeneous anisotropy with NumPy."""
    parameters = dict(aniso_params or {})
    kind = "qfold" if aniso_type == "kfold" else aniso_type
    if kind == "qfold" and "q" not in parameters and "k" in parameters:
        parameters["q"] = parameters.pop("k")

    points = np.asarray(points, dtype=float)
    first = points[..., 0]
    second = points[..., 1]
    radius = np.sqrt(first**2 + second**2)

    if kind == "isotropic":
        return radius
    if kind == "qfold":
        q = int(parameters["q"])
        beta = float(parameters["beta"])
        normal_second = second / (radius + 1.0e-30)
        if q == 2:
            chebyshev = 2.0 * normal_second**2 - 1.0
        elif q == 3:
            chebyshev = 4.0 * normal_second**3 - 3.0 * normal_second
        elif q == 4:
            chebyshev = (
                8.0 * normal_second**4 - 8.0 * normal_second**2 + 1.0
            )
        elif q == 6:
            chebyshev = (
                32.0 * normal_second**6
                - 48.0 * normal_second**4
                + 18.0 * normal_second**2
                - 1.0
            )
        else:
            raise ValueError("Implemented q-fold orders are q = 2, 3, 4, 6.")
        return radius * (1.0 + beta * chebyshev)
    if kind == "riemannian":
        value = np.zeros_like(radius)
        for g11, g12, g22 in parameters["G_list"]:
            value += np.sqrt(
                g11 * first**2 + 2.0 * g12 * first * second + g22 * second**2
            )
        return value
    if kind == "reg_l1":
        epsilon = float(parameters["eps"])
        return np.sqrt(first**2 + epsilon**2 * second**2) + np.sqrt(
            epsilon**2 * first**2 + second**2
        )
    if kind == "l4":
        return (first**4 + second**4) ** 0.25
    raise ValueError(f"Unknown anisotropy type: {aniso_type!r}")


def cahn_hoffman_curve(
    aniso_type: str,
    aniso_params: dict | None = None,
    *,
    samples: int = 720,
    finite_difference_step: float = 1.0e-7,
) -> np.ndarray:
    """Trace grad(gamma_hat)(N) for N on the unit circle."""
    theta = np.linspace(0.0, 2.0 * np.pi, samples + 1)
    normals = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    horizontal = np.asarray([finite_difference_step, 0.0])
    vertical = np.asarray([0.0, finite_difference_step])
    gradient_first = (
        gamma_hat_numpy(normals + horizontal, aniso_type, aniso_params)
        - gamma_hat_numpy(normals - horizontal, aniso_type, aniso_params)
    ) / (2.0 * finite_difference_step)
    gradient_second = (
        gamma_hat_numpy(normals + vertical, aniso_type, aniso_params)
        - gamma_hat_numpy(normals - vertical, aniso_type, aniso_params)
    ) / (2.0 * finite_difference_step)
    return np.stack([gradient_first, gradient_second], axis=-1)


def rescale_to_area(curve, target_area: float) -> np.ndarray:
    """Scale a closed curve so its polygonal area is target_area."""
    coordinates = np.asarray(curve, dtype=float)
    current_area = polygon_area(coordinates)
    if current_area <= 0.0:
        raise ValueError("Cannot rescale a curve with zero area.")
    return coordinates * np.sqrt(abs(target_area) / current_area)


def write_curve(path: str | Path, curve, header: str = "") -> Path:
    """Write a two-column curve table."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(destination, np.asarray(curve), header=f"{header}\nx y".strip())
    return destination
