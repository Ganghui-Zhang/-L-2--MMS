"""Plotting utilities for evolution runs and paper figures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


SNAPSHOT_COLORS = ["#2166ac", "#4dac26", "#d73027", "#b2182b"]
SNAPSHOT_STYLES = ["-", "--", "-.", ":"]
DENSITY_COLORS = ["#2166ac", "#4dac26", "#d73027", "#8e44ad"]
DENSITY_STYLES = ["-", "--", "-.", ":"]


def anisotropy_label(result: Mapping) -> str:
    """Return a publication label using the manuscript's q-fold notation."""
    kind = result["aniso_type"]
    params = result.get("aniso_params", {})
    c_value = float(result.get("C_val", 0.0))
    if kind in {"qfold", "kfold"}:
        q = int(params.get("q", params.get("k")))
        beta = float(params["beta"])
        label = rf"$q$-fold, $(q,\beta)=({q},{beta:g})$"
        if c_value > 0.0:
            label += rf", $C={c_value:g}$"
        return label
    if kind == "riemannian":
        matrix = params["G_list"][0]
        return (
            rf"Riemannian with $M=1$, "
            rf"$G=\mathrm{{diag}}({matrix[0]:g},{matrix[2]:g})$"
        )
    if kind == "l4":
        return r"$\ell^4$-norm"
    if kind == "reg_l1":
        return rf"regularized $\ell^1$, $\varepsilon={params['eps']:g}$"
    if kind == "isotropic":
        return "isotropic"
    return str(kind)


def plot_evolution(result: Mapping, output_path: str | Path) -> Path:
    """Plot one panel per saved curve snapshot."""
    snapshots = result["snapshots"]
    times = sorted(snapshots)
    if not times:
        raise ValueError("The result contains no snapshots.")

    all_coordinates = np.vstack([snapshots[time] for time in times])
    center = 0.5 * (
        all_coordinates.min(axis=0) + all_coordinates.max(axis=0)
    )
    span = np.ptp(all_coordinates, axis=0).max()
    half = 0.54 * span

    columns = len(times) if len(times) <= 4 else (len(times) + 1) // 2
    rows = 1 if len(times) <= 4 else 2
    figure, axes = plt.subplots(rows, columns, figsize=(3.8 * columns, 4.1 * rows))
    axes = np.atleast_1d(axes).ravel()

    for axis, time in zip(axes, times):
        coordinates = snapshots[time]
        axis.plot(coordinates[:, 0], coordinates[:, 1], "r-", linewidth=1.6)
        axis.plot(coordinates[:-1, 0], coordinates[:-1, 1], "r.", markersize=3.5)
        axis.set_aspect("equal")
        axis.set_xlim(center[0] - half, center[0] + half)
        axis.set_ylim(center[1] - half, center[1] + half)
        axis.tick_params(labelsize=24)
        axis.text(
            0.95,
            0.95,
            rf"$t = {time:.2f}$",
            transform=axis.transAxes,
            fontsize=24,
            horizontalalignment="right",
            verticalalignment="top",
        )

    for axis in axes[len(times) :]:
        axis.axis("off")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_diagnostics(result: Mapping, output_path: str | Path) -> Path:
    """Plot energy, mesh quality, iterations, and AP-CSF area error."""
    flow = result["flow"]
    time = np.asarray(result["times"])
    energy = np.asarray(result["energies"])
    mesh_ratio = np.asarray(result["mesh_ratios"])
    iterations = np.asarray(result["newton_iters"])
    panel_count = 4 if flow == "apcsf" else 3
    figure, axes = plt.subplots(1, panel_count, figsize=(5.5 * panel_count, 5.2))

    axes[0].plot(time, energy / energy[0], "b-", linewidth=2.4)
    axes[0].set_xlabel(r"$t$", fontsize=28)
    axes[0].set_ylabel(
        r"$\mathcal{E}_{\gamma}(t)/\mathcal{E}_{\gamma}(0)$", fontsize=28
    )
    axes[0].set_ylim(bottom=0.0)

    offset = 1
    if flow == "apcsf":
        area = np.asarray(result["areas"])
        target = float(result["A_target"])
        axes[1].plot(time, (area - target) / abs(target), "m-", linewidth=1.6)
        axes[1].set_xlabel(r"$t$", fontsize=28)
        axes[1].set_ylabel(r"$\Delta A(t)$", fontsize=28)
        axes[1].ticklabel_format(axis="y", style="scientific", scilimits=(-2, 2))
        offset = 2

    axes[offset].plot(time, mesh_ratio, "r-", linewidth=2.4)
    axes[offset].set_xlabel(r"$t$", fontsize=28)
    axes[offset].set_ylabel(r"$\Psi(t)$", fontsize=28)
    axes[offset].set_yscale("log" if flow == "csf" else "linear")

    iteration_time = time[1 : 1 + len(iterations)]
    axes[offset + 1].plot(iteration_time, iterations, "k-", linewidth=2.0)
    axes[offset + 1].set_xlabel(r"$t$", fontsize=28)
    axes[offset + 1].set_ylabel("Iterations", fontsize=28)
    axes[offset + 1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[offset + 1].set_ylim(bottom=0.0)

    for axis in axes:
        axis.tick_params(labelsize=22)
        axis.grid(True, which="both", alpha=0.25, linewidth=0.5)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_csf_evolution_summary(
    results: Sequence[Mapping], output_directory: str | Path
) -> tuple[Path, Path]:
    """Reproduce the two image files used in fig:csf_evolution."""
    if len(results) != 4:
        raise ValueError("The CSF summary figure requires exactly four cases.")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(10.0, 10.0))
    for axis, result in zip(axes.ravel(), results):
        for index, time in enumerate(sorted(result["snapshots"])):
            coordinates = result["snapshots"][time]
            color = SNAPSHOT_COLORS[index % len(SNAPSHOT_COLORS)]
            style = SNAPSHOT_STYLES[index % len(SNAPSHOT_STYLES)]
            axis.plot(
                coordinates[:, 0],
                coordinates[:, 1],
                linestyle=style,
                color=color,
                linewidth=1.6,
                label=rf"$t = {time:.2f}$",
            )
            axis.plot(
                coordinates[:-1, 0],
                coordinates[:-1, 1],
                ".",
                color=color,
                markersize=3.5,
            )
        axis.set_aspect("equal")
        axis.set_xlim(-1.5, 1.5)
        axis.set_ylim(-1.5, 1.5)
        axis.tick_params(labelsize=24)
        axis.legend(
            fontsize=20,
            loc="best",
            framealpha=0.85,
            edgecolor="0.3",
            fancybox=True,
        )

    figure.tight_layout(pad=2.5)
    evolution_path = output_directory / "csf_combined_evolution.png"
    figure.savefig(evolution_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(24.0, 6.5))
    for index, result in enumerate(results):
        time = np.asarray(result["times"])
        energy = np.asarray(result["energies"])
        mesh_ratio = np.asarray(result["mesh_ratios"])
        iterations = np.asarray(result["newton_iters"])
        label = anisotropy_label(result)
        color = DENSITY_COLORS[index]
        style = DENSITY_STYLES[index]

        axes[0].plot(
            time,
            energy / energy[0],
            linestyle=style,
            color=color,
            linewidth=2.4,
            label=label,
        )
        axes[1].plot(
            time,
            mesh_ratio,
            linestyle=style,
            color=color,
            linewidth=2.4,
            label=label,
        )
        axes[2].plot(
            time[1 : 1 + len(iterations)],
            iterations,
            linestyle=style,
            color=color,
            linewidth=2.4,
            label=label,
        )

    axes[0].set_xlabel(r"$t$", fontsize=30)
    axes[0].set_ylabel(
        r"$\mathcal{E}_{\gamma}(t)\,/\,\mathcal{E}_{\gamma}(0)$",
        fontsize=30,
    )
    axes[0].set_ylim(bottom=0.0)
    axes[1].set_xlabel(r"$t$", fontsize=30)
    axes[1].set_ylabel(r"$\Psi(t)$", fontsize=30)
    axes[1].set_yscale("log")
    axes[2].set_xlabel(r"$t$", fontsize=30)
    axes[2].set_ylabel("Iterations", fontsize=30)
    axes[2].set_ylim(bottom=0.0)
    axes[2].yaxis.set_major_locator(MaxNLocator(integer=True))

    for axis in axes:
        axis.tick_params(labelsize=25)
        axis.grid(True, which="both", alpha=0.25, linewidth=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        fontsize=26,
        loc="upper center",
        ncol=4,
        framealpha=0.85,
        edgecolor="0.3",
        fancybox=True,
        bbox_to_anchor=(0.5, 1.02),
    )
    figure.tight_layout(pad=2.0, rect=(0.0, 0.0, 1.0, 0.90))
    quantities_path = output_directory / "csf_combined_quantities.png"
    figure.savefig(quantities_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return evolution_path, quantities_path
