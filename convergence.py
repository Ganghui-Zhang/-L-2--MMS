"""Adjacent-refinement convergence tests based on manifold distance."""

from __future__ import annotations

from fractions import Fraction
from functools import reduce
from math import gcd
from pathlib import Path

import numpy as np


def polygon_area(coordinates) -> float:
    """Unsigned polygon area from the shoelace formula."""
    points = np.asarray(coordinates, dtype=float)
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 3:
        return 0.0
    x_coordinate = points[:, 0]
    y_coordinate = points[:, 1]
    return 0.5 * abs(
        np.sum(
            x_coordinate * np.roll(y_coordinate, -1)
            - y_coordinate * np.roll(x_coordinate, -1)
        )
    )


def _shapely_polygon(coordinates):
    from shapely.geometry import Polygon

    points = np.asarray(coordinates, dtype=float)
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    polygon = Polygon(points)
    return polygon if polygon.is_valid else polygon.buffer(0)


def _clip_polygon(subject, clip) -> np.ndarray:
    """Sutherland-Hodgman clipping for a convex, counter-clockwise clip."""
    subject_points = np.asarray(subject, dtype=float)
    clip_points = np.asarray(clip, dtype=float)
    if np.allclose(subject_points[0], subject_points[-1]):
        subject_points = subject_points[:-1]
    if np.allclose(clip_points[0], clip_points[-1]):
        clip_points = clip_points[:-1]

    def inside(point, start, end):
        return (
            (end[0] - start[0]) * (point[1] - start[1])
            - (end[1] - start[1]) * (point[0] - start[0])
        ) >= 0.0

    def intersection(first, second, start, end):
        x1, y1 = first
        x2, y2 = second
        x3, y3 = start
        x4, y4 = end
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denominator) < 1.0e-14:
            return first
        parameter = (
            (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
        ) / denominator
        return np.asarray([x1 + parameter * (x2 - x1), y1 + parameter * (y2 - y1)])

    output = list(subject_points)
    for index, clip_end in enumerate(clip_points):
        if not output:
            break
        clip_start = clip_points[index - 1]
        input_points = output
        output = []
        for point_index, current in enumerate(input_points):
            previous = input_points[point_index - 1]
            if inside(current, clip_start, clip_end):
                if not inside(previous, clip_start, clip_end):
                    output.append(
                        intersection(previous, current, clip_start, clip_end)
                    )
                output.append(current)
            elif inside(previous, clip_start, clip_end):
                output.append(intersection(previous, current, clip_start, clip_end))
    return np.asarray(output)


def manifold_distance(first_curve, second_curve) -> float:
    """Area of the symmetric difference between two polygonal curves."""
    try:
        first_polygon = _shapely_polygon(first_curve)
        second_polygon = _shapely_polygon(second_curve)
        return float(first_polygon.symmetric_difference(second_polygon).area)
    except ImportError:
        intersection = _clip_polygon(first_curve, second_curve)
        return abs(
            polygon_area(first_curve)
            + polygon_area(second_curve)
            - 2.0 * polygon_area(intersection)
        )


def aligned_time_step(test_times, nominal_dt: float) -> float:
    """Largest convenient dt <= nominal_dt aligned with all test times."""
    fractions = [Fraction(time).limit_denominator(10**6) for time in test_times]

    def fraction_gcd(first: Fraction, second: Fraction) -> Fraction:
        numerator = gcd(
            first.numerator * second.denominator,
            second.numerator * first.denominator,
        )
        denominator = first.denominator * second.denominator
        return Fraction(numerator, denominator)

    base_time = float(reduce(fraction_gcd, fractions))
    subdivisions = max(1, int(np.ceil(base_time / nominal_dt)))
    return base_time / subdivisions


def write_convergence_tables(
    output_directory: str | Path,
    flow: str,
    case_name: str,
    config: dict,
    refinements,
    errors: dict[float, list[float]],
    dt_factor: float,
) -> tuple[Path, Path]:
    """Write the error and observed-order tables used by the manuscript."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    prefix = "eoc_apcsf" if flow == "apcsf" else "eoc"
    error_path = output_directory / f"{prefix}_{case_name}.dat"
    order_path = output_directory / f"{prefix}_{case_name}_order.dat"
    test_times = list(config["test_times"])
    mesh_sizes = [1.0 / value for value in refinements[:-1]]

    with error_path.open("w", encoding="utf-8") as stream:
        stream.write(f"# {flow.upper()} adjacent-refinement manifold-distance errors\n")
        stream.write(f"# case = {case_name}, tau = {dt_factor:g} h^2\n")
        stream.write("# h " + " ".join(f"E_M(T={time:g})" for time in test_times) + "\n")
        for index, mesh_size in enumerate(mesh_sizes):
            values = " ".join(f"{errors[time][index]:.10e}" for time in test_times)
            stream.write(f"{mesh_size:.10e} {values}\n")

    with order_path.open("w", encoding="utf-8") as stream:
        stream.write(f"# {flow.upper()} observed convergence orders\n")
        stream.write("# h " + " ".join(f"order(T={time:g})" for time in test_times) + "\n")
        for index, mesh_size in enumerate(mesh_sizes[:-1]):
            values = []
            for time in test_times:
                coarse, fine = errors[time][index : index + 2]
                value = np.log2(coarse / fine) if coarse > 0.0 and fine > 0.0 else np.nan
                values.append(f"{value:.10f}")
            stream.write(f"{mesh_size:.10e} {' '.join(values)}\n")
    return error_path, order_path


def run_convergence(
    flow: str,
    case_name: str,
    config: dict,
    *,
    refinements=(16, 32, 64, 128, 256, 512),
    dt_factor: float = 1.0,
    output_directory: str | Path = ".",
    verbose: bool = True,
) -> dict:
    """Run one coupled space-time convergence study with tau proportional to h^2."""
    from mms import solve_apcsf, solve_csf

    test_times = list(config["test_times"])
    curves_by_refinement = {}
    area_errors = {}

    for num_elements in refinements:
        mesh_size = 1.0 / num_elements
        dt = aligned_time_step(test_times, dt_factor * mesh_size**2)
        run_config = dict(config)
        run_config.update(
            {
                "flow": flow,
                "N": num_elements,
                "dt": dt,
                "T_final": max(test_times),
                "snapshot_times": test_times,
                "collapse_perimeter": None,
            }
        )
        if verbose:
            print(
                f"{flow.upper()} {case_name}: N={num_elements}, "
                f"dt={dt:.6e}, steps={round(max(test_times) / dt)}"
            )
        if flow == "csf":
            result = solve_csf(
                case_name, run_config, record_diagnostics=False, verbose=False
            )
        elif flow == "apcsf":
            result = solve_apcsf(
                case_name, run_config, record_diagnostics=False, verbose=False
            )
        else:
            raise ValueError("flow must be 'csf' or 'apcsf'.")
        curves_by_refinement[num_elements] = result["snapshots"]
        if flow == "apcsf":
            area_errors[num_elements] = float(result["max_relative_area_error"])

    errors = {time: [] for time in test_times}
    for coarse, fine in zip(refinements[:-1], refinements[1:]):
        for time in test_times:
            errors[time].append(
                manifold_distance(
                    curves_by_refinement[coarse][time],
                    curves_by_refinement[fine][time],
                )
            )

    paths = write_convergence_tables(
        output_directory,
        flow,
        case_name,
        config,
        refinements,
        errors,
        dt_factor,
    )
    return {
        "errors": errors,
        "curves": curves_by_refinement,
        "area_errors": area_errors,
        "paths": paths,
    }
