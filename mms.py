"""Shared Firedrake implementation of the anisotropic CSF and AP-CSF schemes.

The directly runnable experiment files in this directory import this module.
Edit the Parameters block in those files rather than editing this module for
ordinary paper reproductions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from firedrake import (
    Constant,
    ConvergenceError,
    Function,
    FunctionSpace,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
    PeriodicIntervalMesh,
    SpatialCoordinate,
    TestFunction,
    TestFunctions,
    VectorFunctionSpace,
    as_vector,
    assemble,
    cos,
    derivative,
    dx,
    inner,
    pi,
    sin,
    split,
    sqrt,
)


CSF_SOLVER_PARAMETERS = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "l2",
    "snes_rtol": 1.0e-10,
    "snes_atol": 1.0e-10,
    "snes_max_it": 100,
    "ksp_type": "preonly",
    "pc_type": "lu",
}


APCSF_SOLVER_PARAMETERS = {
    "mat_type": "nest",
    "snes_type": "newtonls",
    "snes_linesearch_type": "l2",
    "snes_rtol": 1.0e-12,
    "snes_atol": 1.0e-12,
    "snes_stol": 0.0,
    "snes_max_it": 200,
    "ksp_type": "fgmres",
    "ksp_rtol": 1.0e-12,
    "ksp_max_it": 200,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "lu",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "none",
}


def canonicalize_anisotropy(
    aniso_type: str, aniso_params: Mapping | None
) -> tuple[str, dict]:
    """Use the paper's q-fold notation while accepting legacy k-fold inputs."""
    kind = "qfold" if aniso_type == "kfold" else aniso_type
    params = dict(aniso_params or {})
    if kind == "qfold":
        if "q" not in params and "k" in params:
            params["q"] = params.pop("k")
        if "q" not in params or "beta" not in params:
            raise ValueError("q-fold anisotropy requires 'q' and 'beta'.")
    return kind, params


def splitting_constant(
    aniso_type: str,
    aniso_params: Mapping | None,
    override: float | None = None,
) -> float:
    """Return the convex-splitting constant used by the paper's schemes."""
    if override is not None:
        if override < 0.0:
            raise ValueError("The splitting constant C must be nonnegative.")
        return float(override)
    kind, params = canonicalize_anisotropy(aniso_type, aniso_params)
    if kind == "qfold":
        q = int(params["q"])
        beta = abs(float(params["beta"]))
        return max(0.0, beta * (q * q - 1) - 1.0)
    return 0.0


def gamma_hat_ufl(p, aniso_type: str, aniso_params: Mapping | None):
    """One-homogeneous extension gamma_hat(p) in UFL."""
    kind, params = canonicalize_anisotropy(aniso_type, aniso_params)
    radius = sqrt(inner(p, p))

    if kind == "isotropic":
        return radius

    if kind == "qfold":
        q = int(params["q"])
        beta = Constant(float(params["beta"]))
        n2 = p[1] / radius
        if q == 2:
            chebyshev = 2 * n2**2 - 1
        elif q == 3:
            chebyshev = 4 * n2**3 - 3 * n2
        elif q == 4:
            chebyshev = 8 * n2**4 - 8 * n2**2 + 1
        elif q == 6:
            chebyshev = 32 * n2**6 - 48 * n2**4 + 18 * n2**2 - 1
        else:
            raise ValueError("Implemented q-fold orders are q = 2, 3, 4, 6.")
        return radius * (Constant(1.0) + beta * chebyshev)

    if kind == "riemannian":
        matrices = params["G_list"]
        if not matrices:
            raise ValueError("Riemannian anisotropy requires a nonempty G_list.")
        value = Constant(0.0)
        for g11, g12, g22 in matrices:
            value += sqrt(
                Constant(g11) * p[0] ** 2
                + Constant(2.0 * g12) * p[0] * p[1]
                + Constant(g22) * p[1] ** 2
            )
        return value

    if kind == "reg_l1":
        eps = Constant(float(params["eps"]))
        return sqrt(p[0] ** 2 + eps**2 * p[1] ** 2) + sqrt(
            eps**2 * p[0] ** 2 + p[1] ** 2
        )

    if kind == "l4":
        return (p[0] ** 4 + p[1] ** 4 + 1.0e-30) ** 0.25

    raise ValueError(f"Unknown anisotropy type: {aniso_type!r}")


def rot90(vector):
    """Counter-clockwise rotation by 90 degrees."""
    return as_vector([-vector[1], vector[0]])


def initialize_curve(function_space, mesh, shape: str):
    """Interpolate one of the paper's initial curves."""
    curve = Function(function_space, name=f"X0_{shape}")
    xi = SpatialCoordinate(mesh)[0]
    theta = 2.0 * pi * xi
    if shape == "circle":
        expression = as_vector([cos(theta), sin(theta)])
    elif shape == "ellipse":
        expression = as_vector([2.0 * cos(theta), sin(theta)])
    elif shape == "flower":
        radius = 1.0 + 0.3 * cos(5.0 * theta)
        expression = as_vector([radius * cos(theta), radius * sin(theta)])
    else:
        raise ValueError(f"Unknown initial shape: {shape!r}")
    curve.interpolate(expression)
    return curve


def ordered_coordinates(curve, mesh) -> np.ndarray:
    """Return periodic P1 nodes in increasing reference-coordinate order."""
    scalar_space = FunctionSpace(mesh, "CG", 1)
    reference = Function(scalar_space)
    reference.interpolate(SpatialCoordinate(mesh)[0])
    order = np.argsort(reference.dat.data_ro.copy())
    coordinates = curve.dat.data_ro.copy()[order]
    return np.vstack([coordinates, coordinates[0]])


def perimeter(curve) -> float:
    tangent = curve.dx(0)
    return float(assemble(sqrt(inner(tangent, tangent)) * dx))


def signed_area(curve) -> float:
    return float(
        assemble(Constant(0.5) * inner(curve, rot90(curve.dx(0))) * dx)
    )


def anisotropic_energy(curve, aniso_type: str, aniso_params: Mapping) -> float:
    normal_measure = rot90(curve.dx(0))
    return float(assemble(gamma_hat_ufl(normal_measure, aniso_type, aniso_params) * dx))


def mesh_statistics(curve, mesh) -> tuple[float, float]:
    edges = np.linalg.norm(np.diff(ordered_coordinates(curve, mesh), axis=0), axis=1)
    minimum = float(edges.min())
    return float(edges.max() / (minimum + 1.0e-30)), minimum


def _snapshot_steps(snapshot_times: Sequence[float], dt: float) -> dict[int, float]:
    steps = {}
    for requested_time in snapshot_times:
        step = int(round(float(requested_time) / dt))
        actual_time = step * dt
        if abs(actual_time - requested_time) > 0.5 * dt + 1.0e-14:
            raise ValueError(f"Snapshot time {requested_time} is not compatible with dt={dt}.")
        steps[step] = float(requested_time)
    return steps


def _metadata(case_name: str, config: Mapping, c_value: float) -> dict:
    kind, params = canonicalize_anisotropy(
        config["aniso_type"], config.get("aniso_params")
    )
    return {
        "case_name": case_name,
        "flow": config["flow"],
        "aniso_type": kind,
        "aniso_params": params,
        "C_val": float(c_value),
        "shape": config["initial_shape"],
        "N": int(config["N"]),
        "dt": float(config["dt"]),
    }


def solve_csf(
    case_name: str,
    config: Mapping,
    *,
    record_diagnostics: bool = True,
    verbose: bool = True,
) -> dict:
    """Solve anisotropic CSF with the semi-implicit L2-MMS PFEM."""
    cfg = dict(config)
    if cfg.get("flow", "csf") != "csf":
        raise ValueError("solve_csf requires a CSF case.")

    aniso_type, aniso_params = canonicalize_anisotropy(
        cfg["aniso_type"], cfg.get("aniso_params")
    )
    num_elements = int(cfg["N"])
    dt_value = float(cfg["dt"])
    final_time = float(cfg["T_final"])
    snapshot_times = cfg.get("snapshot_times", [0.0, final_time])
    c_value = splitting_constant(
        aniso_type, aniso_params, cfg.get("C_override")
    )

    mesh = PeriodicIntervalMesh(num_elements, 1.0)
    vector_space = VectorFunctionSpace(mesh, "CG", 1, dim=2)
    old_curve = Function(vector_space, name="X_old")
    new_curve = Function(vector_space, name="X_new")
    test_curve = TestFunction(vector_space)
    initial_curve = initialize_curve(vector_space, mesh, cfg["initial_shape"])
    old_curve.assign(initial_curve)
    new_curve.assign(initial_curve)

    dt = Constant(dt_value)
    c_split = Constant(c_value)
    p_new = rot90(new_curve.dx(0))
    q_old = rot90(old_curve.dx(0))
    jacobian_old = sqrt(inner(old_curve.dx(0), old_curve.dx(0)))

    velocity = (
        Constant(1.0) / dt
        * inner(new_curve - old_curve, test_curve)
        * jacobian_old
        * dx
    )
    gamma_c_new = gamma_hat_ufl(p_new, aniso_type, aniso_params) + c_split * sqrt(
        inner(p_new, p_new)
    )
    gamma_c_old = gamma_hat_ufl(q_old, aniso_type, aniso_params) + c_split * sqrt(
        inner(q_old, q_old)
    )
    mixed_energy = gamma_c_new**2 / (Constant(2.0) * gamma_c_old) * dx
    implicit_term = derivative(mixed_energy, new_curve, test_curve)
    explicit_energy = c_split * sqrt(inner(q_old, q_old)) * dx
    explicit_term = derivative(explicit_energy, old_curve, test_curve)
    residual = velocity + implicit_term - explicit_term

    parameters = dict(CSF_SOLVER_PARAMETERS)
    parameters.update(cfg.get("solver_parameters", {}))
    problem = NonlinearVariationalProblem(residual, new_curve)
    solver = NonlinearVariationalSolver(problem, solver_parameters=parameters)

    initial_energy = anisotropic_energy(old_curve, aniso_type, aniso_params)
    initial_perimeter = perimeter(old_curve)
    initial_ratio, initial_minimum = mesh_statistics(old_curve, mesh)
    times = [0.0]
    energies = [initial_energy]
    perimeters = [initial_perimeter]
    mesh_ratios = [initial_ratio]
    min_edges = [initial_minimum]
    newton_iterations = []

    save_steps = _snapshot_steps(snapshot_times, dt_value)
    snapshots = {}
    if 0 in save_steps:
        snapshots[save_steps[0]] = ordered_coordinates(old_curve, mesh)

    num_steps = int(round(final_time / dt_value))
    progress_every = max(1, int(cfg.get("progress_every", max(1, num_steps // 10))))
    collapse_perimeter = cfg.get("collapse_perimeter", 0.02)

    if verbose:
        print(
            f"CSF {case_name}: N={num_elements}, dt={dt_value:g}, "
            f"T={final_time:g}, C={c_value:g}"
        )

    for step in range(1, num_steps + 1):
        new_curve.assign(old_curve)
        try:
            solver.solve()
        except ConvergenceError as exc:
            raise RuntimeError(
                f"Newton solve failed at step {step}, t={step * dt_value:g}"
            ) from exc

        newton_iterations.append(solver.snes.getIterationNumber())
        old_curve.assign(new_curve)

        if step in save_steps:
            snapshots[save_steps[step]] = ordered_coordinates(old_curve, mesh)

        if record_diagnostics:
            current_energy = anisotropic_energy(old_curve, aniso_type, aniso_params)
            current_perimeter = perimeter(old_curve)
            current_ratio, current_minimum = mesh_statistics(old_curve, mesh)
            times.append(step * dt_value)
            energies.append(current_energy)
            perimeters.append(current_perimeter)
            mesh_ratios.append(current_ratio)
            min_edges.append(current_minimum)
        else:
            current_perimeter = None

        if verbose and (step % progress_every == 0 or step == num_steps):
            message = f"  step {step}/{num_steps}, t={step * dt_value:g}"
            if record_diagnostics:
                message += f", E={current_energy:.8e}, Psi={current_ratio:.3e}"
            print(message)

        if (
            record_diagnostics
            and collapse_perimeter is not None
            and current_perimeter < float(collapse_perimeter)
        ):
            break

    result = _metadata(case_name, {**cfg, "aniso_type": aniso_type,
                                   "aniso_params": aniso_params}, c_value)
    result.update(
        {
            "times": np.asarray(times),
            "energies": np.asarray(energies),
            "perimeters": np.asarray(perimeters),
            "mesh_ratios": np.asarray(mesh_ratios),
            "min_edges": np.asarray(min_edges),
            "newton_iters": np.asarray(newton_iterations, dtype=int),
            "snapshots": snapshots,
        }
    )
    return result


def solve_apcsf(
    case_name: str,
    config: Mapping,
    *,
    record_diagnostics: bool = True,
    verbose: bool = True,
) -> dict:
    """Solve area-preserving anisotropic CSF with the constrained L2-MMS PFEM."""
    cfg = dict(config)
    if cfg.get("flow", "apcsf") != "apcsf":
        raise ValueError("solve_apcsf requires an AP-CSF case.")

    aniso_type, aniso_params = canonicalize_anisotropy(
        cfg["aniso_type"], cfg.get("aniso_params")
    )
    num_elements = int(cfg["N"])
    dt_value = float(cfg["dt"])
    final_time = float(cfg["T_final"])
    snapshot_times = cfg.get("snapshot_times", [0.0, final_time])
    c_value = splitting_constant(
        aniso_type, aniso_params, cfg.get("C_override")
    )

    mesh = PeriodicIntervalMesh(num_elements, 1.0)
    vector_space = VectorFunctionSpace(mesh, "CG", 1, dim=2)
    multiplier_space = FunctionSpace(mesh, "R", 0)
    mixed_space = vector_space * multiplier_space

    solution = Function(mixed_space, name="solution")
    new_curve, multiplier = split(solution)
    old_curve = Function(vector_space, name="X_old")
    initial_curve = initialize_curve(vector_space, mesh, cfg["initial_shape"])
    old_curve.assign(initial_curve)
    solution.subfunctions[0].assign(initial_curve)
    solution.subfunctions[1].assign(0.0)

    target_area_value = signed_area(initial_curve)
    target_area = Constant(target_area_value)
    dt = Constant(dt_value)
    c_split = Constant(c_value)

    jacobian_old = sqrt(inner(old_curve.dx(0), old_curve.dx(0)))
    q_old = rot90(old_curve.dx(0))
    p_new = rot90(new_curve.dx(0))
    velocity_energy = (
        Constant(0.5)
        / dt
        * inner(new_curve - old_curve, new_curve - old_curve)
        * jacobian_old
        * dx
    )
    gamma_c_new = gamma_hat_ufl(p_new, aniso_type, aniso_params) + c_split * sqrt(
        inner(p_new, p_new)
    )
    gamma_c_old = gamma_hat_ufl(q_old, aniso_type, aniso_params) + c_split * sqrt(
        inner(q_old, q_old)
    )
    mixed_energy = gamma_c_new**2 / (Constant(2.0) * gamma_c_old) * dx
    area_density = Constant(0.5) * inner(new_curve, rot90(new_curve.dx(0)))
    constraint = multiplier * (area_density - target_area) * dx

    action = velocity_energy + mixed_energy + constraint
    residual = derivative(action, solution, TestFunction(mixed_space))
    curve_test, _ = TestFunctions(mixed_space)
    if c_value > 0.0:
        explicit_gradient = c_split * q_old / sqrt(inner(q_old, q_old))
        residual -= inner(explicit_gradient, rot90(curve_test.dx(0))) * dx

    parameters = dict(APCSF_SOLVER_PARAMETERS)
    parameters.update(cfg.get("solver_parameters", {}))
    problem = NonlinearVariationalProblem(residual, solution)
    solver = NonlinearVariationalSolver(problem, solver_parameters=parameters)

    initial_energy = anisotropic_energy(old_curve, aniso_type, aniso_params)
    initial_perimeter = perimeter(old_curve)
    initial_ratio, initial_minimum = mesh_statistics(old_curve, mesh)
    times = [0.0]
    energies = [initial_energy]
    perimeters = [initial_perimeter]
    areas = [target_area_value]
    mesh_ratios = [initial_ratio]
    min_edges = [initial_minimum]
    multipliers = [0.0]
    newton_iterations = []
    max_relative_area_error = 0.0

    save_steps = _snapshot_steps(snapshot_times, dt_value)
    snapshots = {}
    if 0 in save_steps:
        snapshots[save_steps[0]] = ordered_coordinates(old_curve, mesh)

    num_steps = int(round(final_time / dt_value))
    progress_every = max(1, int(cfg.get("progress_every", max(1, num_steps // 10))))

    if verbose:
        print(
            f"AP-CSF {case_name}: N={num_elements}, dt={dt_value:g}, "
            f"T={final_time:g}, C={c_value:g}"
        )

    for step in range(1, num_steps + 1):
        solution.subfunctions[0].assign(old_curve)
        try:
            solver.solve()
        except ConvergenceError as exc:
            raise RuntimeError(
                f"Newton solve failed at step {step}, t={step * dt_value:g}"
            ) from exc

        newton_iterations.append(solver.snes.getIterationNumber())
        curve_solution = solution.subfunctions[0]
        old_curve.assign(curve_solution)
        current_area = signed_area(old_curve)
        max_relative_area_error = max(
            max_relative_area_error,
            abs(current_area - target_area_value) / abs(target_area_value),
        )

        if step in save_steps:
            snapshots[save_steps[step]] = ordered_coordinates(old_curve, mesh)

        if record_diagnostics:
            current_energy = anisotropic_energy(old_curve, aniso_type, aniso_params)
            current_perimeter = perimeter(old_curve)
            current_ratio, current_minimum = mesh_statistics(old_curve, mesh)
            multiplier_value = float(solution.subfunctions[1].dat.data_ro[0])
            times.append(step * dt_value)
            energies.append(current_energy)
            perimeters.append(current_perimeter)
            areas.append(current_area)
            mesh_ratios.append(current_ratio)
            min_edges.append(current_minimum)
            multipliers.append(multiplier_value)

        if verbose and (step % progress_every == 0 or step == num_steps):
            message = f"  step {step}/{num_steps}, t={step * dt_value:g}"
            if record_diagnostics:
                relative_area_error = abs(current_area - target_area_value) / abs(
                    target_area_value
                )
                message += (
                    f", E={current_energy:.8e}, "
                    f"|dA/A|={relative_area_error:.3e}"
                )
            print(message)

    result = _metadata(case_name, {**cfg, "aniso_type": aniso_type,
                                   "aniso_params": aniso_params}, c_value)
    result.update(
        {
            "A_target": target_area_value,
            "max_relative_area_error": max_relative_area_error,
            "times": np.asarray(times),
            "energies": np.asarray(energies),
            "perimeters": np.asarray(perimeters),
            "areas": np.asarray(areas),
            "mesh_ratios": np.asarray(mesh_ratios),
            "min_edges": np.asarray(min_edges),
            "lambdas": np.asarray(multipliers),
            "newton_iters": np.asarray(newton_iterations, dtype=int),
            "snapshots": snapshots,
        }
    )
    return result
