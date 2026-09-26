"""Post-fit sloppiness, adapted from pfit-claude origin/deployed_branch (1b415d8).

Preserves the deployed run_sloppiness_analysis API, automatic AD/FD fallback,
60-parameter limit, report and spectrum plot. Local additions: JSON/CSV output,
bounded finite differences, finite-value checks, and qualified interpretation.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from lib.utils.run_artifacts import parameter_axes

SLOPPY_DECADES = 6.0
NONIDENT_FLOOR = 1e-6
MAX_NPAR = 60
FD_EPS = 1e-4


def analyze_sloppiness(loss_fn, position, reader, *, method="auto", relative_floor=NONIDENT_FLOOR):
    if method not in {"auto", "ad", "finite-difference"}:
        raise ValueError(f"Unknown sloppiness method: {method}")
    if not 0 < relative_floor < 1:
        raise ValueError("Relative eigenvalue floor must lie between zero and one")
    lo, hi, logs = parameter_axes(reader)
    position = np.asarray(position, dtype=float)
    if position.shape != lo.shape or not np.all(np.isfinite(position)) or np.any(np.abs(position) > 1 + 1e-12):
        raise ValueError("Sloppiness point must be finite and within normalized bounds")
    point = lo + (np.clip(position, -1, 1) + 1) * (hi - lo) / 2
    warnings = []

    def loss_in_coordinates(u):
        return loss_fn(2 * (u - jnp.asarray(lo)) / jnp.asarray(hi - lo) - 1)

    value_grad = jax.jit(jax.value_and_grad(loss_in_coordinates))

    def checked_gradient(u):
        value, gradient = value_grad(jnp.asarray(u, dtype=jnp.float64))
        value, gradient = float(value), np.asarray(gradient, dtype=float)
        if not np.isfinite(value) or value == reader.error_loss or not np.all(np.isfinite(gradient)):
            raise ValueError("Invalid loss/gradient or failed integration during sloppiness analysis")
        return value, gradient

    loss, gradient = checked_gradient(point)
    fallback_reason = None
    fd_error = None
    stencils = None
    if method in {"auto", "ad"}:
        try:
            hessian = np.asarray(jax.hessian(loss_in_coordinates)(jnp.asarray(point)), dtype=float)
            if not np.all(np.isfinite(hessian)):
                raise ValueError("Second-order autodiff returned nonfinite curvature")
            used_method = "ad"
        except Exception as exc:
            if method == "ad":
                raise
            fallback_reason = f"{type(exc).__name__}: {exc}"
            method = "finite-difference"
    if method == "finite-difference":
        hessian_coarse, stencils = _difference_hessian(checked_gradient, point, lo, hi, FD_EPS)
        hessian, _ = _difference_hessian(checked_gradient, point, lo, hi, FD_EPS / 2)
        fd_error = float(np.linalg.norm(hessian - hessian_coarse) / max(np.linalg.norm(hessian), 1e-15))
        if fd_error > 0.1:
            warnings.append("Curvature changes by more than 10% when the finite-difference step is halved.")
        used_method = "finite-difference"
    asymmetry = float(np.linalg.norm(hessian - hessian.T) / max(np.linalg.norm(hessian), 1e-15))
    if asymmetry > 0.1:
        warnings.append("Estimated Hessian has substantial asymmetry before symmetrization.")
    hessian = (hessian + hessian.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
    largest = float(eigenvalues[0])
    magnitude = float(np.max(np.abs(eigenvalues)))
    negative = eigenvalues < -relative_floor * magnitude
    relative = eigenvalues / largest if largest > 0 else None
    weak = (eigenvalues < relative_floor * largest) if largest > 0 else np.ones(len(point), dtype=bool)
    weak &= ~negative
    positive = eigenvalues[eigenvalues > 0]
    spectrum_decades = float(np.log10(positive[0]) - np.log10(positive[-1])) if len(positive) else None
    bound_names = [name for name, x in zip(reader.trainable_parameter_names, position) if abs(x) >= 1 - 1e-6]
    # Assess stationarity in the normalized coordinates used by the optimizer.
    gradient_scaled = gradient * (hi - lo) / 2
    gradient_norm = float(np.max(np.abs(gradient_scaled)))
    if gradient_norm > 1e-5:
        warnings.append("The normalized-coordinate gradient exceeds 1e-5; this point is not certified stationary.")
    if bound_names:
        warnings.append("Parameters are at/near bounds; unconstrained Hessian modes do not describe the feasible region fully.")
    if np.any(negative):
        warnings.append("Negative curvature detected; do not interpret this point as a local minimum.")
        verdict = "indefinite curvature"
    elif largest <= 0:
        verdict = "flat curvature"
    elif np.any(weak):
        verdict = "sloppy: locally weak curvature directions"
    else:
        verdict = "no weak curvature directions at the specified relative floor"
    warnings.append("Local curvature only: not proof of structural/global identifiability or convergence.")
    return {
        "status": "ok", "method": used_method, "fallback_reason": fallback_reason,
        "parameter_names": list(reader.trainable_parameter_names),
        "coordinates": [f"log10({n})" if log else n for n, log in zip(reader.trainable_parameter_names, logs)],
        "point": point.tolist(), "normalized_point": position.tolist(), "loss": loss,
        "gradient": gradient.tolist(), "normalized_gradient_inf_norm": gradient_norm,
        "hessian": hessian.tolist(), "eigenvalues": eigenvalues.tolist(),
        "relative_eigenvalues": relative.tolist() if relative is not None else None,
        "eigenvectors": eigenvectors.tolist(), "relative_floor": relative_floor,
        "positive_spectrum_decades": spectrum_decades,
        "weak_modes": int(np.sum(weak)), "negative_modes": int(np.sum(negative)),
        "parameters_at_bounds": bound_names, "verdict": verdict, "warnings": warnings,
        "stationary_at_tolerance": gradient_norm <= 1e-5,
        "finite_difference_relative_change": fd_error, "finite_difference_stencils": stencils,
        "hessian_relative_asymmetry": asymmetry,
    }


def _difference_hessian(checked_gradient, point, lower, upper, step_scale):
    _, baseline = checked_gradient(point)
    columns, stencils = [], []
    for index in range(len(point)):
        h = min(step_scale * max(1.0, abs(point[index])), (upper[index] - lower[index]) / 4)
        if h <= np.spacing(max(1.0, abs(point[index]))):
            raise ValueError("Parameter interval is too narrow for finite-difference curvature")
        def shifted(offset):
            sample = point.copy()
            sample[index] += offset
            return checked_gradient(sample)[1]
        if point[index] - h >= lower[index] and point[index] + h <= upper[index]:
            column = (shifted(h) - shifted(-h)) / (2 * h)
            stencil = "central"
        elif point[index] + 2 * h <= upper[index]:
            column = (-3 * baseline + 4 * shifted(h) - shifted(2 * h)) / (2 * h)
            stencil = "forward"
        else:
            column = (3 * baseline - 4 * shifted(-h) + shifted(-2 * h)) / (2 * h)
            stencil = "backward"
        columns.append(column)
        stencils.append(stencil)
    return np.column_stack(columns), stencils


def write_sloppiness_report(output_dir, loss_fn, position, reader, *, method="auto", enabled=True):
    output_dir = Path(output_dir)
    if not enabled:
        report = {"status": "skipped", "reason": "Disabled for this run"}
    else:
        try:
            report = analyze_sloppiness(loss_fn, position, reader, method=method)
        except Exception as exc:
            report = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    (output_dir / "sloppiness.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["Post-fit sloppiness", f"Status: {report['status']}"]
    if report["status"] == "ok":
        lines.extend([
            f"Method: {report['method']}", f"Coordinates: {', '.join(report['coordinates'])}",
            f"Verdict: {report['verdict']}", f"Weak modes: {report['weak_modes']}",
            f"Negative modes: {report['negative_modes']}",
            f"Normalized gradient infinity norm: {report['normalized_gradient_inf_norm']:.6g}",
            "Eigenvectors are columns in descending eigenvalue order.",
        ])
        lines.append("Eigenvalues (largest to smallest):")
        lines.extend(f"  mode {index + 1}: {value:.8e}" for index, value in enumerate(report["eigenvalues"]))
        if report["positive_spectrum_decades"] is not None:
            lines.append(f"Positive spectrum span: {report['positive_spectrum_decades']:.6g} decades")
        names = report["parameter_names"]
        vectors = np.asarray(report["eigenvectors"])
        for label, vector in [("STIFFEST direction", vectors[:, 0]), ("SLOPPIEST direction", vectors[:, -1])]:
            indices = np.argsort(np.abs(vector))[::-1][:4]
            lines.append(label + ": " + ", ".join(f"{names[i]}({vector[i]:+.2f})" for i in indices))
        lines.append("Eigenvectors describe parameter combinations, not individual parameter identifiability.")
        if report["fallback_reason"]:
            lines.append(f"AD fallback: {report['fallback_reason']}")
        lines.extend(f"Warning: {warning}" for warning in report["warnings"])
        np.savetxt(output_dir / "sloppiness_hessian.csv", report["hessian"], delimiter=",")
        np.savetxt(output_dir / "sloppiness_eigenvectors.csv", report["eigenvectors"], delimiter=",")
        np.savetxt(output_dir / "sloppiness_eigenvalues.csv", report["eigenvalues"], delimiter=",")
    else:
        lines.append(report["reason"])
    (output_dir / "sloppiness_report.txt").write_text("\n".join(lines) + "\n")
    return report


def run_sloppiness_analysis(compute_loss_problem, constants_list, scaled_best_position,
                            param_names, output_dir, *, method="auto", enabled=True):
    """Deployed pfit-claude interface; never let diagnostics break a fit.

    Constants contain log10 bounds on logarithmic axes, physical bounds otherwise.
    The supplied per-experiment objectives are averaged with equal record weight.
    Returns metrics, or None on skip/failure, as in the deployed implementation.
    """
    try:
        if len(param_names) > MAX_NPAR:
            report = {"status": "skipped", "reason": f"More than {MAX_NPAR} parameters"}
            (Path(output_dir) / "sloppiness.json").write_text(json.dumps(report) + "\n")
            print(f"[sloppiness] {report['reason']}; skipping Hessian diagnostic")
            return None
        c0 = constants_list[0]
        logs = np.asarray(c0["is_logscale"], dtype=bool)
        lo = np.asarray(c0["min_limits"], dtype=float).copy()
        hi = np.asarray(c0["max_limits"], dtype=float).copy()
        lo[logs], hi[logs] = 10 ** lo[logs], 10 ** hi[logs]
        reader = SimpleNamespace(
            trainable_parameter_names=list(param_names), min_axis_values=lo,
            max_axis_values=hi, axis_logscale=logs, error_loss=c0.get("error_loss", 1e10),
        )
        from lib.utils.experiments import mean_experiment_loss
        def averaged_loss(scaled):
            return mean_experiment_loss(compute_loss_problem, constants_list, scaled)
        report = write_sloppiness_report(output_dir, averaged_loss, scaled_best_position, reader,
                                         method=method, enabled=enabled)
        if report["status"] != "ok":
            return None
        # Compatibility metrics returned by the deployed implementation.
        report.update(spread=report["positive_spectrum_decades"], n_nonident=report["weak_modes"],
                      sloppy=report["weak_modes"] > 0 or report["negative_modes"] > 0)
        _write_spectrum_plot(report, Path(output_dir))
        return report
    except Exception as exc:
        print(f"[sloppiness] skipped: {type(exc).__name__}: {exc}")
        return None


def _write_spectrum_plot(report, output_dir):
    # Spectrum visualization ported from the deployed pfit-claude implementation.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        values = np.asarray(report["eigenvalues"])
        scale = np.max(np.abs(values))
        norm = values / scale if scale > 0 else np.zeros_like(values)
        fig, ax = plt.subplots(figsize=(4.6, 5.2))
        y = np.clip(np.abs(norm), 1e-20, None)
        for value, yi in zip(values, y):
            color = "C3" if value < 0 else "C0"
            ax.plot([-0.35, 0.35], [yi, yi], color=color, lw=1.6, zorder=3)
            ax.scatter([0], [yi], s=70, color=color, zorder=4)
        ax.axhspan(NONIDENT_FLOOR, 1.0, color="0.85", alpha=0.6, zorder=0, label="within 6 decades")
        ax.set_yscale("log")
        ax.set_xlim(-1, 1)
        ax.set_xticks([])
        ax.set_ylabel("absolute normalized eigenvalue")
        ax.set_title(f"{output_dir.name}\n{report['weak_modes']} weak modes; red = negative")
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        fig.savefig(output_dir / "sloppiness_spectrum.png", dpi=200)
        plt.close(fig)
    except Exception as exc:
        print(f"[sloppiness] plot skipped: {exc}")
