#!/usr/bin/env python3
"""Diagnose autodiff fragility for a completed pfit run.

This is a cheap local probe, not a refit. It rebuilds the run from its snapshot,
evaluates the fitted loss, attempts `jax.value_and_grad`, compares against
finite differences in scaled parameter space, checks nearby perturbations, and
scans the generated model for common non-smooth expressions.

Usage:
    ./venv/bin/python3 tools/autodiff_diagnose.py <session_name> --run <run_id>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lib.utils.dataset_io import load_dataset
from lib.utils.helper_functions import CreatedClass, get_input_reader
from lib.utils.run_store import resolve_run, run_config, run_sources


USER_MODEL_NONSMOOTH_PATTERNS = {
    "state-dependent branch": r"\bjnp\.where\s*\(|\bnp\.where\s*\(",
    "hard clipping": r"\bjnp\.clip\s*\(|\bnp\.clip\s*\(",
    "absolute value cusp": r"\bjnp\.abs\s*\(|\bnp\.abs\s*\(",
    "sign function": r"\bjnp\.sign\s*\(|\bnp\.sign\s*\(",
    "integer/floor operation": r"\bjnp\.(floor|ceil|round|rint)\s*\(|\bnp\.(floor|ceil|round|rint)\s*\(",
    "arg min/max": r"\bjnp\.arg(min|max)\s*\(|\bnp\.arg(min|max)\s*\(",
    "hard comparison": r"(?<![<>=!])([<>]=?)(?![<>=])",
}

GENERATED_SCRIPT_PATTERNS = {
    "state-dependent branch": r"\bjnp\.where\s*\(|\bnp\.where\s*\(",
    "hard clipping": r"\bjnp\.clip\s*\(|\bnp\.clip\s*\(",
    "absolute value cusp": r"\bjnp\.abs\s*\(|\bnp\.abs\s*\(",
    "sign function": r"\bjnp\.sign\s*\(|\bnp\.sign\s*\(",
    "arg min/max": r"\bjnp\.arg(min|max)\s*\(|\bnp\.arg(min|max)\s*\(",
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _resolve_session(name: str | None) -> Path:
    root = REPO_ROOT / "sessions"
    if name:
        session = Path(name)
        if not session.is_dir():
            session = root / name
        if not session.is_dir():
            raise FileNotFoundError(f"Session directory not found: {session}")
        return session
    candidates = sorted((p for p in root.iterdir() if p.is_dir()),
                        key=lambda p: p.stat().st_ctime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No sessions found")
    return candidates[0]


def _load_generated(gen_path: Path):
    module_name = f"generated_script_autodiff_{abs(hash(gen_path))}"
    spec = importlib.util.spec_from_file_location(module_name, gen_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load generated script: {gen_path}")
    gen = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = gen
    spec.loader.exec_module(gen)
    return gen


def _build_problem(session: Path, run: Path):
    sources = run_sources(run, session)
    gen = _load_generated(sources / "generated" / "generated_script.py")
    input_reader = get_input_reader(run_config(run, session))
    input_reader.check_name_uniqueness()
    input_reader.output_dir = run

    experiments = []
    for exp in input_reader.experiments:
        dpath = sources / input_reader.user_input_dirname / exp["filename"]
        data = load_dataset(dpath)
        experiments.append({
            "t_eval": data[:, 0],
            "dataset": data[:, 1:],
            "y0": jnp.array(input_reader.get_y0(len(experiments))),
        })

    prob = CreatedClass(experiments, input_reader,
                        gen._compute_loss_problem, gen._write_problem_result)

    is_log = list(input_reader.axis_logscale)
    lo, hi = [], []
    for i, log_axis in enumerate(is_log):
        mn = float(input_reader.min_axis_values[i])
        mx = float(input_reader.max_axis_values[i])
        lo.append(float(np.log10(mn)) if log_axis else mn)
        hi.append(float(np.log10(mx)) if log_axis else mx)
    prob.set_min_limit(lo)
    prob.set_max_limit(hi)
    prob.set_is_logscale(is_log)
    return input_reader, prob, np.asarray(lo), np.asarray(hi), np.asarray(is_log, dtype=bool)


def _scaled_from_physical(theta: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                          is_log: np.ndarray) -> np.ndarray:
    u = np.where(is_log, np.log10(theta), theta)
    return 2.0 * (u - lo) / (hi - lo) - 1.0


def _safe_float(value: Any) -> tuple[float | None, str | None]:
    try:
        out = float(value)
        if not np.isfinite(out):
            return out, "non-finite"
        return out, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _call_loss(loss_fn, x: np.ndarray) -> dict[str, Any]:
    try:
        value = loss_fn(jnp.asarray(x))
        value.block_until_ready()
        val, err = _safe_float(value)
        return {"ok": err is None, "value": val, "error": err}
    except Exception as exc:
        return {"ok": False, "value": None, "error": f"{type(exc).__name__}: {exc}"}


def _call_grad(loss_fn, x: np.ndarray) -> dict[str, Any]:
    try:
        value, grad = jax.value_and_grad(loss_fn)(jnp.asarray(x))
        value.block_until_ready()
        grad.block_until_ready()
        value_np = float(value)
        grad_np = np.asarray(grad, dtype=float)
        finite = np.isfinite(value_np) and np.all(np.isfinite(grad_np))
        return {
            "ok": bool(finite),
            "value": value_np,
            "grad": grad_np,
            "grad_inf_norm": float(np.nanmax(np.abs(grad_np))) if grad_np.size else 0.0,
            "error": None if finite else "non-finite value or gradient",
        }
    except Exception as exc:
        return {
            "ok": False,
            "value": None,
            "grad": None,
            "grad_inf_norm": None,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }


def _finite_difference(loss_fn, x: np.ndarray, eps: float,
                       error_loss: float) -> dict[str, Any]:
    rows = []
    grad = np.full_like(x, np.nan, dtype=float)
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = eps
        plus = _call_loss(loss_fn, np.clip(x + step, -1.0, 1.0))
        minus = _call_loss(loss_fn, np.clip(x - step, -1.0, 1.0))
        if plus["ok"] and minus["ok"]:
            grad[i] = (plus["value"] - minus["value"]) / (2.0 * eps)
        penalized = [
            side for side in (plus, minus)
            if side["value"] is not None and abs(side["value"] - error_loss) <= max(1e-8, 1e-8 * abs(error_loss))
        ]
        rows.append({
            "axis": i,
            "plus_ok": plus["ok"],
            "minus_ok": minus["ok"],
            "plus_value": plus["value"],
            "minus_value": minus["value"],
            "penalized_sides": len(penalized),
            "plus_error": plus["error"],
            "minus_error": minus["error"],
        })
    return {"grad": grad, "rows": rows}


def _perturbation_probe(loss_fn, x: np.ndarray, eps: float, n_random: int,
                        seed: int, error_loss: float) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    probes = []
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = eps
        probes.append(("axis+", np.clip(x + step, -1.0, 1.0)))
        probes.append(("axis-", np.clip(x - step, -1.0, 1.0)))
    for _ in range(n_random):
        direction = rng.normal(size=x.size)
        norm = np.linalg.norm(direction)
        if norm:
            direction /= norm
        probes.append(("random", np.clip(x + eps * direction, -1.0, 1.0)))

    counts = {"ok": 0, "failed": 0, "penalized": 0, "nonfinite": 0}
    examples = []
    for kind, point in probes:
        result = _call_loss(loss_fn, point)
        value = result["value"]
        penalized = value is not None and abs(value - error_loss) <= max(1e-8, 1e-8 * abs(error_loss))
        if result["ok"]:
            counts["ok"] += 1
        else:
            counts["failed"] += 1
        if penalized:
            counts["penalized"] += 1
        if value is not None and not np.isfinite(value):
            counts["nonfinite"] += 1
        if (not result["ok"] or penalized) and len(examples) < 6:
            examples.append({"kind": kind, "value": value, "error": result["error"]})
    counts["total"] = len(probes)
    return {"counts": counts, "examples": examples}


def _scan_model(sources: Path) -> list[dict[str, Any]]:
    hits = []
    benign_mask_tokens = (
        "valid_", "F_safe", "T_safe", "count_", "F_scale", "T_scale",
        "np.isfinite", "jnp.isfinite", "nanmax", "ign_scale",
    )
    for rel, patterns in (
        ("generated/user_model.py", USER_MODEL_NONSMOOTH_PATTERNS),
        ("generated/generated_script.py", GENERATED_SCRIPT_PATTERNS),
    ):
        path = sources / rel
        if not path.is_file():
            continue
        text = path.read_text()
        for label, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                line = text.count("\n", 0, match.start()) + 1
                snippet = text.splitlines()[line - 1].strip()
                if any(token in snippet for token in benign_mask_tokens):
                    continue
                hits.append({"file": rel, "line": line, "kind": label, "text": snippet})
    return hits


def _read_logs(run: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"node_failure": None, "stage1_final": None, "node_first": None}
    node = run / "NODE_fitting.log"
    if node.is_file():
        text = node.read_text(errors="replace")
        for needle in ("linear solver returned non-finite", "max_steps", "dt_min", "non-finite", "CpuCallback"):
            if needle.lower() in text.lower():
                out["node_failure"] = needle
                break
        vals = []
        for line in text.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0].isdigit():
                try:
                    vals.append(float(parts[1]))
                except ValueError:
                    pass
        if vals:
            out["node_first"] = vals[0]
            out["node_final"] = vals[-1]
    for name in ("de_fitting.log", "pso_fitting.log"):
        path = run / name
        if not path.is_file():
            continue
        vals = []
        for line in path.read_text(errors="replace").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    vals.append(float(parts[1]))
                except ValueError:
                    pass
        if vals:
            out["stage1_log"] = name
            out["stage1_final"] = vals[-1]
            break
    return out


def _recommend(results: dict[str, Any]) -> list[str]:
    recs = []
    grad = results["autodiff_gradient"]
    static_hits = results["static_nonsmooth_hits"]
    perturb = results["perturbation_probe"]["counts"]
    logs = results["logs"]

    if not grad["ok"]:
        recs.append("Autodiff failed at the final design point. First check for non-smooth loss/model constructs and solver failures before changing optimizer settings.")
        if grad.get("nonfinite_axes"):
            recs.append("The non-finite gradient axes are: " + ", ".join(grad["nonfinite_axes"]) + ". Prioritize those parameters' transformations, bounds, and rate-law sensitivities.")
        err = (grad.get("error") or "").lower()
        if "linear solver" in err or "cpucallback" in err or logs.get("node_failure"):
            recs.append("The failure signature points into the diffrax implicit solve/adjoint path. Try an explicit adaptive solver if the RHS has hard switches; otherwise keep a stiff solver but smooth the trigger/loss and loosen gradient tolerances one notch before tightening again.")
    elif grad["grad_inf_norm"] is not None and grad["grad_inf_norm"] > 1e3 * max(abs(grad["value"] or 0.0), 1.0):
        recs.append("The gradient is finite but very large relative to the loss. Check parameter bounds, loss normalization, and active clipping at the fitted point.")

    if static_hits:
        kinds = sorted({h["kind"] for h in static_hits})
        recs.append("Static scan found non-smooth constructs: " + ", ".join(kinds) + ". Replace event-like hard switches with smooth surrogates, or keep such quantities out of the differentiated loss.")

    if perturb["failed"] or perturb["penalized"] or perturb["nonfinite"]:
        recs.append("Nearby perturbations hit failed/non-finite/penalized losses. That means the final point sits near a numerical wall; narrow the offending bounds or make the solver/loss smoother before relying on gradient refinement.")

    comp = results.get("gradient_vs_finite_difference") or {}
    if comp.get("max_relative_error") is not None and comp["max_relative_error"] > 1e2:
        recs.append("Autodiff and finite-difference gradients disagree strongly. This usually indicates solver tolerance noise, discontinuities in the objective, or an adjoint path that is unstable for this solver.")

    if not recs:
        recs.append("No local autodiff pathology was detected. If refinement still fails, inspect longer-range parameter moves or try a gradient-only run with a smaller learning rate.")
    return recs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", nargs="?", help="session name or path; default: newest session")
    parser.add_argument("--run", help="run ID or directory; default: latest run")
    parser.add_argument("--eps", type=float, default=1e-4,
                        help="scaled-space finite-difference/probe step")
    parser.add_argument("--random-probes", type=int, default=16,
                        help="number of random nearby perturbations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-write", action="store_true",
                        help="print only; do not write autodiff_diagnosis.* files")
    args = parser.parse_args()

    session = _resolve_session(args.session)
    run = resolve_run(session, args.run)
    sources = run_sources(run, session)
    input_reader, prob, lo, hi, is_log = _build_problem(session, run)
    theta = np.atleast_1d(np.loadtxt(run / "final_design_point.csv", delimiter=","))
    scaled = _scaled_from_physical(theta, lo, hi, is_log)

    loss_fn = prob._compute_loss
    base = _call_loss(loss_fn, scaled)
    grad = _call_grad(loss_fn, scaled)
    fd = _finite_difference(loss_fn, scaled, args.eps, input_reader.error_loss)
    perturb = _perturbation_probe(loss_fn, scaled, args.eps, args.random_probes,
                                  args.seed, input_reader.error_loss)

    comparison = None
    if grad["grad"] is not None:
        ad = np.asarray(grad["grad"], dtype=float)
        nonfinite_axes = [name for name, value in zip(input_reader.trainable_parameter_names, ad)
                          if not np.isfinite(value)]
        grad["nonfinite_axes"] = nonfinite_axes
        mask = np.isfinite(ad) & np.isfinite(fd["grad"])
        if np.any(mask):
            denom = np.maximum(np.maximum(np.abs(ad[mask]), np.abs(fd["grad"][mask])), 1e-12)
            rel = np.abs(ad[mask] - fd["grad"][mask]) / denom
            comparison = {
                "compared_axes": int(mask.sum()),
                "max_absolute_error": float(np.max(np.abs(ad[mask] - fd["grad"][mask]))),
                "max_relative_error": float(np.max(rel)),
            }

    results = {
        "session": str(session),
        "run": str(run),
        "integrator": input_reader.integrator,
        "gradient_optimizer": input_reader.gradient_optimizer,
        "gradient_rtol": [float(x) for x in np.atleast_1d(input_reader.stepsize_rtol)],
        "gradient_atol": [float(x) for x in np.atleast_1d(input_reader.stepsize_atol)],
        "initial_timestep": input_reader.init_timestep,
        "max_steps": input_reader.max_steps,
        "error_loss": input_reader.error_loss,
        "trainable_parameters": input_reader.trainable_parameter_names,
        "final_design_point": theta,
        "scaled_design_point": scaled,
        "base_loss": base,
        "autodiff_gradient": grad,
        "finite_difference": fd,
        "gradient_vs_finite_difference": comparison,
        "perturbation_probe": perturb,
        "static_nonsmooth_hits": _scan_model(sources),
        "logs": _read_logs(run),
    }
    results["recommendations"] = _recommend(results)

    lines = [
        "Autodiff diagnosis",
        "=" * 60,
        f"session              : {session.name}",
        f"run                  : {run.name}",
        f"integrator           : {input_reader.integrator}",
        f"gradient optimizer   : {input_reader.gradient_optimizer}",
        f"rtol / atol          : {results['gradient_rtol']} / {results['gradient_atol']}",
        f"initial dt / maxsteps: {input_reader.init_timestep} / {input_reader.max_steps}",
        "",
        f"base loss            : {base['value']} ({'ok' if base['ok'] else base['error']})",
        f"value_and_grad       : {'ok' if grad['ok'] else grad['error']}",
    ]
    if grad["grad_inf_norm"] is not None:
        lines.append(f"|grad|_inf          : {grad['grad_inf_norm']:.6g}")
    if grad.get("nonfinite_axes"):
        lines.append(f"non-finite axes     : {', '.join(grad['nonfinite_axes'])}")
    if comparison:
        lines.append(f"FD comparison axes  : {comparison['compared_axes']}")
        lines.append(f"max |AD-FD|         : {comparison['max_absolute_error']:.6g}")
        lines.append(f"max relative error  : {comparison['max_relative_error']:.6g}")
    c = perturb["counts"]
    lines += [
        "",
        f"nearby probes        : {c['ok']}/{c['total']} ok, {c['failed']} failed, "
        f"{c['penalized']} at error_loss, {c['nonfinite']} non-finite",
        f"static nonsmooth hits: {len(results['static_nonsmooth_hits'])}",
    ]
    for hit in results["static_nonsmooth_hits"][:12]:
        lines.append(f"  - {hit['file']}:{hit['line']} {hit['kind']}: {hit['text']}")
    if len(results["static_nonsmooth_hits"]) > 12:
        lines.append(f"  - ... {len(results['static_nonsmooth_hits']) - 12} more")
    lines += ["", "Recommendations:"]
    lines += [f"- {rec}" for rec in results["recommendations"]]
    report = "\n".join(lines) + "\n"
    print(report)

    if not args.no_write:
        (run / "autodiff_diagnosis.txt").write_text(report)
        serializable = dict(results)
        if serializable["autodiff_gradient"].get("grad") is not None:
            serializable["autodiff_gradient"]["grad"] = np.asarray(
                serializable["autodiff_gradient"]["grad"]).tolist()
        serializable["finite_difference"]["grad"] = np.asarray(
            serializable["finite_difference"]["grad"]).tolist()
        (run / "autodiff_diagnosis.json").write_text(
            json.dumps(serializable, indent=2, default=_json_default) + "\n")
        print(f"Wrote {run / 'autodiff_diagnosis.txt'}")
        print(f"Wrote {run / 'autodiff_diagnosis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
