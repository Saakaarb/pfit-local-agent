"""Best-effort reproducibility metadata for fitting runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path


_PACKAGES = ("pfit-local-agent", "jax", "jaxlib", "numpy", "diffrax", "optax", "pyswarms")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_metadata(repo_dir: Path) -> dict:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo_dir), *args], check=True, capture_output=True,
            text=True, timeout=5,
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current") or None,
        "dirty": bool(run("status", "--porcelain")),
    }


def collect_runtime_provenance(
    repo_dir: Path,
    *,
    artifacts: dict[str, Path] | None = None,
    input_reader=None,
    llm_configuration: dict | None = None,
) -> dict:
    """Collect serializable metadata; unavailable fields become explicit errors."""
    repo_dir = Path(repo_dir)
    provenance = {
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "cpu_count": os.cpu_count(),
        },
        "packages": {},
        "artifact_sha256": {},
    }
    try:
        provenance["git"] = _git_metadata(repo_dir)
    except Exception as exc:
        provenance["git"] = {"unavailable": f"{type(exc).__name__}: {exc}"}

    for package in _PACKAGES:
        try:
            provenance["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            provenance["packages"][package] = None
        except Exception as exc:
            provenance["packages"][package] = f"unavailable: {type(exc).__name__}: {exc}"

    for name, path in (artifacts or {}).items():
        try:
            resolved = Path(path)
            provenance["artifact_sha256"][name] = _sha256(resolved) if resolved.is_file() else None
        except Exception as exc:
            provenance["artifact_sha256"][name] = f"unavailable: {type(exc).__name__}: {exc}"

    if input_reader is not None:
        try:
            provenance["optimizer"] = {
                "population_algorithm": input_reader.algorithm,
                "population_random_seed": input_reader.random_seed,
                "population_size": input_reader.n_particles,
                "population_iterations": input_reader.n_iters_pop,
                "gradient_optimizer": input_reader.gradient_optimizer,
                "gradient_iterations": input_reader.n_iters_grad,
            }
        except Exception as exc:
            provenance["optimizer"] = {"unavailable": f"{type(exc).__name__}: {exc}"}
    if llm_configuration is not None:
        provenance["llm_configuration_at_run"] = llm_configuration
    return provenance
