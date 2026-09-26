"""Run resolution ported from pfit-claude deployed_branch (1b415d8)."""
import json
from pathlib import Path
import yaml

def output_root(session):
    session = Path(session)
    config = session / "inputs" / "user_input.yaml"
    if config.is_file():
        data = yaml.safe_load(config.read_text()) or {}
        return session / (data.get("paths") or {}).get("output_dir", "outputs")
    return session / "outputs"


def resolve_run(session, run=None, require_seed=False):
    """Select an explicit run, latest run, or latest successful seed; read legacy output."""
    # An explicit directory remains readable even if the working config was
    # subsequently removed or is currently being edited into invalid YAML.
    root = Path(run).parent if run and Path(run).is_dir() else output_root(session)
    if run:
        candidate = Path(run)
        candidate = candidate if candidate.is_dir() else root / run
        if not candidate.is_dir():
            raise FileNotFoundError(f"Run not found: {candidate}")
        candidates = [candidate]
    else:
        candidates = sorted((p for p in root.glob("*")
                             if p.is_dir() and (p / "run_manifest.json").is_file()),
                            key=lambda p: p.name, reverse=True)
        candidates.append(root)  # legacy flat outputs are never moved or removed
    for candidate in candidates:
        if require_seed:
            if not (candidate / "final_design_point.csv").is_file():
                continue
            manifest = candidate / "run_manifest.json"
            if manifest.is_file() and json.loads(manifest.read_text())["status"] != "completed":
                continue
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No {'initial guess' if require_seed else 'run'} found in {root}")


def run_sources(run, session):
    snapshot = Path(run) / "snapshot"
    return snapshot if snapshot.is_dir() else Path(session)


def run_config(run, session):
    sources = run_sources(run, session)
    runtime = sources / "inputs" / "run_config.yaml"
    return runtime if runtime.is_file() else sources / "inputs" / "user_input.yaml"
