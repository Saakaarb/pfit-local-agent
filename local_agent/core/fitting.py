import os
import shutil
import sys
import json
import yaml
from datetime import datetime
from pathlib import Path


def run_driver(session_dir: Path, input_reader, output_dir_override: Path | None = None,
               *, from_run: str | None = None, allow_legacy_seed: bool = False, gradient_only: bool = False,
               sloppiness: bool = True, sloppiness_method: str = "auto"):
    """Fit a session or warm-start refinement; preserve every earlier run."""
    session_path = Path(session_dir)
    path_to_input = session_path / input_reader.user_input_dirname / "user_input.yaml"
    output_dir = Path(output_dir_override) if output_dir_override is not None else make_run_output_dir(session_path)
    generated_dir = session_path / input_reader.generated_dirname
    generated_script = generated_dir / "generated_script.py"
    if not generated_script.exists():
        raise FileNotFoundError(f"generated_script.py not found at {generated_script}\nRun pfit jax <session_dir> first.")
    from local_agent.agent.readiness import check_ready
    from local_agent.agent.validators import parse_input_yaml
    report = check_ready(session_path)
    if not report.passed:
        raise ValueError("Session is not ready to fit:\n" + "\n".join(report.critical_errors))
    for warning in report.warnings:
        print(f"Readiness warning: {warning}")
    input_reader = parse_input_yaml(path_to_input)
    initial_parameters = None
    source_dir = None
    gradient_only = gradient_only or from_run is not None
    if gradient_only:
        from lib.utils.run_store import resolve_run, run_config
        from lib.utils.run_artifacts import load_restart_seed
        source_dir = resolve_run(session_path, from_run, require_seed=True)
        legacy_allowed = allow_legacy_seed
        if not (source_dir / "final_parameters.json").exists() and (source_dir / "snapshot").is_dir():
            from lib.utils.yamlread import YAMLReader
            seed_reader = YAMLReader.from_file(run_config(source_dir, session_path))
            if seed_reader.trainable_parameter_names != input_reader.trainable_parameter_names:
                raise ValueError("Seed run's trainable parameter names/order differ from this session")
            legacy_allowed = True
        initial_parameters = load_restart_seed(source_dir, input_reader, allow_legacy=legacy_allowed)
        if not (source_dir / "final_parameters.json").exists() and allow_legacy_seed:
            print("Legacy seed: using the current YAML parameter order as explicitly requested")
    elif allow_legacy_seed:
        raise ValueError("--allow-legacy-seed requires gradient-only mode")
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
    import jax
    from lib.utils.helper_functions import fit_generic_system, fit_gradient_only_system
    print("Launching driver script")
    print("Available devices: ", jax.devices("cpu"))
    # Never delete an existing run, including an explicitly selected directory.
    output_dir.mkdir(parents=True, exist_ok=False)
    # Snapshot layout and path-adjusted runtime YAML follow deployed pfit-claude.
    snapshot_dir = output_dir / "snapshot"
    snapshot_inputs = snapshot_dir / "inputs"
    snapshot_generated = snapshot_dir / "generated"
    snapshot_inputs.mkdir(parents=True)
    snapshot_generated.mkdir()
    shutil.copy2(path_to_input, snapshot_inputs / "user_input.yaml")
    config = yaml.safe_load(path_to_input.read_text())
    source_data = session_path / input_reader.user_input_dirname / input_reader.filename_data
    dataset_name = "dataset_1.csv"
    shutil.copy2(source_data, snapshot_inputs / dataset_name)
    config["experiments"][0]["data_file"] = dataset_name
    config["paths"] = {"user_input_dir": "inputs", "generated_dir": "generated", "output_dir": "outputs"}
    runtime_config = snapshot_inputs / "run_config.yaml"
    runtime_config.write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copy2(generated_script, snapshot_generated / "generated_script.py")
    user_model = generated_dir / "user_model.py"
    if user_model.exists():
        shutil.copy2(user_model, snapshot_generated / "user_model.py")
    events = generated_dir / "agent_logs" / "workflow_events.jsonl"
    if events.exists():
        (snapshot_generated / "agent_logs").mkdir()
        shutil.copy2(events, snapshot_generated / "agent_logs" / "workflow_events.jsonl")
    if initial_parameters is not None:
        import numpy as np
        np.savetxt(output_dir / "seed_design_point.csv", initial_parameters, delimiter=",")
    manifest = {
        "mode": "gradient-only" if gradient_only else "full",
        "source_run": source_dir.name if source_dir else None,
        "seed_source": str(source_dir.resolve()) if source_dir else None,
        "legacy_seed_order_assumed": bool(allow_legacy_seed and source_dir and not (source_dir / "final_parameters.json").exists()),
        "sloppiness_enabled": sloppiness, "sloppiness_method": sloppiness_method,
        "status": "running",
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Run directory: {output_dir}")
    try:
        if gradient_only:
            result = fit_gradient_only_system(
                runtime_config, output_dir, snapshot_generated, snapshot_dir, initial_parameters,
                sloppiness=sloppiness, sloppiness_method=sloppiness_method,
            )
        else:
            result = fit_generic_system(
                runtime_config, output_dir, snapshot_generated, snapshot_dir,
                sloppiness=sloppiness, sloppiness_method=sloppiness_method,
            )
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                        error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        manifest["status"] = "completed"
        return result
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def make_run_output_dir(session_dir: Path, run_id: str | None = None) -> Path:
    run_id = run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    from lib.utils.run_store import output_root
    return output_root(session_dir) / run_id


def select_session_dir(sessions_root: Path, argv: list[str]) -> Path:
    sessions_root = Path(sessions_root)
    if len(argv) == 2:
        session_dir = sessions_root / Path(argv[1])
        if not session_dir.is_dir():
            raise ValueError(f"Session directory {session_dir} does not exist")
        return session_dir

    if not sessions_root.exists() or not any(sessions_root.iterdir()):
        raise ValueError(
            "No session_dir provided and sessions/ is empty. "
            "Usage: python fit_parameters.py <session_dir>"
        )

    session_dirs = [path for path in sessions_root.iterdir() if path.is_dir()]
    session_dirs.sort(key=lambda path: path.stat().st_ctime, reverse=True)
    session_dir = session_dirs[0]
    print(f"No session_dir provided. Using most recently created session: {session_dir}")
    return session_dir


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv
    if not Path("sessions").is_dir():
        raise ValueError(
            "No sessions directory found. Please create a sessions directory "
            "and add a session subdirectory as described in the README."
        )

    try:
        session_dir = select_session_dir(Path("sessions"), argv)
    except ValueError as exc:
        if len(argv) == 1:
            print(str(exc))
            return 1
        raise

    from local_agent.agent.validators import parse_input_yaml

    input_file_path = Path(session_dir) / "inputs" / "user_input.yaml"
    input_reader = parse_input_yaml(input_file_path)
    run_driver(session_dir, input_reader)
    return 0
