import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


def run_driver(session_dir: Path, input_reader, output_dir_override: Path | None = None):
    """
    Execute the parameter fitting workflow for a session.

    Heavy numerical dependencies are imported lazily so validation, prompt
    rendering, and local LLM generation can run without a full JAX stack.
    """
    session_path = Path(session_dir)
    path_to_input = session_path / input_reader.user_input_dirname / "user_input.yaml"
    path_to_output_dir = (
        Path(output_dir_override)
        if output_dir_override is not None
        else session_path / input_reader.output_dirname
    )
    generated_dir = session_path / input_reader.generated_dirname
    generated_script = generated_dir / "generated_script.py"

    if not generated_script.exists():
        raise FileNotFoundError(
            f"generated_script.py not found at {generated_script}\n"
            "Run the existing generated-script creation step first. "
            "For local LLM orchestration, run: pfit jax <session_dir>."
        )

    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

    import jax

    from lib.utils.helper_functions import fit_generic_system

    print("Launching driver script")
    print("Available devices: ", jax.devices("cpu"))

    if path_to_output_dir.exists():
        shutil.rmtree(path_to_output_dir)
    path_to_output_dir.mkdir(parents=True)

    print("Launching fitting process...")
    fit_generic_system(path_to_input, path_to_output_dir, generated_dir, session_path)


def make_run_output_dir(session_dir: Path, run_id: str | None = None) -> Path:
    run_id = run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return Path(session_dir) / "outputs" / run_id


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

    from lib.utils.helper_functions import get_input_reader

    input_file_path = Path(session_dir) / "inputs" / "user_input.yaml"
    input_reader = get_input_reader(input_file_path)
    run_driver(session_dir, input_reader)
    return 0
